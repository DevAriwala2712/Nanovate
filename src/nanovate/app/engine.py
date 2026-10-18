"""Analysis engine of the demonstrator app: calls the research core exactly as the experiments do.

The frozen artefacts come from ``experiments.harness.Context(ExpCfg.full())`` (the object every v2 experiment
uses): the verifier ``PipelineCfg`` from results/v2/calib/decisions.json, the CNN weights m6_cnn/cnn.pt with the
probability threshold from results/m6_baselines/rules.json, and the per-dose die-to-die rules fitted on
Manhattan val (seed 0). The app never refits a threshold differently and never changes these.
Per input: Module-1 calibration (cached per layer id) -> ``verify_patch`` -> CNN ``predict_proba`` ->
``d2d_score`` / ``d2d_flag``. Overlays (conductor field, SDF energy) are recomputed with the same calls the
pipeline and ``nanovate.demo.walk_through`` use; they are for display and never feed a verdict.
"""
from __future__ import annotations

import dataclasses
import hashlib
import json
import math
import subprocess
import threading
import time
from collections import OrderedDict
from pathlib import Path

import numpy as np

from experiments.common import ExpCfg
from experiments.harness import CNN_PATH, Context
from ..baselines.cnn import predict_proba
from ..baselines.die_to_die import D2DRule, d2d_flag, d2d_score
from ..config import TestbedCfg
from ..modules.align_sdf import align_to_template, energy_map
from ..modules.calibration import Calibration, calibrate, features
from ..modules.fields import build_field
from ..pipeline import verify_patch
from ..testbed.dataset import window
from .sources import DOSE_GRID, InputError, InspectionInput

REPO_FILES = ("results/v2/calib/decisions.json", "results/m6_baselines/rules.json", CNN_PATH, "experiments/harness.py")


def check_repo_root(root: Path = Path(".")) -> None:
    """The app reads frozen artefacts by relative path, as the experiments do; refuse to start elsewhere."""
    missing = [f for f in REPO_FILES if not (root / f).exists()]
    if missing:
        raise SystemExit(f"nanovate.app must run from the repository root (missing: {', '.join(missing)}); cd there and retry.")


def _sha256_file(path: str | Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def git_commit() -> str:
    """HEAD commit, with ``-dirty`` when the working tree has uncommitted changes; 'unknown' outside git."""
    try:
        head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, timeout=5, check=True).stdout.strip()
        dirty = subprocess.run(["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, timeout=5).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except Exception:  # noqa: BLE001 (no git, not a repo)
        return "unknown"


def finite(x):
    """JSON-safe copy: numpy scalars -> python, +-inf / nan -> None, tuples -> lists."""
    if isinstance(x, dict):
        return {k: finite(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [finite(v) for v in x]
    if isinstance(x, (np.bool_, bool)):
        return bool(x)
    if isinstance(x, (np.integer,)):
        return int(x)
    if isinstance(x, (float, np.floating)):
        return None if not math.isfinite(float(x)) else float(x)
    return x


def code_fingerprint() -> str:
    """sha256 over the research code a die-to-die rule fit depends on (invalidates the on-disk rule cache)."""
    h = hashlib.sha256()
    files = sorted(p for p in Path("src/nanovate").rglob("*.py") if "app" not in p.parts) + \
        [Path("experiments/harness.py"), Path("experiments/select_config.py"), Path("experiments/common.py")]
    for p in files:
        h.update(str(p).encode()); h.update(p.read_bytes())
    return h.hexdigest()[:16]


class Engine:
    """Holds the frozen methods, the caches and one analysis lock (GUDHI / torch, honest timings)."""

    def __init__(self, rule_cache: str | Path | None = "results/app_cache/d2d_rules.json", max_inputs: int = 64):
        check_repo_root()
        self.exp = ExpCfg.full()
        self.ctx = Context(self.exp)
        self.pc = self.ctx.pc
        self.tcfg = TestbedCfg()
        self.win = window(self.tcfg)
        self.commit = git_commit()
        self.cnn_sha256 = _sha256_file(CNN_PATH)
        self.config = {
            "pipeline_cfg": finite(self.pc.to_dict()),
            "pipeline_cfg_source": "results/v2/calib/decisions.json",
            "testbed_cfg": dataclasses.asdict(self.tcfg),
            "cnn": {"weights": CNN_PATH, "sha256": self.cnn_sha256, "threshold": self.ctx.cnn_thr,
                    "threshold_source": "results/m6_baselines/rules.json"},
            "d2d_fit": {"procedure": "experiments.harness.Context.d2d_rules (Manhattan val, per dose)", "seed": self.exp.seed,
                        "val_layouts": self.exp.val_layouts, "val_dies": self.exp.val_dies, "version": self.exp.version},
        }
        self.config_hash = hashlib.sha256(json.dumps(self.config, sort_keys=True).encode()).hexdigest()[:16]
        self.rule_cache = Path(rule_cache) if rule_cache else None
        self._rule_key = f"{self.config_hash}-{code_fingerprint()}"
        self._load_rule_cache()
        self.inputs: OrderedDict[str, InspectionInput] = OrderedDict()
        self.max_inputs = max_inputs
        self.cals: dict[str, tuple[Calibration, str]] = {}   # layer_id -> (calibration, scribe sha256)
        self.lock = threading.Lock()

    # ---- inputs ---------------------------------------------------------------------
    def put(self, inp: InspectionInput) -> str:
        self.inputs[inp.input_id] = inp
        self.inputs.move_to_end(inp.input_id)
        while len(self.inputs) > self.max_inputs:
            self.inputs.popitem(last=False)
        return inp.input_id

    def get(self, input_id: str) -> InspectionInput:
        if input_id not in self.inputs:
            raise InputError("unknown_input_id", f"no input with id {input_id!r} (inputs live in memory; simulate or upload again)",
                             "input_id", status=404)
        self.inputs.move_to_end(input_id)
        return self.inputs[input_id]

    # ---- Module 1: calibration cached per layer id ----------------------------------
    def calibration(self, inp: InspectionInput) -> tuple[Calibration, dict]:
        ref = inp.calibration
        if ref.scribe:
            h = hashlib.sha256()
            for cad, img in ref.scribe:
                h.update(np.ascontiguousarray(cad, dtype=bool).tobytes()); h.update(np.ascontiguousarray(img, dtype=np.float32).tobytes())
            key, sha = inp.meta.layer_id, h.hexdigest()
            hit = self.cals.get(key)
            if hit is not None and hit[1] == sha:
                return hit[0], {"layer_id": key, "ms": hit[0].timing_ms, "cached": True}
            cal = calibrate([(c, i) for c, i in ref.scribe])
            replaced = hit is not None
            self.cals[key] = (cal, sha)
            return cal, {"layer_id": key, "ms": cal.timing_ms, "cached": False, "replaced_previous": replaced}
        cid = ref.calibration_id
        if not cid:
            raise InputError("calibration_missing", "Upload 3 scribe patches or name an existing calibration id; "
                             "the verifier will not guess the conductor/dielectric levels.", "calibration")
        if cid not in self.cals:
            raise InputError("unknown_calibration_id", f"calibration id {cid!r} is not cached (known: {sorted(self.cals)})", "calibration_id")
        cal = self.cals[cid][0]
        return cal, {"layer_id": cid, "ms": cal.timing_ms, "cached": True}

    # ---- die-to-die rules: Context.d2d_rules, cached on disk -------------------------
    def _load_rule_cache(self) -> None:
        if not (self.rule_cache and self.rule_cache.exists()):
            return
        try:
            blob = json.loads(self.rule_cache.read_text()).get(self._rule_key, {})
        except (OSError, ValueError):
            return
        for dose, rules in blob.items():
            self.ctx._rules[float(dose)] = {m: D2DRule(**r) for m, r in rules.items()}

    def d2d_rules(self, dose: float) -> tuple[dict[str, D2DRule], bool]:
        """(rules, was_cached). A miss fits the rule exactly as the experiments do (~20 s on CPU)."""
        dose = float(dose)
        cached = dose in self.ctx._rules
        rules = self.ctx.d2d_rules(dose)
        if not cached and self.rule_cache:
            self.rule_cache.parent.mkdir(parents=True, exist_ok=True)
            try:
                blob = json.loads(self.rule_cache.read_text()) if self.rule_cache.exists() else {}
            except ValueError:
                blob = {}
            blob = {self._rule_key: blob.get(self._rule_key, {})}          # drop entries of other configs / code
            blob[self._rule_key][f"{dose:g}"] = {m: dataclasses.asdict(r) for m, r in rules.items()}
            self.rule_cache.write_text(json.dumps(blob, indent=1, sort_keys=True))
        return rules, cached

    def rules_cached(self) -> list[float]:
        return sorted(self.ctx._rules)

    # ---- one analysis ---------------------------------------------------------------
    def analyse(self, inp: InspectionInput) -> dict:
        """All three methods on one input, plus display overlays. Arrays are returned as numpy (the API encodes them)."""
        with self.lock:
            cal, cal_info = self.calibration(inp)
            img, cad, pc, win = inp.image, inp.cad, self.pc, self.win

            r = verify_patch(img, cad, cal, win, pc)
            verifier = {"verdict": r.verdict, "reason": r.reason, "singleton": r.singleton,
                        "aligner_shift": list(r.shift), "recovered_shift": [-r.shift[0], -r.shift[1]], "timings_ms": dict(r.timings_ms)}

            t0 = time.perf_counter()
            p = float(predict_proba(self.ctx.cnn, img[None], cad[None], 1)[0])
            cnn = {"prob": p, "threshold": self.ctx.cnn_thr, "flagged": bool(p >= self.ctx.cnn_thr), "ms": (time.perf_counter() - t0) * 1e3,
                   "domain_note": None if inp.meta.source_type == "simulated"
                   else "trained on simulated Manhattan tiles; this input is outside its training domain"}

            d2d = self._d2d(inp, cal)

            template = cal.predict(cad)
            aligned, _ = align_to_template(img, template)
            kw = {}
            if pc.field_option == "template":
                kw = dict(template=template, cad_smooth=features(cad, *cal.sigmas)[..., 1], contrast=cal.mu_cond - cal.mu_diel)
            field, _ = build_field(aligned, pc.field_option, levels=cal.levels, pre_smooth=pc.pre_smooth, **kw)
            emap = energy_map(img, cad, cal, win, pc.gamma, pc.cap, aligned=aligned)
        return {"verifier": verifier, "cnn": cnn, "d2d": d2d, "calibration": cal_info,
                "levels": {"mu_diel": cal.mu_diel, "mu_cond": cal.mu_cond},
                "overlays": {"field": field, "sdf_energy": emap, "window": [win.margin, win.size - win.margin]}}

    def _d2d(self, inp: InspectionInput, cal: Calibration) -> dict:
        dose = inp.meta.dose
        if dose is None or float(dose) not in DOSE_GRID:
            why = "no rule fitted at this dose" if dose is not None else "dose unknown, so no rule applies"
            return {"available": False, "why": f"not available: {why} (rules exist for {list(DOSE_GRID)} e-/px)"}
        refs = list(inp.references)
        if not refs:
            return {"available": False, "why": "not available: no reference die"}
        rules, cached = self.d2d_rules(dose)
        out: dict = {"available": True, "rule_dose": float(dose), "rule_cached": cached}
        t0 = time.perf_counter()
        f, area = d2d_flag(d2d_score(inp.image, refs[:1], inp.cad, cal, "single"), rules["single"], self.win)
        out["single"] = {"flagged": bool(f), "max_area": int(area), "rule": dataclasses.asdict(rules["single"]),
                         "ms": (time.perf_counter() - t0) * 1e3}
        if len(refs) >= 2:
            t0 = time.perf_counter()
            f, area = d2d_flag(d2d_score(inp.image, refs[:2], inp.cad, cal, "double"), rules["double"], self.win)
            out["double"] = {"flagged": bool(f), "max_area": int(area), "rule": dataclasses.asdict(rules["double"]),
                             "ms": (time.perf_counter() - t0) * 1e3}
        else:
            out["double"] = {"available": False, "why": "not available: double difference needs two reference dies"}
        return out
