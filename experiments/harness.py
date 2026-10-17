"""Evaluation harness shared by experiments 1-6: one row per sample with every method's decision.

Methods: the topology verifier (frozen ``PipelineCfg`` from results/calib/decisions.json), die-to-die
single and double difference (per-dose rules fitted on Manhattan val at the nominal nuisance), and the
Manhattan-trained CNN (weights m6_cnn/cnn.pt, probability threshold chosen on Manhattan val in M6).
Nothing here selects a parameter on test data: rules and thresholds are loaded or fitted on val.
Synthetic data throughout.
"""
from __future__ import annotations

import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.common import ExpCfg
from experiments.select_config import load_pipeline_cfg, results_root, with_dose
from nanovate.baselines.cnn import load_cnn, predict_proba
from nanovate.baselines.die_to_die import D2DRule, d2d_flag, d2d_score, d2d_stats, fit_d2d_rule
from nanovate.config import TestbedCfg
from nanovate.modules.calibration import Calibration, calibrate
from nanovate.pipeline import PipelineCfg, verify_patch
from nanovate.testbed.dataset import KINDS, get_layout, make_sample, window
from nanovate.testbed.scribe import make_scribe_patches

KILLER_KINDS = ("bridge", "open", "particle")
SIGMAS, THETAS = (0.0, 1.0, 2.0), (3.0, 4.0, 5.0, 6.0, 8.0, 10.0, 12.0)
AREAS = (2, 4, 6, 8, 12, 16, 24, 32, 48, 64)
CNN_PATH, CNN_RULES = "m6_cnn/cnn.pt", "results/m6_baselines/rules.json"


def nuisance_cfg(base: TestbedCfg, ler_sigma: float | None = None, grain: float | None = None) -> TestbedCfg:
    n = base.nuisance
    n = dataclasses.replace(n, ler_sigma=n.ler_sigma if ler_sigma is None else ler_sigma,
                            grain_contrast=n.grain_contrast if grain is None else grain)
    return dataclasses.replace(base, nuisance=n)


class Context:
    """Caches calibrations, reference dies, D2D rules and the CNN for one experiment run."""

    def __init__(self, cfg: ExpCfg, pc: PipelineCfg | None = None):
        self.cfg = cfg
        self.pc = pc or load_pipeline_cfg(cfg)
        self.base = TestbedCfg()
        self.win = window(self.base)
        self._cal: dict = {}
        self._refs: dict = {}
        self._rules: dict = {}
        self._lay: dict = {}
        self.cnn, self.cnn_cfg, _ = load_cnn(CNN_PATH)
        self.cnn_thr = float(json.loads(Path(CNN_RULES).read_text())["cnn_threshold"])

    # ---- Module 1 -----------------------------------------------------------------
    def cal(self, tcfg: TestbedCfg) -> Calibration:
        key = (tcfg.sem.dose, tcfg.nuisance.ler_sigma, tcfg.nuisance.grain_contrast)
        if key not in self._cal:
            self._cal[key] = calibrate(make_scribe_patches(tcfg, seed=self.cfg.seed))
        return self._cal[key]

    def layout(self, family: str, split: str, layout_id: int, tcfg: TestbedCfg):
        key = (family, split, layout_id)
        if key not in self._lay:
            self._lay[key] = get_layout(family, split, layout_id, tcfg, self.cfg.seed)
        return self._lay[key]

    # ---- die-to-die references ----------------------------------------------------
    def refs(self, family: str, split: str, layout_id: int, tcfg: TestbedCfg, layout=None) -> list[np.ndarray]:
        key = (family, split, layout_id, tcfg.sem.dose, tcfg.nuisance.ler_sigma, tcfg.nuisance.grain_contrast)
        if key not in self._refs:
            self._refs[key] = [make_sample(family, split, layout_id, 900 + j, "nuisance", tcfg, self.cfg.seed, layout=layout).image
                               for j in range(2)]
        return self._refs[key]

    # ---- die-to-die rules: per dose, Manhattan val, nominal nuisance ---------------
    def d2d_rules(self, dose: float) -> dict[str, D2DRule]:
        if dose in self._rules:
            return self._rules[dose]
        tcfg = with_dose(self.base, dose)
        cal = self.cal(tcfg)
        st = {"single": [], "double": []}
        y = []
        for lay in range(self.cfg.val_layouts):
            L = get_layout("manhattan", "val", lay, tcfg, self.cfg.seed)
            refs = self.refs("manhattan", "val", lay, tcfg, L)
            for die in range(self.cfg.val_dies):
                for kind in KINDS:
                    s = make_sample("manhattan", "val", lay, die, kind, tcfg, self.cfg.seed, layout=L)
                    st["single"].append(d2d_stats(d2d_score(s.image, refs[:1], L.cad, cal, "single"), self.win, SIGMAS, THETAS))
                    st["double"].append(d2d_stats(d2d_score(s.image, refs, L.cad, cal, "double"), self.win, SIGMAS, THETAS))
                    y.append(s.killer)
        self._rules[dose] = {m: fit_d2d_rule(st[m], y, "val", "manhattan", m, SIGMAS, THETAS, AREAS) for m in st}
        return self._rules[dose]

    # ---- one sample, all methods ---------------------------------------------------
    def evaluate(self, family: str, split: str, layout_id: int, die: int, kind: str, tcfg: TestbedCfg, *,
                 shift=None, warp: float = 0.0, acq_tag: str = "", layout=None, rule_dose: float | None = None,
                 methods: tuple = ("verifier", "d2d", "cnn"), pc: PipelineCfg | None = None, tag: str = "",
                 noise_free: bool = False) -> dict:
        """Generate one sample and run the requested methods. ``rule_dose`` picks the D2D rule (default: sample dose).

        ``noise_free`` feeds the verifier the noise-free expected image (same blur, bloom, ramp and shift; no
        Poisson / read noise): a diagnostic of the verifier's ceiling at this optical resolution, not a result.
        """
        seed = self.cfg.seed
        L = layout or self.layout(family, split, layout_id, tcfg)
        s = make_sample(family, split, layout_id, die, kind, tcfg, seed, layout=L, shift=shift, acq_tag=acq_tag, warp_amp=warp)
        row = dict(family=family, split=split, layout=layout_id, die=die, kind=kind, label=s.label, killer=s.killer,
                   dose=tcfg.sem.dose, ler_sigma=tcfg.nuisance.ler_sigma, grain=tcfg.nuisance.grain_contrast,
                   shift_y=s.shift[0], shift_x=s.shift[1], warp=warp, resamples=s.resamples, tag=tag)
        for k, v in s.params.get("defect", {}).items():
            if isinstance(v, (int, float)):
                row[f"p_{k}"] = float(v)
        cal = self.cal(tcfg)
        if "verifier" in methods:
            r = verify_patch(s.expected if noise_free else s.image, s.cad, cal, self.win, pc or self.pc)
            row.update(verdict=r.verdict, d_b0=r.reason["delta_betti"][0], d_b1=r.reason["delta_betti"][1],
                       bottleneck=r.reason["bottleneck_max"], energy=r.reason["max_sdf_energy"],
                       ms_align=r.timings_ms["align"], ms_energy=r.timings_ms["sdf_energy"], ms_topology=r.timings_ms["topology"],
                       singleton=r.singleton)
        if "d2d" in methods:
            rules = self.d2d_rules(rule_dose if rule_dose is not None else tcfg.sem.dose)
            refs = self.refs(family, split, layout_id, tcfg, L)
            t0 = time.perf_counter()
            row["d2d_single"] = d2d_flag(d2d_score(s.image, refs[:1], L.cad, cal, "single"), rules["single"], self.win)[0]
            row["ms_d2d"] = (time.perf_counter() - t0) * 1e3
            row["d2d_double"] = d2d_flag(d2d_score(s.image, refs, L.cad, cal, "double"), rules["double"], self.win)[0]
        if "cnn" in methods:
            t0 = time.perf_counter()
            p = float(predict_proba(self.cnn, s.image[None], s.cad[None], 1)[0])
            row["ms_cnn"] = (time.perf_counter() - t0) * 1e3
            row["cnn_prob"], row["cnn"] = p, p >= self.cnn_thr
        return row

    def rows(self, specs: list[dict], progress: str = "") -> pd.DataFrame:
        out = []
        for i, sp in enumerate(specs):
            out.append(self.evaluate(**sp))
            if progress and (i + 1) % 100 == 0:
                print(f"  {progress}: {i + 1}/{len(specs)}", flush=True)
        return pd.DataFrame(out)


def killer_flags(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Boolean 'flagged as killer' per method from an evaluated frame."""
    out = {}
    if "verdict" in df:
        out["verifier"] = df.verdict.str.startswith("KILLER").values
    for m in ("d2d_single", "d2d_double", "cnn"):
        if m in df:
            out[m] = df[m].values.astype(bool)
    return out


def not_flagged(df: pd.DataFrame) -> dict[str, np.ndarray]:
    """Boolean 'suppressed' per method (verifier: verdict == NUISANCE; baselines: not flagged)."""
    out = {m: ~v for m, v in killer_flags(df).items()}
    if "verdict" in df:
        out["verifier"] = (df.verdict == "NUISANCE").values
    return out
