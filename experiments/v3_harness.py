"""v3 evaluation context: the v2 harness with (1) a FRESH test seed and (2) the frozen v3 event rule. Synthetic data.

* Test samples, their layouts and their die-to-die reference dies are keyed by ``test_seed`` (no earlier version
  generated test data with it). Val data, scribe patches (Module 1), die-to-die rules and the CNN threshold keep
  seed 0, exactly as frozen before the test run.
* The verifier column ``verdict`` is the frozen PRIMARY v3 rule (results/v3/calib/decisions.json). Two extra columns
  come from the same evidence of the same patch: ``verdict_sec`` (the pre-declared secondary variant,
  results/v3/calib/secondary.json) and ``verdict_v2rule`` (v2's Delta-beta ladder with v2's frozen thresholds),
  so the comparison is paired. Every evaluated row is also kept in ``self.log`` (including Exp 2's noise-free ceiling).
Nothing here selects a parameter.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np

from experiments.common import ExpCfg
from experiments.harness import Context
from experiments.v3_select import v2_pc
from nanovate.baselines.cnn import predict_proba
from nanovate.baselines.die_to_die import d2d_flag, d2d_score
from nanovate.config import TestbedCfg
from nanovate.modules.events import EventRuleCfg, decide
from nanovate.modules.topology import classify_verdict
from nanovate.pipeline import PipelineCfg
from nanovate.pipeline_v3 import patch_evidence
from nanovate.testbed.dataset import get_layout, make_sample


def _pc(d: dict) -> PipelineCfg:
    d = {k: (float("inf") if v is None and k in ("d_star", "e_star") else float("-inf") if v is None and k == "z_star" else v) for k, v in d.items()}
    return PipelineCfg(**d)


class V3Context(Context):
    SMOKE_SEED_OFFSET = 7919   # --quick runs use test_seed + this, never the official fresh test seed

    def __init__(self, cfg: ExpCfg, calib_dir: str | Path = "results/v3/calib"):
        dec = json.loads((Path(calib_dir) / "decisions.json").read_text())
        sec = json.loads((Path(calib_dir) / "secondary.json").read_text())
        super().__init__(cfg, _pc(dec["pipeline_cfg"]))
        self.rc = EventRuleCfg(**dec["event_rule_cfg"])
        self.rc_sec = EventRuleCfg(**sec["event_rule_cfg"])
        self.pc_v2 = v2_pc()
        self.test_seed = int(dec["test_seed"]) + (self.SMOKE_SEED_OFFSET if cfg.quick else 0)
        self.log: list[dict] = []

    def split_seed(self, split: str) -> int:
        return self.test_seed if split == "test" else self.cfg.seed

    def layout(self, family, split, layout_id, tcfg):
        key = (family, split, layout_id)
        if key not in self._lay:
            self._lay[key] = get_layout(family, split, layout_id, tcfg, self.split_seed(split))
        return self._lay[key]

    def refs(self, family, split, layout_id, tcfg, layout=None):
        key = (family, split, layout_id, tcfg.sem.dose, tcfg.nuisance.ler_sigma, tcfg.nuisance.grain_contrast)
        if key not in self._refs:
            self._refs[key] = [make_sample(family, split, layout_id, 900 + j, "nuisance", tcfg, self.split_seed(split), layout=layout).image
                               for j in range(2)]
        return self._refs[key]

    def evaluate(self, family: str, split: str, layout_id: int, die: int, kind: str, tcfg: TestbedCfg, *,
                 shift=None, warp: float = 0.0, acq_tag: str = "", layout=None, rule_dose: float | None = None,
                 methods: tuple = ("verifier", "d2d", "cnn"), pc: PipelineCfg | None = None, tag: str = "",
                 noise_free: bool = False) -> dict:
        seed = self.split_seed(split)
        L = layout or self.layout(family, split, layout_id, tcfg)
        s = make_sample(family, split, layout_id, die, kind, tcfg, seed, layout=L, shift=shift, acq_tag=acq_tag, warp_amp=warp)
        row = dict(family=family, split=split, layout=layout_id, die=die, kind=kind, label=s.label, killer=s.killer,
                   dose=tcfg.sem.dose, ler_sigma=tcfg.nuisance.ler_sigma, grain=tcfg.nuisance.grain_contrast,
                   shift_y=s.shift[0], shift_x=s.shift[1], warp=warp, resamples=s.resamples, tag=tag, noise_free=noise_free, seed=seed)
        for k, v in s.params.get("defect", {}).items():
            if isinstance(v, (int, float)):
                row[f"p_{k}"] = float(v)
        cal = self.cal(tcfg)
        if "verifier" in methods:
            p = pc or self.pc
            ev, sc, t = patch_evidence(s.expected if noise_free else s.image, s.cad, cal, self.win, p, self.rc)
            t0 = time.perf_counter()
            v, events = decide(ev, self.rc, p.tau, sc["bottleneck"], sc["energy"], sc["peak_z"], p.d_star, p.e_star, p.z_star)
            t["topology"] += (time.perf_counter() - t0) * 1e3
            v_sec, _ = decide(ev, self.rc_sec, p.tau, sc["bottleneck"], sc["energy"], sc["peak_z"], p.d_star, p.e_star, p.z_star)
            q = self.pc_v2
            v_v2 = classify_verdict(sc["delta_betti_v2"], sc["bottleneck"], sc["energy"], q.d_star, q.e_star, sc["peak_z"], q.z_star)
            row.update(verdict=v, verdict_sec=v_sec, verdict_v2rule=v_v2, n_bridge_events=len(events["bridge"]), n_open_events=len(events["open"]),
                       d_b0=sc["delta_betti_v2"][0], d_b1=sc["delta_betti_v2"][1], bottleneck=sc["bottleneck"], energy=sc["energy"],
                       peak_z=sc["peak_z"], ms_align=t["align"], ms_energy=t["sdf_energy"], ms_topology=t["topology"], singleton=True)
        if "d2d" in methods:
            rules = self.d2d_rules(rule_dose if rule_dose is not None else tcfg.sem.dose)
            refs = self.refs(family, split, layout_id, tcfg, L)
            t0 = time.perf_counter()
            row["d2d_single"] = d2d_flag(d2d_score(s.image, refs[:1], L.cad, cal, "single"), rules["single"], self.win)[0]
            row["ms_d2d"] = (time.perf_counter() - t0) * 1e3
            row["d2d_double"] = d2d_flag(d2d_score(s.image, refs, L.cad, cal, "double"), rules["double"], self.win)[0]
        if "cnn" in methods:
            t0 = time.perf_counter()
            pr = float(predict_proba(self.cnn, s.image[None], s.cad[None], 1)[0])
            row["ms_cnn"] = (time.perf_counter() - t0) * 1e3
            row["cnn_prob"], row["cnn"] = pr, pr >= self.cnn_thr
        self.log.append(row)
        return row


__all__ = ["V3Context", "np"]
