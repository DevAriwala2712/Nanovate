"""B2 acceptance test (NEXT_PATHS Track B2): fast single-level Delta-beta vs GUDHI on Manhattan VAL. Synthetic data.

Pre-registered in docs/NEXT_PATHS.md before this script existed: "identical Delta-beta to GUDHI on every validation
patch ... any verdict differs from GUDHI -> fix or drop". The level is fixed a priori at 0.5 (the conductor/dielectric
midpoint of the field); every level 0.05..0.95 is also scanned to ask whether ANY single level would pass (an
in-sample upper bound, not a selection). GUDHI Delta-beta is the frozen v2 path (results/v3/calib/decisions.json
pipeline_cfg) and is cross-checked against verify_patch's own reason on every patch. No test data is generated.

Writes results/b2/{rows.csv, summary.csv, config.json} (--quick: results/b2/quick/, git-ignored). Usage: python -m experiments.b2_fast_betti [--quick]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.v3_harness import _pc
from nanovate.config import TestbedCfg
from nanovate.modules.align_sdf import align_to_template
from nanovate.modules.calibration import calibrate, features
from nanovate.modules.fast_betti import fast_delta_beta
from nanovate.modules.fields import build_field
from nanovate.modules.topology import _reference, betti_from_intervals, classify_verdict, compute_persistence_intervals
from nanovate.pipeline import verify_patch
from nanovate.testbed.dataset import KINDS, get_layout, make_sample
from nanovate.testbed.scribe import make_scribe_patches
from nanovate.topo_utils import Window

OUT = Path("results/b2")
PRIMARY_LEVEL = 0.5
LEVELS = tuple(np.round(np.arange(0.05, 0.951, 0.05), 2))


def _cls(db) -> str:
    return classify_verdict(tuple(db)) if tuple(db) != (0, 0) else "ZERO"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    dec = json.loads(Path("results/v3/calib/decisions.json").read_text())
    pc = _pc(dec["pipeline_cfg"])
    tcfg, seed, win = TestbedCfg(), 0, Window(256, 16)
    cal = calibrate(make_scribe_patches(tcfg, seed=seed))
    layouts = range(3) if a.quick else range(24)
    rows = []
    for L in layouts:
        lay = get_layout("manhattan", "val", L, tcfg, seed)
        for die in (0, 1):
            for kind in KINDS:
                s = make_sample("manhattan", "val", L, die, kind, tcfg, seed, layout=lay)
                for noise_free in (False, True):
                    img = s.expected if noise_free else s.image
                    T = cal.predict(s.cad)
                    al, _ = align_to_template(img, T)
                    f, _ = build_field(al, option="template", levels=cal.levels, pre_smooth=pc.pre_smooth, template=T,
                                       cad_smooth=features(s.cad, *cal.sigmas)[..., 1], contrast=cal.mu_cond - cal.mu_diel)
                    reg, d0r, d1r = _reference(s.cad, win, pc.region_tol, pc.h1_fill)
                    t0 = time.perf_counter()
                    d0i, _ = compute_persistence_intervals(np.where(reg, f, s.cad.astype(float)))
                    d1i = compute_persistence_intervals(np.where(reg, f, 0.0))[1]
                    bi, br = betti_from_intervals(d0i, d1i, pc.tau), betti_from_intervals(d0r, d1r, pc.tau)
                    g = (bi[0] - br[0], bi[1] - br[1])
                    t_g = (time.perf_counter() - t0) * 1e3
                    vp = tuple(verify_patch(img, s.cad, cal, win, pc).reason["delta_betti"])
                    r = dict(layout=L, die=die, kind=kind, noise_free=noise_free, label=s.label, gudhi_d0=g[0], gudhi_d1=g[1],
                             gudhi_matches_verify_patch=(g == vp), ms_gudhi=t_g)
                    for lv in LEVELS:
                        t0 = time.perf_counter()
                        fb = fast_delta_beta(f, s.cad, reg, float(lv))
                        ms = (time.perf_counter() - t0) * 1e3
                        r[f"fast_{lv:.2f}_d0"], r[f"fast_{lv:.2f}_d1"] = fb
                        r[f"same_db_{lv:.2f}"] = fb == g
                        r[f"same_cls_{lv:.2f}"] = _cls(fb) == _cls(g)
                        if lv == PRIMARY_LEVEL:
                            r["ms_fast"] = ms
                    rows.append(r)
    df = pd.DataFrame(rows)
    out = OUT / "quick" if a.quick else OUT
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "rows.csv", index=False)
    summ = [dict(level=lv, primary=(lv == PRIMARY_LEVEL), n=len(df), identical_delta_beta=int(df[f"same_db_{lv:.2f}"].sum()),
                 identical_rate=float(df[f"same_db_{lv:.2f}"].mean()), same_verdict_class=int(df[f"same_cls_{lv:.2f}"].sum()),
                 passes_acceptance=bool(df[f"same_db_{lv:.2f}"].all())) for lv in LEVELS]
    s = pd.DataFrame(summ)
    s.to_csv(out / "summary.csv", index=False)
    cfg = dict(data="synthetic", split="val (Manhattan)", layouts=list(layouts), dies=[0, 1], kinds=list(KINDS), noise_free=[False, True],
               dose=tcfg.sem.dose, seed=seed, tau=pc.tau, primary_level=PRIMARY_LEVEL, levels=list(map(float, LEVELS)),
               pipeline_cfg=dec["pipeline_cfg"], acceptance="identical Delta-beta to GUDHI on every val patch",
               gudhi_vs_verify_patch_all_equal=bool(df.gudhi_matches_verify_patch.all()),
               ms_gudhi_median=float(df.ms_gudhi.median()), ms_fast_median=float(df.ms_fast.median()))
    (out / "config.json").write_text(json.dumps(cfg, indent=2))
    print(s.to_string()); print({k: cfg[k] for k in ("gudhi_vs_verify_patch_all_equal", "ms_gudhi_median", "ms_fast_median")})


if __name__ == "__main__":
    main()
