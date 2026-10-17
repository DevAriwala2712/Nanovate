"""M4 measurements (synthetic data, Manhattan val, seed 0). Nothing here is tuned.

A. Near-miss separation (risk R3 dry run): for samples with d_beta = 0, how well do the bottleneck
   distance and the max SDF tension energy separate near-miss (notch, filament) from nuisance?
   Reported as AUC (0.5 = chance); Delta-beta outcomes per kind are printed too.
B. Experiment-4 preview: fraction of verdicts unchanged vs the zero-shift acquisition of the same
   wafer, for shifts of ~±8 px, with and without Module-3 alignment.
Fixed rule: calibrated field (Module-1 levels), pre_smooth 1, splice, region_tol 8, tau 0.45.
Usage: python scripts/align_sdf_report.py [--layouts 8] [--dies 2]
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

from nanovate.config import TestbedCfg, save_config
from nanovate.modules.align_sdf import align_to_template, max_tension_energy
from nanovate.modules.calibration import calibrate
from nanovate.modules.topology import verify_topology
from nanovate.testbed.dataset import get_layout, make_sample, window
from nanovate.testbed.scribe import make_scribe_patches

KINDS = ("nuisance", "notch", "filament", "bridge", "open", "particle")
SHIFTS = ((8.0, 0.0), (-8.0, 5.0), (3.5, -7.5), (-6.0, -6.0))


def auc(pos: np.ndarray, neg: np.ndarray) -> float:
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    return float((pos[:, None] > neg[None, :]).mean() + 0.5 * (pos[:, None] == neg[None, :]).mean())


def verdict(img, s, cal, win):
    return verify_topology(img, s.cad, win=win, tau=0.45, field_option="calibrated", levels=cal.levels,
                           pre_smooth=1.0, region_tol=8)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layouts", type=int, default=8)
    ap.add_argument("--dies", type=int, default=2)
    ap.add_argument("--out", default="results/m4_align_sdf")
    a = ap.parse_args()
    base = TestbedCfg()
    win = window(base)
    rowsA, rowsB = [], []
    for dose in (50, 200):
        cfg = dataclasses.replace(base, sem=dataclasses.replace(base.sem, dose=float(dose)))
        cal = calibrate(make_scribe_patches(cfg, seed=0))
        for lay in range(a.layouts):
            L = get_layout("manhattan", "val", lay, cfg, 0)
            for die in range(a.dies):
                for kind in KINDS:
                    s = make_sample("manhattan", "val", lay, die, kind, cfg, 0, layout=L)
                    aligned, d = align_to_template(s.image, cal.predict(s.cad))
                    v, r = verdict(aligned, s, cal, win)
                    e = max_tension_energy(s.image, s.cad, cal, win)
                    rowsA.append(dict(dose=dose, layout=lay, die=die, kind=kind, verdict=v,
                                      d_b0=r["delta_betti"][0], d_b1=r["delta_betti"][1],
                                      bottleneck=r["bottleneck_max"], energy=e, shift_err=max(abs(d[0] + s.shift[0]), abs(d[1] + s.shift[1]))))
                    if dose == 50 and kind in ("nuisance", "bridge", "open", "particle"):
                        s0 = make_sample("manhattan", "val", lay, die, kind, cfg, 0, layout=L, shift=(0.0, 0.0))
                        a0, _ = align_to_template(s0.image, cal.predict(s0.cad))
                        v0 = verdict(a0, s0, cal, win)[0]
                        for sh in SHIFTS:
                            ss = make_sample("manhattan", "val", lay, die, kind, cfg, 0, layout=L, shift=sh)
                            va = verdict(align_to_template(ss.image, cal.predict(ss.cad))[0], ss, cal, win)[0]
                            vn = verdict(ss.image, ss, cal, win)[0]
                            rowsB.append(dict(kind=kind, shift=str(sh), same_aligned=va == v0, same_unaligned=vn == verdict(s0.image, s0, cal, win)[0]))
        print(f"dose {dose} done", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    A, B = pd.DataFrame(rowsA), pd.DataFrame(rowsB)
    A.to_csv(out / "near_miss_rows.csv", index=False)
    B.to_csv(out / "shift_rows.csv", index=False)
    save_config({"testbed": dataclasses.asdict(base), "layouts": a.layouts, "dies": a.dies, "seed": 0,
                 "gamma": 0.2, "cap": 10.0, "smooth": 1.0, "rule": "calibrated levels, smooth 1, splice, tol 8, tau .45"}, out / "config.json")
    n = a.layouts * a.dies
    print(f"\n== A. d_beta outcomes by kind (n = {n} per kind/dose)")
    print(A.groupby(["dose", "kind"]).apply(lambda g: pd.Series({"dbeta0": (g.d_b0 == 0).mean() * (g.d_b1 == 0).mean() if False else ((g.d_b0 == 0) & (g.d_b1 == 0)).mean(),
                                                                "median_bottleneck": g.bottleneck.median(), "median_energy": g.energy.median(),
                                                                "median_shift_err_px": g.shift_err.median()}), include_groups=False).round(3).to_string())
    print("\n== A. AUC among d_beta = 0 samples (near-miss = notch|filament vs nuisance)")
    for dose in (50, 200):
        z = A[(A.dose == dose) & (A.d_b0 == 0) & (A.d_b1 == 0)]
        nm, nu = z[z.kind.isin(["notch", "filament"])], z[z.kind == "nuisance"]
        print(f"dose {dose}: n near-miss={len(nm)}, n nuisance={len(nu)}  AUC bottleneck={auc(nm.bottleneck.values, nu.bottleneck.values):.2f}  AUC energy={auc(nm.energy.values, nu.energy.values):.2f}")
    print(f"\n== B. verdict unchanged vs zero shift (dose 50, n = {len(B) // 2} per mode; shifts ~±8 px)")
    print(B.groupby("kind")[["same_aligned", "same_unaligned"]].mean().round(2).to_string())
    print("overall", B[["same_aligned", "same_unaligned"]].mean().round(3).to_dict())


if __name__ == "__main__":
    main()
