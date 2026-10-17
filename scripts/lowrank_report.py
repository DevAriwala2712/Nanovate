"""M5 measurements: does the Module-2 stack field beat per-patch smoothing for Module 4? (synthetic, Manhattan val)

Each layout gets a K = 32 stack: 26 clean dies + 2 bridge + 2 open + 2 particle (each die has its own
shift, LER, grain, noise). Verdicts use the Module-1 calibrated levels and the fixed M2 rule (splice,
region_tol 8, tau 0.45) -- nothing tuned. Fields compared (all after Module-3 alignment):
  smooth1   : aligned image, Gaussian pre_smooth 1 px (the M2/M4 baseline)
  medgauss  : singleton fallback (median 3x3 + Gaussian 1 px)
  VARIANTS : L + S from the stack (soft_k2 = spec default; others are ablations)
Evaluated dies per layout: 8 clean + all 6 defective (false-killer n = 8/layout, killer n = 6/layout).
Usage: python scripts/lowrank_report.py [--layouts 6] [--doses 20 100]
"""
from __future__ import annotations

import argparse
import dataclasses
import time
from pathlib import Path

import numpy as np
import pandas as pd

from nanovate.config import TestbedCfg, save_config
from nanovate.modules.align_sdf import align_to_template
from nanovate.modules.calibration import calibrate
from nanovate.modules.lowrank import denoise_singleton, denoise_stack
from nanovate.modules.topology import KILLER_BRIDGE, KILLER_OPEN_PARTICLE, NUISANCE, verify_topology
from nanovate.testbed.dataset import get_layout, make_sample, window
from nanovate.testbed.scribe import make_scribe_patches

KINDS = ["nuisance"] * 26 + ["bridge"] * 2 + ["open"] * 2 + ["particle"] * 2
# soft_k2 is the spec default; the others are ablations (shrinkage / residual smoothing)
VARIANTS = {
    "soft_k2 (spec)": dict(k=2.0),
    "hard_k3": dict(k=3.0, shrink="hard"),
    "soft_k2_rs1": dict(k=2.0, resid_smooth=1.0),
    "soft_k3_rs1": dict(k=3.0, resid_smooth=1.0),
    "hard_k3_rs1": dict(k=3.0, shrink="hard", resid_smooth=1.0),
    "hard_k4_rs1": dict(k=4.0, shrink="hard", resid_smooth=1.0),
    "support_k2_rs1": dict(k=2.0, shrink="support", resid_smooth=1.0),
    "support_k3_rs1": dict(k=3.0, shrink="support", resid_smooth=1.0),
}
EVAL = list(range(8)) + list(range(26, 32))
EXPECT = {"nuisance": NUISANCE, "bridge": KILLER_BRIDGE, "open": KILLER_OPEN_PARTICLE, "particle": KILLER_OPEN_PARTICLE}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layouts", type=int, default=6)
    ap.add_argument("--doses", type=int, nargs="+", default=[20, 100])
    ap.add_argument("--out", default="results/m5_lowrank")
    a = ap.parse_args()
    base = TestbedCfg()
    win = window(base)
    rows, timing = [], []
    for dose in a.doses:
        cfg = dataclasses.replace(base, sem=dataclasses.replace(base.sem, dose=float(dose)))
        cal = calibrate(make_scribe_patches(cfg, seed=0))
        for lay in range(a.layouts):
            L = get_layout("manhattan", "val", lay, cfg, 0)
            S = [make_sample("manhattan", "val", lay, k, kd, cfg, 0, layout=L) for k, kd in enumerate(KINDS)]
            T = cal.predict(L.cad)
            imgs = [s.image for s in S]
            aligned = [align_to_template(i, T)[0] for i in imgs]
            fields = {"smooth1": {k: aligned[k] for k in EVAL}, "medgauss": {k: denoise_singleton(aligned[k]) for k in EVAL}}
            for name, kw in VARIANTS.items():
                t0 = time.perf_counter()
                res = denoise_stack(imgs, L.cad, cal, seed=0, **kw)
                timing.append(dict(dose=dose, variant=name, ms_per_patch=(time.perf_counter() - t0) * 1e3 / len(imgs)))
                fields[name] = {k: res.field[k] for k in EVAL}
            for method, fd in fields.items():
                for k in EVAL:
                    v, r = verify_topology(fd[k], S[k].cad, win=win, tau=0.45, field_option="calibrated", levels=cal.levels,
                                           pre_smooth=1.0 if method == "smooth1" else 0.0, region_tol=8)
                    rows.append(dict(dose=dose, layout=lay, die=k, kind=KINDS[k], method=method, verdict=v,
                                     correct=v == EXPECT[KINDS[k]], d_b0=r["delta_betti"][0], d_b1=r["delta_betti"][1]))
        print(f"dose {dose} done", flush=True)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "rows.csv", index=False)
    pd.DataFrame(timing).to_csv(out / "timing.csv", index=False)
    save_config({"testbed": dataclasses.asdict(base), "layouts": a.layouts, "doses": a.doses, "kinds": KINDS,
                 "eval_dies": EVAL, "rule": "calibrated levels, splice, region_tol 8, tau .45"}, out / "config.json")
    nui = df[df.kind == "nuisance"].assign(false_killer=lambda d: ~d.correct)
    kil = df[df.kind != "nuisance"]
    tab = pd.concat([nui.groupby(["method", "dose"]).false_killer.mean(), kil.groupby(["method", "dose"]).correct.mean().rename("killer_correct"),
                     *[kil[kil.kind == k].groupby(["method", "dose"]).correct.mean().rename(f"{k}_correct") for k in ("bridge", "open", "particle")]],
                    axis=1).round(2)
    print(f"\nn per cell: nuisance {a.layouts * 8}, killer {a.layouts * 6} (bridge/open/particle {a.layouts * 2} each)")
    print(tab.to_string())
    print("\nstack timing ms/patch (K = 32, SVD + threshold + alignment, CPU):")
    print(pd.DataFrame(timing).groupby(["dose", "variant"]).ms_per_patch.mean().round(1).to_string())


if __name__ == "__main__":
    main()
