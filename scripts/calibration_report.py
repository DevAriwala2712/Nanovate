"""M3 measurements for Module 1 (synthetic data): transfer error, noise model, timing, M2 effect.

* transfer error  ||T(C) - E[I]|| / ||E[I]|| on held-out in-die *val* patches per family
  (E[I] = noise-free expected image at shift 0, includes the charging ramp; the patch's LER,
  grain and the scribe-vs-die ramp are the genuine transfer risk). Reported with registration
  on/off and with per-patch ramp removal on/off.
* noise model alpha / sigma2 vs the simulator's (alpha = 1 e-/count, read variance 1 e-^2).
* calibration time (ms), 3 scribe patches, CPU.
* M2 effect: Manhattan-val verdicts with oracle vs calibrated levels (fixed rule: calibrated field,
  pre_smooth 1, splice, region_tol 8, tau 0.45 -- NOT tuned here).
Usage: python scripts/calibration_report.py [--seeds 3] [--per-family 6]
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

from nanovate.config import TestbedCfg, save_config
from nanovate.modules.calibration import calibrate, remove_ramp
from nanovate.modules.fields import oracle_levels
from nanovate.modules.topology import NUISANCE, KILLER_BRIDGE, KILLER_OPEN_PARTICLE, verify_topology
from nanovate.testbed.dataset import get_layout, make_sample, window
from nanovate.testbed.layouts import FAMILIES
from nanovate.testbed.scribe import make_scribe_patches

DOSES = (20, 50, 100, 200)


def rel(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b) / np.linalg.norm(b))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--per-family", type=int, default=6)
    ap.add_argument("--out", default="results/m3_calibration")
    a = ap.parse_args()
    base = TestbedCfg()
    win = window(base)
    rows, m2 = [], []
    for dose in DOSES:
        cfg = dataclasses.replace(base, sem=dataclasses.replace(base.sem, dose=float(dose)))
        for seed in range(a.seeds):
            patches = make_scribe_patches(cfg, seed=seed)
            cals = {reg: calibrate(patches, register=reg) for reg in (True, False)}
            cal = cals[True]
            orc = oracle_levels(cfg.sem)
            rows.append(dict(kind="levels", dose=dose, seed=seed, family="-", register=True,
                             rel_err_mu_diel=cal.mu_diel / orc[0] - 1, rel_err_mu_cond=cal.mu_cond / orc[1] - 1,
                             alpha=cal.alpha, sigma2=cal.sigma2, timing_ms=cal.timing_ms,
                             scribe_shift_max=max(max(abs(s[0]), abs(s[1])) for s in cal.shifts)))
            for fam in FAMILIES:
                for lay in range(a.per_family):
                    s = make_sample(fam, "val", lay, 0, "nuisance", cfg, 100 + seed, shift=(0.0, 0.0))
                    for reg, c in cals.items():
                        T = c.predict(s.cad)
                        E = s.expected.astype(float)
                        rows.append(dict(kind="transfer", dose=dose, seed=seed, family=fam, register=reg,
                                         rel_err=rel(T, E), rel_err_ramp_removed=rel(T, remove_ramp(E, T))))
            if seed == 0:  # M2 effect, Manhattan val only
                for lay in range(8):
                    lay_obj = get_layout("manhattan", "val", lay, cfg, 0)
                    for die in range(2):
                        for k in ("nuisance", "bridge", "open", "particle"):
                            s = make_sample("manhattan", "val", lay, die, k, cfg, 0, layout=lay_obj)
                            for name, lv in (("oracle", orc), ("calibrated", cal.levels)):
                                v, _ = verify_topology(s.image, s.cad, win=win, tau=0.45, field_option="calibrated",
                                                       levels=lv, pre_smooth=1.0, region_tol=8)
                                exp = {"nuisance": NUISANCE, "bridge": KILLER_BRIDGE}.get(k, KILLER_OPEN_PARTICLE)
                                m2.append(dict(dose=dose, levels=name, kind=k, correct=v == exp, verdict=v))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / "rows.csv", index=False)
    m2d = pd.DataFrame(m2)
    m2d.to_csv(out / "m2_effect_rows.csv", index=False)
    save_config({"testbed": dataclasses.asdict(base), "doses": DOSES, "seeds": a.seeds,
                 "per_family": a.per_family, "split": "val (transfer, M2 effect)"}, out / "config.json")
    lv = df[df.kind == "levels"]
    print("== levels / noise model / timing (mean over seeds; n =", a.seeds, "per dose)")
    print(lv.groupby("dose")[["rel_err_mu_diel", "rel_err_mu_cond", "alpha", "sigma2", "timing_ms", "scribe_shift_max"]].mean().round(3).to_string())
    tr = df[df.kind == "transfer"]
    print("\n== transfer rel. error ||T-E||/||E|| (mean; n =", a.seeds * a.per_family, "patches per cell)")
    print(tr.groupby(["family", "register", "dose"])[["rel_err", "rel_err_ramp_removed"]].mean().round(3).unstack("dose").to_string())
    print("\n== M2 effect (false_killer = nuisance not NUISANCE; killer_correct), n = 16 nuisance / 48 killer per cell")
    g = m2d.assign(nui=m2d.kind == "nuisance")
    fk = g[g.nui].assign(fk=lambda d: ~d.correct).groupby(["levels", "dose"]).fk.mean()
    kc = g[~g.nui].groupby(["levels", "dose"]).correct.mean()
    print(pd.concat([fk.rename("false_killer"), kc.rename("killer_correct")], axis=1).round(2).to_string())


if __name__ == "__main__":
    main()
