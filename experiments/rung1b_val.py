"""Rung 1b Part A: evaluate the three robustness candidates on synthetic Manhattan VAL and freeze (synthetic data).

Exactly the protocol in docs/REAL_DATA_RUNG1B_PREREG.md Part A (committed 4044b41 before this ran):
  patches  val layouts 0-23 x dies 0-1 x 6 kinds x doses {20, 50, 200} + noise-free dose 50 (1,152)
  shifts   the 288 dose-50 patches re-acquired at U(-8, 8) px per axis vs a zero-shift acquisition (no warp)
  variants frozen v3 | +C2 clamp | +C3 bounded alignment | +C2+C3; C1 guard as a trigger count over val cells
  accept   C1: 0 triggers. C2, C3: paired killer-F1 diff >= -0.01 with CI including 0 or above, and shift stability not
           lower by > 1 point. Frozen result -> results/real/rung1b/val/decisions.json.
Usage: python -m experiments.rung1b_val [--quick]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.common import f1_score
from experiments.v3_harness import _pc
from nanovate.config import TestbedCfg
from nanovate.modules.events import EventRuleCfg
from nanovate.pipeline_robust import RobustCfg, calibrate_robust, verify_patch_robust
from nanovate.testbed.dataset import KINDS, get_layout, make_sample
from nanovate.testbed.scribe import make_scribe_patches
from nanovate.topo_utils import Window

OUT = Path("results/real/rung1b/val")
VARIANTS = {"v3": RobustCfg(), "C2": RobustCfg(clamp=(0.5, 99.5)), "C3": RobustCfg(max_shift=12.0),
            "C2+C3": RobustCfg(clamp=(0.5, 99.5), max_shift=12.0)}
N_BOOT, SEED = 1000, 0


def tcfg_for(dose: float, ler: float = 1.0, grain: float = 0.15) -> TestbedCfg:
    b = TestbedCfg()
    return dataclasses.replace(b, sem=dataclasses.replace(b.sem, dose=float(dose)),
                               nuisance=dataclasses.replace(b.nuisance, ler_sigma=float(ler), grain_contrast=float(grain)))


def paired_f1_diff(pa, pb, y, n_boot=N_BOOT, seed=SEED):
    pa, pb, y = map(np.asarray, (pa, pb, y))
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(y), size=(n_boot, len(y)))
    d = np.array([f1_score(pa[i], y[i]) - f1_score(pb[i], y[i]) for i in idx])
    return float(f1_score(pa, y) - f1_score(pb, y)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    out = OUT / "quick" if a.quick else OUT
    out.mkdir(parents=True, exist_ok=True)
    dec = json.loads(Path("results/v3/calib/decisions.json").read_text())
    pc, rc, win = _pc(dec["pipeline_cfg"]), EventRuleCfg(**dec["event_rule_cfg"]), Window(256, 16)
    layouts = range(2) if a.quick else range(24)

    # C1 trigger count over val calibration cells (seed 0 scribe patches)
    trig = []
    for dose in ((50,) if a.quick else (5, 10, 20, 35, 50, 100, 200)):
        for ler in ((1.0,) if a.quick else (0.5, 1.0, 1.5, 2.0, 2.5)):
            for grain in ((0.15,) if a.quick else (0.0, 0.1, 0.15, 0.2, 0.3)):
                r = calibrate_robust(make_scribe_patches(tcfg_for(dose, ler, grain), seed=0), RobustCfg(guard=True))
                trig.append(dict(dose=dose, ler=ler, grain=grain, reliable=r.reliable, contrast=r.contrast,
                                 loo_min=min(r.loo_contrasts), loo_max=max(r.loo_contrasts)))
    trig = pd.DataFrame(trig)
    trig.to_csv(out / "c1_guard_cells.csv", index=False)

    cals = {}

    def rcal(dose, rcfg):
        k = (dose, rcfg.clamp)
        if k not in cals:
            cals[k] = calibrate_robust(make_scribe_patches(tcfg_for(dose), seed=0), rcfg)
        return cals[k]

    rows = []
    for L in layouts:
        for dose in (20, 50, 200):
            t = tcfg_for(dose)
            lay = get_layout("manhattan", "val", L, t, 0)
            for die in (0, 1):
                for kind in KINDS:
                    s = make_sample("manhattan", "val", L, die, kind, t, 0, layout=lay)
                    imgs = [("noisy", s.image)] + ([("noise_free", s.expected)] if dose == 50 else [])
                    for cond, img in imgs:
                        r = dict(layout=L, die=die, kind=kind, dose=dose, cond=cond, killer=s.label == "KILLER")
                        for n, rcfg in VARIANTS.items():
                            res = verify_patch_robust(img, s.cad, rcal(dose, rcfg), win, pc, rc, rcfg)
                            r[f"v_{n}"] = res.verdict
                            r[f"rej_{n}"] = res.reason["align_rejected"]
                        rows.append(r)
    df = pd.DataFrame(rows)
    df.to_csv(out / "rows.csv", index=False)

    rng = np.random.default_rng(SEED)
    srows = []
    t = tcfg_for(50)
    for L in layouts:
        lay = get_layout("manhattan", "val", L, t, 0)
        for die in (0, 1):
            for kind in KINDS:
                sh = tuple(float(v) for v in rng.uniform(-8, 8, 2))
                s0 = make_sample("manhattan", "val", L, die, kind, t, 0, layout=lay, shift=(0.0, 0.0), acq_tag="ref")
                s1 = make_sample("manhattan", "val", L, die, kind, t, 0, layout=lay, shift=sh, acq_tag="s0")
                r = dict(layout=L, die=die, kind=kind, shift_y=sh[0], shift_x=sh[1])
                for n, rcfg in VARIANTS.items():
                    v0 = verify_patch_robust(s0.image, s0.cad, rcal(50, rcfg), win, pc, rc, rcfg).verdict
                    v1 = verify_patch_robust(s1.image, s1.cad, rcal(50, rcfg), win, pc, rc, rcfg)
                    r[f"same_{n}"] = v0 == v1.verdict
                    r[f"rej_{n}"] = v1.reason["align_rejected"]
                srows.append(r)
    sd = pd.DataFrame(srows)
    sd.to_csv(out / "shift_rows.csv", index=False)

    y = df.killer.values
    kil = {n: df[f"v_{n}"].str.startswith("KILLER").values for n in VARIANTS}
    summ, accept = [], {}
    for n in VARIANTS:
        d, lo, hi = paired_f1_diff(kil[n], kil["v3"], y) if n != "v3" else (0.0, 0.0, 0.0)
        stab = float(sd[f"same_{n}"].mean())
        summ.append(dict(variant=n, killer_f1=f1_score(kil[n], y), f1_diff_vs_v3=d, ci_lo=lo, ci_hi=hi, n=len(y),
                         verdicts_changed_vs_v3=int((df[f"v_{n}"] != df["v_v3"]).sum()), align_rejected=int(df[f"rej_{n}"].sum()),
                         shift_stability=stab, shift_n=len(sd), shift_align_rejected=int(sd[f"rej_{n}"].sum())))
    s = pd.DataFrame(summ).set_index("variant")
    stab0 = s.loc["v3", "shift_stability"]
    for n in ("C2", "C3"):
        r = s.loc[n]
        accept[n] = bool(r.f1_diff_vs_v3 >= -0.01 and r.ci_hi >= 0 and r.shift_stability >= stab0 - 0.01)
    accept["C1"] = bool(trig.reliable.all())
    s.to_csv(out / "summary.csv")
    frozen = dict(data="synthetic", split="val (Manhattan)", prereg="docs/REAL_DATA_RUNG1B_PREREG.md Part A", accept=accept,
                  robust_cfg=RobustCfg(guard=accept["C1"], clamp=(0.5, 99.5) if accept["C2"] else None,
                                       max_shift=12.0 if accept["C3"] else None).to_dict(),
                  base=dict(pipeline_cfg=dec["pipeline_cfg"], event_rule_cfg=dec["event_rule_cfg"]),
                  c1_cells=len(trig), c1_triggers=int((~trig.reliable).sum()), quick=a.quick)
    (out / "decisions.json").write_text(json.dumps(frozen, indent=2))
    print(s.round(4).to_string()); print(json.dumps(accept)); print("C1 cells", len(trig), "triggers", int((~trig.reliable).sum()))


if __name__ == "__main__":
    main()
