"""M2: splice vs region-local pairs (and region_tol 4 vs 8) on the Manhattan-val probe set.

Same samples/seed as scripts/probe_m2.py. Manhattan *val* only; oracle levels (Module 1 not
built); synthetic data. Output: results/m2_region_modes/{rows.csv, summary.csv, config.json}.
Usage: python scripts/compare_region_modes.py [--layouts 8] [--dies 2] [--seed 0]
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

from nanovate.config import TestbedCfg, save_config
from nanovate.modules.fields import build_field
from nanovate.modules.localpers import local_persistence
from nanovate.modules.topology import (
    KILLER_BRIDGE, KILLER_OPEN_PARTICLE, NUISANCE,
    betti_from_intervals, classify_verdict, compute_persistence_intervals,
)
from nanovate.testbed.dataset import get_layout, make_sample, window
from nanovate.topo_utils import relevance_region

DOSES = (20, 50, 100, 200)
TAUS = tuple(round(t, 2) for t in np.arange(0.30, 0.701, 0.05))
TOLS = (4, 8)
KINDS = ("nuisance", "bridge", "open", "particle")
EXPECT = {"nuisance": NUISANCE, "bridge": KILLER_BRIDGE,
          "open": KILLER_OPEN_PARTICLE, "particle": KILLER_OPEN_PARTICLE}
FIELDS = (("calibrated", 0.0), ("calibrated", 1.0), ("percentile", 1.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layouts", type=int, default=8)
    ap.add_argument("--dies", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/m2_region_modes")
    a = ap.parse_args()

    base = TestbedCfg()
    win = window(base)
    rows = []
    for lay in range(a.layouts):
        layout = get_layout("manhattan", "val", lay, base, a.seed)
        cadf = layout.cad.astype(float)
        regs = {tol: relevance_region(layout.cad, win, tol=tol) for tol in TOLS}
        ref_local = {tol: local_persistence(cadf, regs[tol]) for tol in TOLS}
        d0c, d1c = compute_persistence_intervals(cadf)
        ref_splice = {t: betti_from_intervals(d0c, d1c, t) for t in TAUS}
        for dose in DOSES:
            cfg = dataclasses.replace(base, sem=dataclasses.replace(base.sem, dose=float(dose)))
            for die in range(a.dies):
                for kind in KINDS:
                    s = make_sample("manhattan", "val", lay, die, kind, cfg, a.seed, layout=layout)
                    for opt, sm in FIELDS:
                        f, _ = build_field(s.image, opt, sem_cfg=cfg.sem, pre_smooth=sm)
                        for tol in TOLS:
                            lp = local_persistence(f, regs[tol])
                            d0, d1 = compute_persistence_intervals(np.where(regs[tol], f, cadf))
                            for t in TAUS:
                                for mode in ("pairs", "splice"):
                                    if mode == "pairs":
                                        b, r = lp.betti(t), ref_local[tol].betti(t)
                                    else:
                                        b, r = betti_from_intervals(d0, d1, t), ref_splice[t]
                                    db = (b[0] - r[0], b[1] - r[1])
                                    v = classify_verdict(db)
                                    rows.append((lay, die, dose, kind, opt, sm, tol, mode, t,
                                                 db[0], db[1], v, v == EXPECT[kind]))
        print(f"layout {lay} done", flush=True)
    df = pd.DataFrame(rows, columns=["layout", "die", "dose", "kind", "field", "pre_smooth", "region_tol",
                                     "mode", "tau", "d_b0", "d_b1", "verdict", "correct"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "rows.csv", index=False)
    save_config({"testbed": dataclasses.asdict(base), "doses": DOSES, "taus": TAUS, "tols": TOLS,
                 "layouts": a.layouts, "dies": a.dies, "seed": a.seed, "levels": "oracle",
                 "split": "val", "family": "manhattan"}, out / "config.json")
    key = ["mode", "region_tol", "field", "pre_smooth", "tau", "dose"]
    nui = df[df.kind == "nuisance"].assign(false_killer=lambda d: d.verdict != NUISANCE)
    kil = df[df.kind != "nuisance"]
    summ = pd.concat([nui.groupby(key).false_killer.mean(), kil.groupby(key).correct.mean().rename("killer_correct"),
                      kil[kil.kind == "bridge"].groupby(key).correct.mean().rename("bridge_recall")], axis=1).reset_index()
    summ["n_nuis"] = a.layouts * a.dies
    summ["n_kill"] = a.layouts * a.dies * 3
    summ.to_csv(out / "summary.csv", index=False)
    print("saved", out)


if __name__ == "__main__":
    main()
