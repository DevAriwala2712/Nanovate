"""M2 early-risk probe: false-killer rate and killer recall vs dose, field option, tau.

Manhattan *val* only (AGENTS.md §5.2: tau is never tuned on test or on families B/C).
Persistence is computed once per (sample, field, pre_smooth); the tau sweep re-uses the
intervals. Calibrated field uses ORACLE levels (Module 1 not built yet) -- reported as such.
Synthetic data. Usage: python scripts/probe_m2.py [--layouts 8] [--dies 2] [--seed 0]
"""
from __future__ import annotations

import argparse
import dataclasses
from pathlib import Path

import numpy as np
import pandas as pd

from nanovate.config import TestbedCfg, save_config
from nanovate.modules.fields import build_field
from nanovate.modules.topology import (
    KILLER_BRIDGE, KILLER_OPEN_PARTICLE, NUISANCE,
    betti_from_intervals, classify_verdict, compute_persistence_intervals,
)
from nanovate.testbed.dataset import get_layout, make_sample, window
from nanovate.topo_utils import relevance_region

DOSES = (5, 10, 20, 50, 100, 200)
TAUS = tuple(round(t, 2) for t in np.arange(0.30, 0.701, 0.05))
KINDS = ("nuisance", "bridge", "open", "particle")
EXPECT = {"nuisance": NUISANCE, "bridge": KILLER_BRIDGE,
          "open": KILLER_OPEN_PARTICLE, "particle": KILLER_OPEN_PARTICLE}
FIELDS = (("percentile", 0.0), ("percentile", 1.0), ("calibrated", 0.0), ("calibrated", 1.0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layouts", type=int, default=8)
    ap.add_argument("--dies", type=int, default=2)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="results/m2_probe")
    a = ap.parse_args()

    base = TestbedCfg()
    win = window(base)
    rows = []
    for lay in range(a.layouts):
        layout = get_layout("manhattan", "val", lay, base, a.seed)
        reg = relevance_region(layout.cad, win)
        cadf = layout.cad.astype(float)
        d0c, d1c = compute_persistence_intervals(cadf)
        ref = {t: betti_from_intervals(d0c, d1c, t) for t in TAUS}
        for dose in DOSES:
            cfg = dataclasses.replace(base, sem=dataclasses.replace(base.sem, dose=float(dose)))
            for die in range(a.dies):
                for kind in KINDS:
                    s = make_sample("manhattan", "val", lay, die, kind, cfg, a.seed, layout=layout)
                    for opt, sm in FIELDS:
                        f, _ = build_field(s.image, opt, sem_cfg=cfg.sem, pre_smooth=sm)
                        d0, d1 = compute_persistence_intervals(np.where(reg, f, cadf))
                        for t in TAUS:
                            b = betti_from_intervals(d0, d1, t)
                            db = (b[0] - ref[t][0], b[1] - ref[t][1])
                            v = classify_verdict(db)
                            rows.append((lay, die, dose, kind, opt, sm, t, db[0], db[1], v, v == EXPECT[kind]))
    df = pd.DataFrame(rows, columns=["layout", "die", "dose", "kind", "field", "pre_smooth", "tau",
                                     "d_b0", "d_b1", "verdict", "correct"])
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "probe.csv", index=False)
    save_config({"testbed": dataclasses.asdict(base), "doses": DOSES, "taus": TAUS, "layouts": a.layouts,
                 "dies": a.dies, "seed": a.seed, "levels": "oracle", "split": "val", "family": "manhattan"},
                out / "config.json")

    nui = df[df.kind == "nuisance"].assign(false_killer=lambda d: d.verdict != NUISANCE)
    kil = df[df.kind != "nuisance"]
    fk = nui.groupby(["field", "pre_smooth", "tau", "dose"]).false_killer.mean().rename("false_killer")
    rc = kil.groupby(["field", "pre_smooth", "tau", "dose"]).correct.mean().rename("killer_correct")
    summ = pd.concat([fk, rc], axis=1).reset_index()
    summ["n_nuis"] = a.layouts * a.dies
    summ["n_kill"] = a.layouts * a.dies * 3
    summ.to_csv(out / "summary.csv", index=False)
    print(f"n per cell: nuisance={a.layouts * a.dies}, killer={a.layouts * a.dies * 3}")
    for (f, sm), g in summ.groupby(["field", "pre_smooth"]):
        print(f"\n== field={f} pre_smooth={sm}: false_killer / killer_correct, rows=tau cols=dose")
        print(g.pivot(index="tau", columns="dose", values="false_killer").round(2).to_string())
        print(g.pivot(index="tau", columns="dose", values="killer_correct").round(2).to_string())


if __name__ == "__main__":
    main()
