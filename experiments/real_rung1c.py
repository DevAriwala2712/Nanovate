"""Track C Rung 1c: ONE real repeat with the near-miss gate re-selected under synthetic edge-halo cells (REAL DATA).

Also reused by real_rung1d (summarise).

Same populations, references, calibrations and labels as Rung 1b (experiments/real_rung1b.py); the only change is
(d*, e*, z*) from results/real/rung1c/val/decisions.json. Paired tile by tile with Rung 1b. Pre-registration:
docs/REAL_DATA_RUNG1C_PREREG.md. Refuses to run unless committed, or twice. Usage: python -m experiments.real_rung1c --run
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import sys
import time
from pathlib import Path

import pandas as pd

from experiments import real_rung1b as R
from experiments.real_rung1 import _cluster_ci, _git, build, frozen_cfg
from nanovate.pipeline_robust import RobustCfg

OUT = Path("results/real/rung1c")
PREREG = Path("docs/REAL_DATA_RUNG1C_PREREG.md")
FROZEN = OUT / "val" / "decisions.json"
RUN1B_TILES = Path("results/real/rung1b/tiles.csv")


def check_prereg() -> str:
    for f in (PREREG, FROZEN):
        if not _git("log", "-1", "--format=%H", "--", str(f)):
            sys.exit(f"refusing to run: {f} is not committed")
        if _git("status", "--porcelain", "--", str(f)):
            sys.exit(f"refusing to run: {f} has uncommitted changes")
    if (OUT / "RUN_DONE").exists():
        sys.exit(f"refusing to run: {OUT / 'RUN_DONE'} exists (Rung 1c runs once)")
    return _git("rev-parse", "HEAD")


def summarise(t: pd.DataFrame) -> pd.DataFrame:
    t = t.assign(near_miss=t.verdict == "NEAR_MISS", nuisance=t.verdict == "NUISANCE")
    rows = []
    for pop, g in t.groupby("population"):
        for scope, b in [(pop, g)] + [(f"{pop}/{k}", gg) for k, gg in g.groupby("group")]:
            b = b[b.clean & ~b.overlay]
            for col, name in (("near_miss", "NEAR_MISS on clean tiles"), ("killer", "KILLER on clean tiles"),
                              ("verified", "coverage"), ("nuisance", "NUISANCE on clean tiles")):
                m, lo, hi, n, ni = _cluster_ci(b, col)
                rows.append(dict(population=scope, metric=name, mean=m, ci_lo=lo, ci_hi=hi, n_tiles=n, n_images=ni))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", required=True)
    ap.parse_args()
    commit = check_prereg()
    pc, rc = frozen_cfg()
    dec = json.loads(FROZEN.read_text())
    nm = dec["near_miss"]
    if not dec["feasible"]:
        pc = dataclasses.replace(pc, d_star=float("inf"), e_star=float("inf"), z_star=float("-inf"))
    else:
        pc = dataclasses.replace(pc, d_star=nm["d_star"], e_star=nm["e_star"], z_star=nm["z_star"])
    b = dec["base"]["robust_cfg"]
    rcfg = RobustCfg(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in b.items()})
    lab1 = pd.read_csv("results/real/rung1/clean_labels.csv")
    labf = pd.read_csv(R.LABELS)
    labels = {**dict(zip(lab1.image_id, lab1.label)), **dict(zip(labf.image_id, labf.label))}
    t0 = time.time()
    orig = [g for g in build("C")[0] if not g["name"].startswith("INGAN")]
    to, co = R.run(orig, {"primary": rcfg}, labels, pc, rc)
    tf, cf = R.run([R.fresh_group()], {"primary": rcfg}, labels, pc, rc)
    to["population"], tf["population"] = "orig", "fresh"
    t = pd.concat([to, tf])
    p1b = pd.read_csv(RUN1B_TILES)
    p1b = p1b[p1b.variant == "primary"][["population", "image_id", "tile", "verdict"]].rename(columns={"verdict": "verdict_rung1b"})
    t = t.merge(p1b, on=["population", "image_id", "tile"], how="left")
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "tiles.csv", index=False)
    pd.concat([co, cf]).to_csv(OUT / "calibration.csv", index=False)
    s = summarise(t)
    s.to_csv(OUT / "summary.csv", index=False)
    (OUT / "config.json").write_text(json.dumps(dict(data="REAL SEM images (Track C Rung 1c)", reference="FITTED, not CAD", prereg=str(PREREG),
                                                     prereg_commit=commit, near_miss=dict(d_star=pc.d_star, e_star=pc.e_star, z_star=pc.z_star),
                                                     robust_cfg=rcfg.to_dict(), wall_s=time.time() - t0), indent=2, default=str))
    (OUT / "RUN_DONE").write_text(json.dumps(dict(commit=commit, finished=time.strftime("%Y-%m-%d %H:%M:%S")), indent=2))
    print(s.round(4).to_string())


if __name__ == "__main__":
    main()
