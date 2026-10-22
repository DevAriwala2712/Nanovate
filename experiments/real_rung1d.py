"""Track C Rung 1d: ONE real repeat with the variant and near-miss gate frozen on synthetic CD-offset val (REAL DATA).

Same populations, references, calibrations and labels as Rungs 1b/1c. The robust config (incl. cd_correct) and
(d*, e*, z*) come from results/real/rung1d/val/decisions.json. Paired tile by tile with Rung 1c. Pre-registration:
docs/REAL_DATA_RUNG1D_PREREG.md. Refuses to run unless committed, or twice. Usage: python -m experiments.real_rung1d --run
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
from experiments.real_rung1 import _git, build, frozen_cfg
from experiments.real_rung1c import summarise
from nanovate.pipeline_robust import RobustCfg

OUT = Path("results/real/rung1d")
PREREG = Path("docs/REAL_DATA_RUNG1D_PREREG.md")
FROZEN = OUT / "val" / "decisions.json"
RUN1C_TILES = Path("results/real/rung1c/tiles.csv")


def check_prereg() -> str:
    for f in (PREREG, FROZEN):
        if not _git("log", "-1", "--format=%H", "--", str(f)):
            sys.exit(f"refusing to run: {f} is not committed")
        if _git("status", "--porcelain", "--", str(f)):
            sys.exit(f"refusing to run: {f} has uncommitted changes")
    if (OUT / "RUN_DONE").exists():
        sys.exit(f"refusing to run: {OUT / 'RUN_DONE'} exists (Rung 1d runs once)")
    return _git("rev-parse", "HEAD")


def frozen_robust(dec: dict) -> RobustCfg:
    return RobustCfg(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in dec["robust_cfg"].items()})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", action="store_true", required=True)
    ap.parse_args()
    commit = check_prereg()
    pc, rc = frozen_cfg()
    dec = json.loads(FROZEN.read_text())
    th = dec["thresholds"][dec["choice"]]
    inf = float("inf")
    pc = dataclasses.replace(pc, d_star=inf if th["d_star"] is None else th["d_star"], e_star=inf if th["e_star"] is None else th["e_star"],
                             z_star=-inf if th["z_star"] is None else th["z_star"])
    rcfg = frozen_robust(dec)
    lab1 = pd.read_csv("results/real/rung1/clean_labels.csv")
    labf = pd.read_csv(R.LABELS)
    labels = {**dict(zip(lab1.image_id, lab1.label)), **dict(zip(labf.image_id, labf.label))}
    t0 = time.time()
    orig = [g for g in build("C")[0] if not g["name"].startswith("INGAN")]
    to, co = R.run(orig, {"primary": rcfg}, labels, pc, rc)
    tf, cf = R.run([R.fresh_group()], {"primary": rcfg}, labels, pc, rc)
    to["population"], tf["population"] = "orig", "fresh"
    t = pd.concat([to, tf])
    p1c = pd.read_csv(RUN1C_TILES)[["population", "image_id", "tile", "verdict"]].rename(columns={"verdict": "verdict_rung1c"})
    t = t.merge(p1c, on=["population", "image_id", "tile"], how="left")
    OUT.mkdir(parents=True, exist_ok=True)
    t.to_csv(OUT / "tiles.csv", index=False)
    pd.concat([co, cf]).to_csv(OUT / "calibration.csv", index=False)
    s = summarise(t)
    s.to_csv(OUT / "summary.csv", index=False)
    (OUT / "config.json").write_text(json.dumps(dict(data="REAL SEM images (Track C Rung 1d)", reference="FITTED, not CAD", prereg=str(PREREG),
                                                     prereg_commit=commit, choice=dec["choice"],
                                                     near_miss=dict(d_star=pc.d_star, e_star=pc.e_star, z_star=pc.z_star),
                                                     robust_cfg=rcfg.to_dict(), wall_s=time.time() - t0), indent=2, default=str))
    (OUT / "RUN_DONE").write_text(json.dumps(dict(commit=commit, finished=time.strftime("%Y-%m-%d %H:%M:%S")), indent=2))
    print(pd.concat([co, cf]).to_string()); print(s.round(4).to_string())


if __name__ == "__main__":
    main()
