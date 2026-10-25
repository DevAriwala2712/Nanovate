"""Track C real-wiring false-alarm test (M2): the frozen verifier (Rung 1d state) on real metal-2 SEM images against the
authors' validated track/via labels (REAL DATA; doi:10.17617/3.HY5SYN, CC-BY-4.0). Pre-registration: docs/REAL_DATA_M2_PREREG.md.

Arms: native (14.65 nm/px, primary) and x0.5 (29.3 nm/px; tracks ~ testbed width, pre-declared secondary).
  --dry   manifest + label-QC statistics (no verifier).  --run   the one pre-registered run (guarded).
Usage: python -m experiments.real_m2 --dry | --run [--workers N]
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from skimage.transform import rescale

from experiments.real_rung1 import _cluster_ci, _git, frozen_cfg
from nanovate.pipeline_robust import RobustCfg, calibrate_robust, verify_patch_robust
from nanovate.real.m2 import PX_NM, ROOT, ids, load_image, load_label, unlabelled_conductor
from nanovate.topo_utils import Window

OUT = Path("results/real/m2")
PREREG = Path("docs/REAL_DATA_M2_PREREG.md")
FROZEN_1D = Path("results/real/rung1d/val/decisions.json")
CAL_INDEX = (0, 160, 320)
ARMS = {"native": 1.0, "half": 0.5}
TILE = 256


def frozen():
    pc, rc = frozen_cfg()
    dec = json.loads(FROZEN_1D.read_text())
    th = dec["thresholds"][dec["choice"]]
    pc = dataclasses.replace(pc, d_star=th["d_star"], e_star=th["e_star"], z_star=th["z_star"])
    rcfg = RobustCfg(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in dec["robust_cfg"].items()})
    return pc, rc, rcfg


def arm_arrays(i: str, s: float):
    """(image, label raster, QC mask) at scale s (x0.5: image area-averaged, masks by > 0.5 of the averaged raster)."""
    img = load_image(i)
    lab = load_label(i, img.shape)
    q = unlabelled_conductor(img, lab)
    if s != 1.0:
        img = rescale(img, s, order=1, anti_aliasing=True, preserve_range=True)
        lab = rescale(lab.astype(float), s, order=1, anti_aliasing=True, preserve_range=True) > 0.5
        q = rescale(q.astype(float), s, order=1, anti_aliasing=True, preserve_range=True) > 0.0
    return img, lab, q


def grid(shape):
    return [(r, c) for r in range(shape[0] // TILE) for c in range(shape[1] // TILE)]


def crop(a, rc):
    return a[rc[0] * TILE:(rc[0] + 1) * TILE, rc[1] * TILE:(rc[1] + 1) * TILE]


def central(shape):
    return (shape[0] // TILE) // 2, (shape[1] // TILE) // 2


_CAL: dict = {}


def _calibration(arm: str):
    if arm not in _CAL:
        pc, rc, rcfg = frozen()
        I = ids()
        P = []
        for k in CAL_INDEX:
            img, lab, _ = arm_arrays(I[k], ARMS[arm])
            rcx = central(img.shape)
            P.append((crop(lab, rcx).astype(float), crop(img, rcx)))
        _CAL[arm] = calibrate_robust(P, rcfg)
    return _CAL[arm]


def _image_task(spec: dict) -> list[dict]:
    pc, rc, rcfg = frozen()
    arm, i = spec["arm"], spec["id"]
    rcal = _calibration(arm)
    img, lab, q = arm_arrays(i, ARMS[arm])
    win = Window(TILE, 16)
    rows = []
    for rcx in grid(img.shape):
        cad = crop(lab, rcx)
        if not cad.any() or cad.all():
            rows.append(dict(arm=arm, image_id=i, tile=f"{rcx[0]},{rcx[1]}", qc_flag=bool(crop(q, rcx).any()), skipped="one-class label tile"))
            continue
        t0 = time.perf_counter()
        r = verify_patch_robust(crop(img, rcx), cad, rcal, win, pc, rc, rcfg)
        z = r.reason
        rows.append(dict(arm=arm, image_id=i, tile=f"{rcx[0]},{rcx[1]}", qc_flag=bool(crop(q, rcx).any()), skipped="",
                         label_frac=float(cad.mean()), verdict=r.verdict, killer=r.verdict.startswith("KILLER"),
                         near_miss=r.verdict == "NEAR_MISS", verified=r.verdict != "UNVERIFIED_CALIBRATION",
                         d_beta0=z["delta_betti"][0], d_beta1=z["delta_betti"][1], n_bridge_events=z["n_bridge_events"],
                         n_open_events=z["n_open_events"], bottleneck=z["bottleneck_max"], max_sdf_energy=z["max_sdf_energy"],
                         peak_z=z["peak_z"], align_rejected=z["align_rejected"], m3_shift_px=float(np.hypot(*r.shift)),
                         ms=(time.perf_counter() - t0) * 1e3))
    return rows


def check_prereg() -> str:
    for f in (PREREG,):
        if not _git("log", "-1", "--format=%H", "--", str(f)):
            sys.exit(f"refusing to run: {f} is not committed")
        if _git("status", "--porcelain", "--", str(f)):
            sys.exit(f"refusing to run: {f} has uncommitted changes")
    if (OUT / "RUN_DONE").exists():
        sys.exit(f"refusing to run: {OUT / 'RUN_DONE'} exists (runs once)")
    return _git("rev-parse", "HEAD")


def summarise(t: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for arm, g in t[t.skipped == ""].groupby("arm"):
        for scope, b in (("label-QC passed (primary)", g[~g.qc_flag]), ("label-QC flagged", g[g.qc_flag]), ("all tiles", g)):
            for col, name in (("killer", "KILLER"), ("near_miss", "NEAR_MISS"), ("verified", "coverage")):
                m, lo, hi, n, ni = _cluster_ci(b.assign(**{col: b[col].astype(bool)}), col)
                rows.append(dict(arm=arm, scope=scope, metric=name, mean=m, ci_lo=lo, ci_hi=hi, n_tiles=n, n_images=ni))
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry", action="store_true")
    mode.add_argument("--run", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    I = ids()
    cal_ids = [I[k] for k in CAL_INDEX]
    if a.dry:
        man = []
        for i in I:
            p = ROOT / "sems" / "m2" / f"sem{i}.png"
            img, lab, q = arm_arrays(i, 1.0)
            g = grid(img.shape)
            man.append(dict(image_id=i, role="calibration" if i in cal_ids else "evaluation", file=str(p),
                            sha256=hashlib.sha256(p.read_bytes()).hexdigest(), px_nm=PX_NM, px_source="dataset README (authors)",
                            height_px=img.shape[0], width_px=img.shape[1], label_frac=float(lab.mean()), qc_px=int(q.sum()),
                            qc_tiles=int(sum(crop(q, rc).any() for rc in g)), tiles=len(g), detector="BSE", kv=15, dwell_us=3,
                            doi="10.17617/3.HY5SYN", licence="CC-BY-4.0", data_kind="REAL SEM image"))
        m = pd.DataFrame(man)
        m.to_csv(OUT / "manifest.csv", index=False)
        print(m[["qc_px", "qc_tiles", "label_frac"]].describe().round(3).to_string())
        print("eval tiles", int(m[m.role == "evaluation"].tiles.sum()), "QC-flagged", int(m[m.role == "evaluation"].qc_tiles.sum()))
        return
    commit = check_prereg()
    t0 = time.time()
    specs = [dict(arm=arm, id=i) for arm in ARMS for i in I if i not in cal_ids]
    with ProcessPoolExecutor(a.workers) as ex:
        t = pd.DataFrame([r for rows in ex.map(_image_task, specs, chunksize=2) for r in rows])
    t.to_csv(OUT / "tiles.csv", index=False)
    s = summarise(t)
    s.to_csv(OUT / "summary.csv", index=False)
    cals = {arm: dict(reliable=_calibration(arm).reliable, contrast=_calibration(arm).contrast, loo=list(_calibration(arm).loo_contrasts),
                      clamp=_calibration(arm).clamp_range) for arm in ARMS}
    pc, rc, rcfg = frozen()
    (OUT / "config.json").write_text(json.dumps(dict(data="REAL SEM images (M2 real-wiring test)", reference="authors' validated labels (not designer CAD)",
                                                     prereg=str(PREREG), prereg_commit=commit, arms=ARMS, cal_ids=cal_ids, calibration=cals,
                                                     near_miss=dict(d_star=pc.d_star, e_star=pc.e_star, z_star=pc.z_star),
                                                     robust_cfg=rcfg.to_dict(), wall_s=time.time() - t0), indent=2, default=str))
    (OUT / "RUN_DONE").write_text(json.dumps(dict(commit=commit, finished=time.strftime("%Y-%m-%d %H:%M:%S")), indent=2))
    print(json.dumps(cals, default=str)); print(s.round(4).to_string())


if __name__ == "__main__":
    main()
