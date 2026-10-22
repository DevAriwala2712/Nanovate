"""Track C Rung 1b: ONE repeat of the real-image false-alarm test with the Rung 1b robustness changes (REAL DATA).

Pre-registration: docs/REAL_DATA_RUNG1B_PREREG.md (Part A: candidates chosen on synthetic val, frozen in
results/real/rung1b/val/decisions.json; Part B: this run). Refuses to run unless both are committed, or twice.

Populations (arm C, native pixels only):
  orig   the Rung 1 tiles (G29 + G01), same FITTED references, calibration tiles and clean labels -> paired with Rung 1
  fresh  HELD OUT: 20 Gla_01 single-well top views never scored by any version; calibration = 3 images by sorted index
Variants: primary = the frozen Rung 1b config; secondary (pre-declared) = the same without the C1 guard.
Usage: python -m experiments.real_rung1b --dry | --run
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu
from skimage.measure import label as sk_label, regionprops

from experiments.real_rung1 import SOURCES, _cluster_ci, _git, build, frozen_cfg
from nanovate.pipeline_robust import UNVERIFIED, RobustCfg, calibrate_robust, verify_patch_robust
from nanovate.real.lattice import Lattice, render
from nanovate.real.rung1 import TILE, Reference, crop, holes_inside, load_fei, tile_grid, touches_overlay
from nanovate.topo_utils import Window

DATA = Path("data/real")
OUT = Path("results/real/rung1b")
PREREG = Path("docs/REAL_DATA_RUNG1B_PREREG.md")
FROZEN = OUT / "val" / "decisions.json"
LABELS = OUT / "fresh_clean_labels.csv"
RUN1_TILES = Path("results/real/rung1/tiles.csv")
OVERLAY_BOX = (slice(241, 272), slice(369, 400))   # Helios centre cross in 512 x 768 frames (same box as G29's R != B pixels)
FRESH_CAL_INDEX = (0, 10, 19)
N_BOOT, SEED = 1000, 0


def check_prereg(out: Path = OUT) -> str:
    for f in (PREREG, FROZEN, LABELS):
        if not _git("log", "-1", "--format=%H", "--", str(f)):
            sys.exit(f"refusing to run: {f} is not committed")
        if _git("status", "--porcelain", "--", str(f)):
            sys.exit(f"refusing to run: {f} has uncommitted changes")
    if (Path(out) / "RUN_DONE").exists():
        sys.exit(f"refusing to run: {Path(out) / 'RUN_DONE'} exists (Rung 1b runs once)")
    return _git("rev-parse", "HEAD")


def fresh_paths() -> list[str]:
    """Gla_01 single-well top views: z*_d_<n>_*.tif, tilt 0, excluding the 220-280 array image (used in Rung 1)."""
    import tifffile
    out = []
    for p in sorted((DATA / "zmw" / "extracted" / "Gla_0127_14").rglob("z*_d_*.tif")):
        if "220-280" in p.name or "overview" in p.name:
            continue
        with tifffile.TiffFile(p) as t:
            if abs(float(t.fei_metadata["Stage"]["StageT"])) > 1e-3:
                continue
        out.append(str(p))
    return out


def fresh_group() -> dict:
    """One well per image: reference = a single disc of the design diameter at the detected well centre (FITTED position)."""
    ims = {}
    refs = {}
    for p in fresh_paths():
        im = load_fei(p, "FRESH")
        ov = np.zeros(im.img.shape, bool)
        ov[OVERLAY_BOX] = True
        im.overlay = ov
        # the field is centred on its well (the centre cross marks the beam position): take the dark blob nearest the centre
        sm = ndi.gaussian_filter(im.img, 2.0)
        lab = sk_label(sm < threshold_otsu(sm))
        H, W = im.img.shape
        cand = [r for r in regionprops(lab) if r.area >= 20 and r.bbox[0] > 0 and r.bbox[1] > 0 and r.bbox[2] < H and r.bbox[3] < W]
        big = min(cand, key=lambda r: np.hypot(r.centroid[0] - H / 2, r.centroid[1] - W / 2))
        lat = Lattice((0.0, 1e7), (1e7, 0.0), (float(big.centroid[0]), float(big.centroid[1])), float(im.design_d_nm / im.px_nm))
        ims[im.image_id] = im
        refs[im.image_id] = Reference(lat, "design (file name)", (0.0,), 1)
    ids = sorted(ims)
    cal_ids = [ids[i] for i in FRESH_CAL_INDEX]
    ev = [(k, rc) for k in ids if k not in cal_ids for rc in tile_grid(ims[k].img.shape)]
    return dict(name="FRESH", images=ims, refs=refs, cal_ids=cal_ids, eval_tiles=ev)


def _cal_window(shape: tuple[int, int], centre: tuple[float, float]) -> tuple[slice, slice]:
    r = int(np.clip(round(centre[0]) - TILE // 2, 0, shape[0] - TILE))
    c = int(np.clip(round(centre[1]) - TILE // 2, 0, shape[1] - TILE))
    return slice(r, r + TILE), slice(c, c + TILE)


def scribe_patches(g: dict) -> list:
    out = []
    if g["name"] == "FRESH":
        for k in g["cal_ids"]:
            im, lat = g["images"][k], g["refs"][k].lattice
            sl = _cal_window(im.img.shape, lat.o)       # well-centred window (contains the centre cross, disclosed)
            out.append((render(lat, im.img.shape)[sl].astype(float), im.img[sl]))
    else:
        for k, rc in g["cal_tiles"]:
            im = g["images"][k]
            out.append((crop(render(g["refs"][k].lattice, im.img.shape), rc).astype(float), crop(im.img, rc)))
    return out


def run(groups: list[dict], rcfgs: dict, labels: dict, pc, rc) -> tuple[pd.DataFrame, pd.DataFrame]:
    win = Window(TILE, 16)
    rows, cals = [], []
    for g in groups:
        P = scribe_patches(g)
        cad_of = {k: render(g["refs"][k].lattice, im.img.shape) for k, im in g["images"].items()}
        for vname, rcfg in rcfgs.items():
            rcal = calibrate_robust(P, rcfg)
            cals.append(dict(variant=vname, group=g["name"], reliable=rcal.reliable, contrast=rcal.contrast,
                             loo=json.dumps(rcal.loo_contrasts), clamp_range=json.dumps(rcal.clamp_range), calib_ms=rcal.cal.timing_ms,
                             cd_bias_px=rcal.cd_bias))
            for k, rc_ in g["eval_tiles"]:
                im = g["images"][k]
                cad, img = crop(cad_of[k], rc_), crop(im.img, rc_)
                t0 = time.perf_counter()
                res = verify_patch_robust(img, cad, rcal, win, pc, rc, rcfg)
                ms = (time.perf_counter() - t0) * 1e3
                z = res.reason
                rows.append(dict(variant=vname, group=g["name"], image_id=k, tile=f"{rc_[0]},{rc_[1]}", px_nm=im.px_nm,
                                 clean=labels.get(k) == "clean", overlay=touches_overlay(im.overlay, rc_), holes_inside=holes_inside(cad),
                                 verdict=res.verdict, verdict_unguarded=z["verdict_unguarded"], verified=res.verdict != UNVERIFIED,
                                 killer=str(res.verdict).startswith("KILLER"), killer_unguarded=str(z["verdict_unguarded"]).startswith("KILLER"),
                                 d_beta0=z["delta_betti"][0], d_beta1=z["delta_betti"][1], n_bridge_events=z["n_bridge_events"],
                                 n_open_events=z["n_open_events"], bottleneck=z["bottleneck_max"], max_sdf_energy=z["max_sdf_energy"],
                                 max_sdf_energy_design=z.get("max_sdf_energy_design"), peak_z=z["peak_z"],
                                 align_rejected=z["align_rejected"], m3_shift_px=float(np.hypot(*res.shift)), ms_total=ms))
    return pd.DataFrame(rows), pd.DataFrame(cals)


def summarise(t: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (v, pop), g in t.groupby(["variant", "population"]):
        base = g[g.clean & ~g.overlay]
        for col, name in (("killer", "KILLER on clean tiles (UNVERIFIED counts as not KILLER)"), ("verified", "coverage (verified fraction)"),
                          ("killer_unguarded", "KILLER on clean tiles if the guard were off")):
            m, lo, hi, n, ni = _cluster_ci(base, col)
            out.append(dict(variant=v, population=pop, metric=name, mean=m, ci_lo=lo, ci_hi=hi, n_tiles=n, n_images=ni))
        ver = base[base.verified]
        m, lo, hi, n, ni = _cluster_ci(ver, "killer")
        out.append(dict(variant=v, population=pop, metric="KILLER among verified clean tiles", mean=m, ci_lo=lo, ci_hi=hi, n_tiles=n, n_images=ni))
        for gname, gg in base.groupby("group"):
            for col in ("killer", "verified", "killer_unguarded"):
                m, lo, hi, n, ni = _cluster_ci(gg, col)
                out.append(dict(variant=v, population=f"{pop}/{gname}", metric=col, mean=m, ci_lo=lo, ci_hi=hi, n_tiles=n, n_images=ni))
        ov = g[g.clean & g.overlay]
        if len(ov):
            m, lo, hi, n, ni = _cluster_ci(ov, "killer")
            out.append(dict(variant=v, population=pop, metric="KILLER on clean overlay tiles (not in target)", mean=m, ci_lo=lo, ci_hi=hi, n_tiles=n, n_images=ni))
    return pd.DataFrame(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry", action="store_true")
    mode.add_argument("--run", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    orig = [g for g in build("C")[0] if not g["name"].startswith("INGAN")]
    fresh = fresh_group()
    if a.dry:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        sheet = DATA / "rung1b_sheets"
        sheet.mkdir(parents=True, exist_ok=True)
        man = []
        for k, im in fresh["images"].items():
            cad = render(fresh["refs"][k].lattice, im.img.shape)
            fig, ax = plt.subplots(figsize=(7.68, 5.12), dpi=100)
            ax.imshow(im.img, cmap="gray"); ax.contour(cad, [0.5], colors="c", linewidths=0.6)
            for r in (256,):
                ax.axhline(r, color="y", lw=0.5)
            for c in (256, 512):
                ax.axvline(c, color="y", lw=0.5)
            ax.set_title(f"FRESH {k} ({'cal' if k in fresh['cal_ids'] else 'eval'})", fontsize=7); ax.axis("off")
            fig.savefig(sheet / f"{k}.png", bbox_inches="tight"); plt.close(fig)
            man.append(dict(image_id=k, group="FRESH", role="calibration" if k in fresh["cal_ids"] else "evaluation", path=im.path,
                            sha256=im.meta["sha256"], px_nm=im.px_nm, px_source=im.px_source, design_d_nm=im.design_d_nm,
                            **{x: im.meta.get(x) for x in ("system", "detector", "hv_V", "current_A", "dwell_s", "hfw_m", "tilt_rad")},
                            doi=SOURCES["G01"]["doi"], licence=SOURCES["G01"]["licence"], credit=SOURCES["G01"]["credit"], data_kind="REAL SEM image"))
        pd.DataFrame(man).to_csv(OUT / "fresh_manifest.csv", index=False)
        print("fresh images", len(fresh["images"]), "cal", fresh["cal_ids"], "eval tiles", len(fresh["eval_tiles"]))
        return
    commit = check_prereg()
    pc, rc = frozen_cfg()
    frozen = json.loads(FROZEN.read_text())
    prim = RobustCfg(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in frozen["robust_cfg"].items()})
    rcfgs = {"primary": prim, "secondary_no_guard": RobustCfg(**{**{k: (tuple(v) if isinstance(v, list) else v) for k, v in frozen["robust_cfg"].items()}, "guard": False})}
    lab1 = pd.read_csv("results/real/rung1/clean_labels.csv")
    labf = pd.read_csv(LABELS)
    labels = {**dict(zip(lab1.image_id, lab1.label)), **dict(zip(labf.image_id, labf.label))}
    t0 = time.time()
    to, co = run(orig, rcfgs, labels, pc, rc)
    tf, cf = run([fresh], rcfgs, labels, pc, rc)
    to["population"], tf["population"] = "orig", "fresh"
    t = pd.concat([to, tf]); c = pd.concat([co, cf])
    # paired with Rung 1 (same orig tiles)
    r1 = pd.read_csv(RUN1_TILES)
    r1 = r1[(r1.arm == "C")][["image_id", "tile", "verdict"]].rename(columns={"verdict": "verdict_rung1"})
    t = t.merge(r1, on=["image_id", "tile"], how="left")
    t.to_csv(OUT / "tiles.csv", index=False)
    c.to_csv(OUT / "calibration.csv", index=False)
    s = summarise(t)
    s.to_csv(OUT / "summary.csv", index=False)
    (OUT / "config.json").write_text(json.dumps(dict(data="REAL SEM images (Track C Rung 1b)", reference="FITTED, not CAD", prereg=str(PREREG),
                                                     prereg_commit=commit, frozen=frozen, variants={k: v.to_dict() for k, v in rcfgs.items()},
                                                     fresh_cal_index=list(FRESH_CAL_INDEX), overlay_box=[[241, 272], [369, 400]],
                                                     n_boot=N_BOOT, seed=SEED, wall_s=time.time() - t0), indent=2, default=str))
    (OUT / "RUN_DONE").write_text(json.dumps(dict(commit=commit, finished=time.strftime("%Y-%m-%d %H:%M:%S")), indent=2))
    print(s.to_string())


if __name__ == "__main__":
    main()
