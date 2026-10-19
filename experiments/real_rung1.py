"""Track C Rung 1: the FROZEN v3 verifier on real SEM images of defect-free hole arrays (REAL DATA, run once).

Pre-registration: docs/REAL_DATA_PREREG.md (must be committed, with results/real/rung1/clean_labels.csv, before --run).
Frozen config: results/v3/calib/decisions.json (pipeline_cfg + event_rule_cfg), Window(256, 16), Module-1 defaults.

  --dry   build manifest, fitted references and labelling sheets (data/real/rung1_sheets/, git-ignored). No verifier.
  --run   the one pre-registered run: arm C (native px, primary) and arm B (diameter-matched, secondary).
          Refuses if the pre-registration is not committed or a run already happened (results/real/rung1/RUN_DONE).

Reference = FITTED lattice (not CAD). Intensities are 8-bit grey levels; the dose is unknown.
Usage: python -m experiments.real_rung1 --dry | --run
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from nanovate.modules.align_sdf import align_to_template
from nanovate.modules.calibration import calibrate
from nanovate.modules.events import EventRuleCfg
from nanovate.pipeline_v3 import verify_patch_v3
from nanovate.real.lattice import detect_holes, render
from nanovate.real.rung1 import (G01_CAL_IMAGE, G01_CAL_ROWS, G01_CAL_TILES, G29_CAL_INDEX, TILE, VIA_PX, RealImage, best_cal_tile,
                                 build_reference, crop, fit_basis_nm, fitted_diameter_nm, holes_inside, load_fei,
                                 load_figure_jpg, rescale_image, select_zmw, tile_grid)
from nanovate.topo_utils import Window

DATA = Path("data/real")
OUT = Path("results/real/rung1")
PREREG = Path("docs/REAL_DATA_PREREG.md")
LABELS = OUT / "clean_labels.csv"
DECISIONS = Path("results/v3/calib/decisions.json")
SOURCES = {
    "G29": dict(doi="10.5281/zenodo.8060099", licence="CC-BY-4.0", file="Gla_29_Pd_1.zip",
                credit="Yang, Klughammer, Barth, Tanenbaum, Dekker; ACS Nano 2023 (ZMW nanowells, array v2)"),
    "G01": dict(doi="10.5281/zenodo.8060099", licence="CC-BY-4.0", file="Gla_0127_14.zip",
                credit="Yang, Klughammer, Barth, Tanenbaum, Dekker; ACS Nano 2023 (ZMW nanowells, array v1)"),
    "INGAN": dict(doi="10.5281/zenodo.3747155", licence="CC-BY-4.0", file="Figure 2(b)/(d) top-view JPGs",
                  credit="Athanasiou et al.; InGaN nanohole arrays (figure panels, spot-check only)"),
}
N_BOOT, SEED = 1000, 0


def _git(*args: str) -> str:
    return subprocess.run(["git", *args], capture_output=True, text=True).stdout.strip()


def check_prereg(out: Path = OUT) -> str:
    """The run guard: committed, unmodified pre-registration and labels; no earlier run. Returns the commit hash."""
    for f in (PREREG, LABELS):
        if not _git("log", "-1", "--format=%H", "--", str(f)):
            sys.exit(f"refusing to run: {f} is not committed")
        if _git("status", "--porcelain", "--", str(f)):
            sys.exit(f"refusing to run: {f} has uncommitted changes")
    if (Path(out) / "RUN_DONE").exists():
        sys.exit(f"refusing to run: {Path(out) / 'RUN_DONE'} exists (Rung 1 runs once)")
    return _git("rev-parse", "HEAD")


def frozen_cfg():
    from experiments.v3_harness import _pc
    dec = json.loads(DECISIONS.read_text())
    return _pc(dec["pipeline_cfg"]), EventRuleCfg(**dec["event_rule_cfg"])


# --------------------------------------------------------------------------- build (no verifier)

def _centres_nm(im: RealImage, rows_lt: int | None = None) -> np.ndarray:
    c = np.array([r.centroid for r in detect_holes(im.img)])
    if rows_lt is not None:
        c = c[c[:, 0] < rows_lt]
    return c * im.px_nm


def _group(name, images, cal_ids, basis, d_fit_nm, cal_tiles, eval_tiles, scale=1.0, note=""):
    return dict(name=name, images=images, cal_ids=cal_ids, basis=basis, d_fit_nm=d_fit_nm, cal_tiles=cal_tiles,
                eval_tiles=eval_tiles, scale=scale, note=note)


def build(arm: str) -> tuple[list[dict], list[dict]]:
    """Groups (images, fitted references, calibration and evaluation tiles) for arm 'C' or 'B'; plus exclusions."""
    g29p, g01p, excl = select_zmw(DATA / "zmw" / "extracted")
    excluded = [dict(path=p, reason=r, arm="both") for p, r in excl]
    g29 = [load_fei(p, "G29") for p in g29p]
    g01 = {Path(p).name: load_fei(p, "G01") for p in g01p}
    groups = []

    # G29: basis from 3 calibration images; diameters from the design value in each file name
    cal29 = [g29[i] for i in G29_CAL_INDEX]
    b29 = fit_basis_nm([_centres_nm(im) for im in cal29])
    # G01: basis and diameter from the calibration strip (row 0) of one overview image
    cal01 = g01[G01_CAL_IMAGE]
    b01 = fit_basis_nm([_centres_nm(cal01, rows_lt=G01_CAL_ROWS * TILE)])
    d01 = fitted_diameter_nm([cal01], regions=[(r, c) for r in range(G01_CAL_ROWS) for c in range(cal01.img.shape[1] // TILE)])

    if arm == "C":
        refs = {im.image_id: build_reference(im, *b29, None) for im in g29}
        cal_tiles = [(im.image_id, best_cal_tile(render(refs[im.image_id].lattice, im.img.shape), im.overlay)) for im in cal29]
        ev = [(im.image_id, rc) for im in g29 if im.image_id not in {c.image_id for c in cal29} for rc in tile_grid(im.img.shape)]
        groups.append(_group("G29", {im.image_id: im for im in g29}, [im.image_id for im in cal29], b29, None, cal_tiles, ev)
                      | dict(refs=refs))
        refs1 = {k: build_reference(im, *b01, d01) for k, im in g01.items()}
        ev1 = [(k, rc) for k, im in g01.items() for rc in tile_grid(im.img.shape) if not (k == G01_CAL_IMAGE and rc[0] < G01_CAL_ROWS)]
        groups.append(_group("G01", g01, [G01_CAL_IMAGE], b01, d01, [(G01_CAL_IMAGE, rc) for rc in G01_CAL_TILES], ev1) | dict(refs=refs1))
        # INGAN spot-check: per image, Module 1 on row 0, evaluate row 1. One tile row holds a single row of holes, so the
        # lattice and diameter are fitted on the WHOLE image (spot-check only, descriptive; disclosed as a weaker reference)
        for p in sorted((DATA / "ingan" / "raw").glob("*.jpg")):
            im = load_figure_jpg(p)
            b = fit_basis_nm([_centres_nm(im)])
            d = fitted_diameter_nm([im])
            ref = build_reference(im, *b, d)
            nm = "INGAN:" + im.image_id
            groups.append(_group(nm, {im.image_id: im}, [im.image_id], b, d, [(im.image_id, (0, c)) for c in range(im.img.shape[1] // TILE)],
                                 [(im.image_id, (1, c)) for c in range(im.img.shape[1] // TILE)], note="spot-check, descriptive only")
                          | dict(refs={im.image_id: ref}))
    elif arm == "B":
        # G29 is excluded if the diameter-matched image is shorter than one tile
        for im in g29:
            s = VIA_PX / (im.design_d_nm / im.px_nm)
            if im.img.shape[0] * s < TILE:
                excluded.append(dict(path=im.path, reason=f"arm B: rescaled x{s:.3f} -> {im.img.shape[0] * s:.0f} px tall < {TILE}", arm="B"))
        imsB = {k: rescale_image(im, VIA_PX / (d01 / im.px_nm)) for k, im in g01.items()}
        refsB = {k: build_reference(im, *b01, d01) for k, im in imsB.items()}
        sB = imsB[G01_CAL_IMAGE].meta["rescale"]
        evB = [(k, rc) for k, im in imsB.items() for rc in tile_grid(im.img.shape)
               if not (k == G01_CAL_IMAGE and rc[0] * TILE < G01_CAL_ROWS * TILE * sB)]   # no overlap with the native calibration rows
        groups.append(_group("G01", imsB, [G01_CAL_IMAGE], b01, d01, [(G01_CAL_IMAGE, rc) for rc in G01_CAL_TILES], evB) | dict(refs=refsB))
    else:
        raise ValueError(arm)
    return groups, excluded


def manifest(groups_c: list[dict], excluded: list[dict]) -> pd.DataFrame:
    rows = []
    for g in groups_c:
        for k, im in g["images"].items():
            src = SOURCES["INGAN" if g["name"].startswith("INGAN") else g["name"]]
            ref = g["refs"][k]
            rows.append(dict(image_id=k, group=g["name"], role="calibration" if k in g["cal_ids"] else "evaluation",
                             path=str(Path(im.path)), sha256=im.meta["sha256"], px_nm=im.px_nm, px_source=im.px_source,
                             height_px=im.img.shape[0], width_px=im.img.shape[1], design_d_nm=im.design_d_nm,
                             ref_d_nm=ref.lattice.d * im.px_nm, ref_d_source=ref.d_source, holes_detected=ref.n_holes_detected,
                             **{k2: im.meta.get(k2) for k2 in ("system", "detector", "mode", "hv_V", "current_A", "dwell_s", "hfw_m",
                                                               "tilt_rad", "integrate", "average", "dtype", "clip_lo", "clip_hi", "note")},
                             doi=src["doi"], licence=src["licence"], source_file=src["file"], credit=src["credit"],
                             data_kind="REAL SEM image"))
    return pd.DataFrame(rows)


def sheets(groups: list[dict], out: Path) -> None:
    """Labelling sheets: each image with its FITTED reference contour and tile grid (git-ignored folder)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out.mkdir(parents=True, exist_ok=True)
    for g in groups:
        for k, im in g["images"].items():
            cad = render(g["refs"][k].lattice, im.img.shape)
            fig, ax = plt.subplots(figsize=(im.img.shape[1] / 100, im.img.shape[0] / 100), dpi=100)
            ax.imshow(im.img, cmap="gray"); ax.contour(cad, [0.5], colors="c", linewidths=0.6)
            for r in range(im.img.shape[0] // TILE + 1):
                ax.axhline(r * TILE, color="y", lw=0.5)
            for c in range(im.img.shape[1] // TILE + 1):
                ax.axvline(c * TILE, color="y", lw=0.5)
            ax.set_title(f"{g['name']} {k} ({'cal' if k in g['cal_ids'] else 'eval'})", fontsize=7); ax.axis("off")
            fig.savefig(out / f"{g['name'].replace(':', '_')}__{k}.png", bbox_inches="tight"); plt.close(fig)


# --------------------------------------------------------------------------- run

def _cluster_ci(df: pd.DataFrame, col: str) -> tuple[float, float, float, int, int]:
    """Mean of boolean ``col`` over tiles, 95% CI by bootstrap over images (tiles of one image move together)."""
    if df.empty:
        return float("nan"), float("nan"), float("nan"), 0, 0
    by = [g[col].values.astype(float) for _, g in df.groupby("image_id")]
    rng = np.random.default_rng(SEED)
    stats = []
    for _ in range(N_BOOT):
        pick = rng.integers(0, len(by), len(by))
        x = np.concatenate([by[i] for i in pick])
        stats.append(x.mean())
    return float(df[col].mean()), float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5)), len(df), len(by)


def run_arm(arm: str, groups: list[dict], labels: pd.DataFrame, pc, rc) -> tuple[pd.DataFrame, pd.DataFrame]:
    win = Window(TILE, 16)
    lab = dict(zip(labels.image_id, labels.label))
    tiles, cals = [], []
    for g in groups:
        ims, refs = g["images"], g["refs"]
        cad_of = {k: render(refs[k].lattice, im.img.shape) for k, im in ims.items()}
        scribe = [(crop(cad_of[k], rc_).astype(float), crop(ims[k].img, rc_)) for k, rc_ in g["cal_tiles"]]
        cal = calibrate(scribe)
        for (k, rc_), (c_, y) in zip(g["cal_tiles"], scribe):
            T = cal.predict(c_ > 0.5)
            cals.append(dict(arm=arm, group=g["name"], image_id=k, tile=f"{rc_[0]},{rc_[1]}", calib_ms=cal.timing_ms,
                             rel_err=np.linalg.norm(T - y) / np.linalg.norm(y),
                             rel_err_contrast=np.linalg.norm(T - y) / np.linalg.norm(y - y.mean()),
                             mu_film=cal.mu_cond, mu_hole=cal.mu_diel, alpha=cal.alpha, sigma2=cal.sigma2))
        for k, rc_ in g["eval_tiles"]:
            im = ims[k]
            cad = crop(cad_of[k], rc_)
            img = crop(im.img, rc_)
            t0 = time.perf_counter()
            res = verify_patch_v3(img, cad, cal, win, pc, rc)
            ms = (time.perf_counter() - t0) * 1e3
            T = cal.predict(cad)
            Ia, _ = align_to_template(img, T)
            s = win.sl
            e = (T - Ia)[s]
            noise = np.clip(cal.alpha * T[s] + cal.sigma2, 0, None).sum()
            rz = res.reason
            db = rz.get("delta_betti", (None, None))
            tiles.append(dict(arm=arm, group=g["name"], image_id=k, tile=f"{rc_[0]},{rc_[1]}", px_nm=im.px_nm,
                              scale=im.meta.get("rescale", 1.0), label=lab.get(k, "unlabelled"),
                              clean=lab.get(k) == "clean", overlay=touches_overlay(im.overlay, rc_), holes_inside=holes_inside(cad), verdict=res.verdict,
                              killer=str(res.verdict).startswith("KILLER"), near_miss=res.verdict == "NEAR_MISS",
                              d_beta0=db[0], d_beta1=db[1], n_bridge_events=rz.get("n_bridge_events"), n_open_events=rz.get("n_open_events"),
                              bottleneck=rz.get("bottleneck_max"), max_sdf_energy=rz.get("max_sdf_energy"), peak_z=rz.get("peak_z"),
                              events=json.dumps(rz.get("events"), default=str)[:2000],
                              m3_shift_r=res.shift[0], m3_shift_c=res.shift[1], m3_shift_px=float(np.hypot(*res.shift)),
                              calib_rel_err=float(np.linalg.norm(e) / np.linalg.norm(Ia[s])),
                              calib_rel_err_contrast=float(np.linalg.norm(e) / np.linalg.norm(Ia[s] - Ia[s].mean())),
                              calib_rel_err_contrast_noise_corr=float(np.sqrt(max(np.sum(e ** 2) - noise, 0.0)) / np.linalg.norm(Ia[s] - Ia[s].mean())),
                              ms_total=ms, **{f"ms_{a}": b for a, b in res.timings_ms.items()}))
    return pd.DataFrame(tiles), pd.DataFrame(cals)


def summarise(t: pd.DataFrame, holes: pd.DataFrame) -> pd.DataFrame:
    rows = []

    def add(metric, sub, col, scope):
        m, lo, hi, n, ni = _cluster_ci(sub, col)
        rows.append(dict(metric=metric, scope=scope, mean=m, ci_lo=lo, ci_hi=hi, n_tiles=n, n_images=ni))

    for arm, ta in t.groupby("arm"):
        zmw = ta[~ta.group.str.startswith("INGAN")]
        clean = zmw[zmw.clean & ~zmw.overlay]
        ov = zmw[zmw.clean & zmw.overlay]
        if len(ov):
            add("KILLER on clean tiles touching the display overlay (excluded from primary)", ov, "killer", f"arm {arm}, all ZMW groups")
        add("false alarm: KILLER on clean tiles", clean, "killer", f"arm {arm}, all ZMW groups")
        add("NEAR_MISS on clean tiles", clean, "near_miss", f"arm {arm}, all ZMW groups")
        for gname, g in clean.groupby("group"):
            add("false alarm: KILLER on clean tiles", g, "killer", f"arm {arm}, {gname}")
            add("NEAR_MISS on clean tiles", g, "near_miss", f"arm {arm}, {gname}")
        for gname, g in ta[ta.group.str.startswith("INGAN")].groupby("group"):
            add("KILLER (spot-check, not clean)", g, "killer", f"arm {arm}, {gname}")
        for col, nm in (("calib_rel_err", "calibration rel. error ||T-I||/||I||"),
                        ("calib_rel_err_contrast", "calibration rel. error ||T-I||/||I-mean||"),
                        ("calib_rel_err_contrast_noise_corr", "calibration rel. error, noise-corrected, contrast-relative"),
                        ("m3_shift_px", "alignment: Module-3 residual shift (px)"), ("ms_total", "time per tile (ms)")):
            for gname, g in zmw.groupby("group"):
                x = g[col].astype(float).values
                rng = np.random.default_rng(SEED)
                bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(N_BOOT)]
                rows.append(dict(metric=nm, scope=f"arm {arm}, {gname}", mean=float(x.mean()), ci_lo=float(np.percentile(bs, 2.5)),
                                 ci_hi=float(np.percentile(bs, 97.5)), n_tiles=len(x), n_images=g.image_id.nunique()))
    for (arm, gname), h in holes.groupby(["arm", "group"]):
        x = h.resid_nm.values
        rng = np.random.default_rng(SEED)
        bs = [x[rng.integers(0, len(x), len(x))].mean() for _ in range(N_BOOT)]
        rows.append(dict(metric="alignment: hole centre to FITTED reference centre (nm)", scope=f"arm {arm}, {gname}",
                         mean=float(x.mean()), ci_lo=float(np.percentile(bs, 2.5)), ci_hi=float(np.percentile(bs, 97.5)),
                         n_tiles=len(x), n_images=h.image_id.nunique()))
    return pd.DataFrame(rows)


def examples(t: pd.DataFrame, groups_by_arm: dict, out: Path, k: int = 5) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    bad = t[(t.arm == "C") & t.clean & ~t.overlay & t.killer].sort_values(["group", "image_id", "tile"]).head(k)
    paths = []
    for _, r in bad.iterrows():
        g = next(g for g in groups_by_arm["C"] if g["name"] == r.group)
        im = g["images"][r.image_id]
        rc_ = tuple(int(v) for v in r.tile.split(","))
        img, cad = crop(im.img, rc_), crop(render(g["refs"][r.image_id].lattice, im.img.shape), rc_)
        fig, ax = plt.subplots(1, 2, figsize=(7, 3.8))
        ax[0].imshow(img, cmap="gray"); ax[0].contour(cad, [0.5], colors="c", linewidths=0.6); ax[0].set_title("real image + FITTED reference", fontsize=7)
        ax[1].imshow(cad, cmap="gray"); ax[1].set_title("FITTED reference (film = white)", fontsize=7)
        for a in ax:
            a.axis("off")
        fig.suptitle(f"REAL DATA ({SOURCES[r.group]['doi']}, {SOURCES[r.group]['licence']}) {r.image_id} tile {r.tile}\n"
                     f"{r.verdict}: dbeta=({r.d_beta0},{r.d_beta1}) bridge ev {r.n_bridge_events} open ev {r.n_open_events} "
                     f"bottleneck {r.bottleneck:.3f} SDF energy {r.max_sdf_energy:.2f}", fontsize=7)
        p = out / f"example_{len(paths)}_{r.group}_{Path(r.image_id).stem}_{r.tile.replace(',', '-')}.png"
        fig.savefig(p, dpi=110, bbox_inches="tight"); plt.close(fig)
        paths.append(str(p))
    return paths


def main() -> None:
    ap = argparse.ArgumentParser()
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry", action="store_true")
    mode.add_argument("--run", action="store_true")
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    gc, exc = build("C")
    gb, excb = build("B")
    exc = exc + [e for e in excb if e["arm"] == "B"]
    if a.dry:
        manifest(gc, exc).to_csv(OUT / "manifest.csv", index=False)
        pd.DataFrame(exc).to_csv(OUT / "excluded.csv", index=False)
        sheets(gc, DATA / "rung1_sheets" / "C")
        sheets(gb, DATA / "rung1_sheets" / "B")
        for arm, gs in (("C", gc), ("B", gb)):
            for g in gs:
                print(arm, g["name"], "images", len(g["images"]), "cal tiles", len(g["cal_tiles"]), "eval tiles", len(g["eval_tiles"]),
                      "basis nm", np.round(g["basis"], 1).tolist(), "d_fit nm", g["d_fit_nm"])
        return
    commit = check_prereg()
    pc, rc = frozen_cfg()
    labels = pd.read_csv(LABELS)
    t0 = time.time()
    tc, cc = run_arm("C", gc, labels, pc, rc)
    tb, cb = run_arm("B", gb, labels, pc, rc)
    t, cal = pd.concat([tc, tb]), pd.concat([cc, cb])
    holes = []
    for arm, gs in (("C", gc), ("B", gb)):
        for g in gs:
            for k, ref in g["refs"].items():
                if k in g["cal_ids"] and not g["name"].startswith("INGAN"):
                    continue
                im = g["images"][k]
                holes += [dict(arm=arm, group=g["name"], image_id=k, resid_nm=r * im.px_nm) for r in ref.hole_resid_px]
    holes = pd.DataFrame(holes)
    t.to_csv(OUT / "tiles.csv", index=False)
    cal.to_csv(OUT / "calibration.csv", index=False)
    holes.to_csv(OUT / "hole_residuals.csv", index=False)
    s = summarise(t, holes)
    s.to_csv(OUT / "summary.csv", index=False)
    ex = examples(t, {"C": gc}, OUT)
    manifest(gc, exc).to_csv(OUT / "manifest.csv", index=False)
    pd.DataFrame(exc).to_csv(OUT / "excluded.csv", index=False)
    cfg = dict(data="REAL SEM images (Track C Rung 1)", reference="FITTED lattice, not CAD", prereg=str(PREREG), prereg_commit=commit,
               frozen_config=str(DECISIONS), pipeline_cfg=json.loads(DECISIONS.read_text())["pipeline_cfg"],
               event_rule_cfg=json.loads(DECISIONS.read_text())["event_rule_cfg"], window=dict(size=TILE, margin=16),
               arms=dict(C="native pixels (primary)", B=f"diameter-matched to {VIA_PX:g} px (secondary)"), n_boot=N_BOOT, seed=SEED,
               g29_cal_index=list(G29_CAL_INDEX), g01_cal_image=G01_CAL_IMAGE, g01_cal_tiles=[list(x) for x in G01_CAL_TILES],
               wall_s=time.time() - t0, examples=ex)
    (OUT / "config.json").write_text(json.dumps(cfg, indent=2, default=str))
    (OUT / "RUN_DONE").write_text(json.dumps(dict(commit=commit, finished=time.strftime("%Y-%m-%d %H:%M:%S")), indent=2))
    print(s.to_string())


if __name__ == "__main__":
    main()
