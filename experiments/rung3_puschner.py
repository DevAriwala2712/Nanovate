"""Track C Rung 3: frozen verifier in REFERENCE-CELL mode on the Puschner et al. 90 nm chip (REAL DATA).

Data: doi:10.17617/3.396Q7I (CC-BY-4.0): backside SEM tiles + abstracted, trojanized GDSII (cell boxes + anonymised
types; 10 cells swapped, positions withheld). Pre-registration: docs/REAL_DATA_RUNG3_PREREG.md.
Stages:
  --register  coarse registration (results of the scale/orientation search) + affine correction fitted on sample tiles
              + per-tile residual offsets for all tiles.  No verifier.
  --dry       window geometry, home tiles, group references chosen; coverage counts.  No verifier.
  --run       the one pre-registered run (guarded).
Units: mosaic px (12.23 nm/px from registration), working frame x0.5.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import pickle
import sys
import time
import zipfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.real_rung1 import _cluster_ci, _git, frozen_cfg
from nanovate.pipeline_robust import RobustCfg
from nanovate.real.puschner import EXTRACTED, gds_to_mosaic, parse_positions, read_tile, tile_offset
from nanovate.topo_utils import Window

ZIP = "data/real/puschner/raw/90nm_tiles.zip"
WORK = Path("data/real/puschner/work")
OUT = Path("results/real/rung3")
PREREG = Path("docs/REAL_DATA_RUNG3_PREREG.md")
FROZEN_1D = Path("results/real/rung1d/val/decisions.json")
S = 0.5                 # working scale (x0.5)
WIN = 256               # window, px at working scale
STEP = 224              # window step along long cells
K_REF = 15              # reference instances per (type, orientation) group
MIN_GROUP = 4
SEED = 0
TILE = 4096


def frozen():
    pc, rc = frozen_cfg()
    dec = json.loads(FROZEN_1D.read_text())
    th = dec["thresholds"][dec["choice"]]
    pc = dataclasses.replace(pc, d_star=th["d_star"], e_star=th["e_star"], z_star=th["z_star"])
    return pc, rc, RobustCfg(**{k: (tuple(v) if isinstance(v, list) else v) for k, v in dec["robust_cfg"].items()})


# --------------------------------------------------------------------------- registration
def _offset_task(args):
    f, ty, tx = args
    inst = pickle.load(open(WORK / "instances_reg.pkl", "rb"))
    with zipfile.ZipFile(ZIP) as z:
        return (f, *tile_offset(read_tile(z, f), inst, ty, tx))


def register(workers: int) -> None:
    pos = parse_positions(EXTRACTED / "img-global-positions-0.txt")
    reg = json.loads((WORK / "registration_coarse.json").read_text())
    T = gds_to_mosaic(reg)
    inst = pickle.load(open(WORK / "instances.pkl", "rb"))

    def apply(inst, corr=None):
        r0, c0 = T(inst.x0.values, inst.y0.values)
        r1, c1 = T(inst.x1.values, inst.y1.values)
        R0, R1, C0, C1 = np.minimum(r0, r1), np.maximum(r0, r1), np.minimum(c0, c1), np.maximum(c0, c1)
        if corr is not None:
            rc, cc = (R0 + R1) / 2, (C0 + C1) / 2
            dr = corr[0] + corr[1] * rc + corr[2] * cc
            dc = corr[3] + corr[4] * rc + corr[5] * cc
            R0, R1, C0, C1 = R0 + dr, R1 + dr, C0 + dc, C1 + dc
        out = inst.copy()
        out["R0"], out["R1"], out["C0"], out["C1"] = R0, R1, C0, C1
        return out

    pickle.dump(apply(inst), open(WORK / "instances_reg.pkl", "wb"))
    samp = pos[(pos.col % 4 == 2) & (pos.row % 4 == 2)]
    with ProcessPoolExecutor(workers) as ex:
        o = pd.DataFrame(list(ex.map(_offset_task, [(r.file, r.y, r.x) for r in samp.itertuples()])), columns=["file", "dr", "dc", "score"])
    o = o.merge(samp, on="file")
    good = o[o.score >= o.score.quantile(0.25)]                     # fit on the better-matching 75%
    Xc = np.c_[np.ones(len(good)), good.y + TILE / 2, good.x + TILE / 2]
    br, *_ = np.linalg.lstsq(Xc, good.dr, rcond=None)
    bc, *_ = np.linalg.lstsq(Xc, good.dc, rcond=None)
    corr = [*br, *bc]
    pickle.dump(apply(inst, corr), open(WORK / "instances_reg.pkl", "wb"))
    with ProcessPoolExecutor(workers) as ex:
        allo = pd.DataFrame(list(ex.map(_offset_task, [(r.file, r.y, r.x) for r in pos.itertuples()], chunksize=4)),
                            columns=["file", "dr", "dc", "score"])
    allo = allo.merge(pos, on="file")
    allo.to_csv(WORK / "tile_offsets.csv", index=False)
    json.dump(dict(coarse=reg, affine_correction=corr, sample_tiles=len(o), fit_tiles=len(good),
                   residual_after_affine=dict(dr_median=float(allo.dr.median()), dc_median=float(allo.dc.median()),
                                              abs_dr_p90=float(allo.dr.abs().quantile(.9)), abs_dc_p90=float(allo.dc.abs().quantile(.9)),
                                              score_median=float(allo.score.median()))),
              open(WORK / "registration.json", "w"), indent=1)
    print(json.load(open(WORK / "registration.json"))["residual_after_affine"])


# --------------------------------------------------------------------------- windows (no verifier)
ROW_TRIM_UM = 0.30      # cell boxes are 3.12 um tall on a 2.52 um row pitch: trim the 0.30 um rail/well overlap each side


def robust_offsets(offs: pd.DataFrame) -> pd.DataFrame:
    """Per-tile (dr, dc): the tile's own match if score >= 0.2 and inside the search range, else the median of its
    8 neighbours' good matches, else the median of all good core matches."""
    o = offs.copy()
    good = (o.score >= 0.2) & (o.dr.abs() < 160) & (o.dc.abs() < 160) & (o.n_cells > 200)
    key = {(r.col, r.row): (r.dr, r.dc) for r in o[good].itertuples()}
    gmed = (float(o[good].dr.median()), float(o[good].dc.median()))
    dr, dc, src = [], [], []
    for r in o.itertuples():
        if good.loc[r.Index]:
            dr.append(r.dr); dc.append(r.dc); src.append("own")
            continue
        nb = [key[(r.col + a, r.row + b)] for a in (-1, 0, 1) for b in (-1, 0, 1) if (r.col + a, r.row + b) in key]
        if nb:
            dr.append(float(np.median([v[0] for v in nb]))); dc.append(float(np.median([v[1] for v in nb]))); src.append("neighbours")
        else:
            dr.append(gmed[0]); dc.append(gmed[1]); src.append("global")
    return o.assign(dr_use=dr, dc_use=dc, offset_source=src)


def windows_for(inst: pd.DataFrame, tiles: pd.DataFrame) -> pd.DataFrame:
    """One row per (instance, window j) with its home tile (the tile that contains the whole window, most centrally,
    after that tile's offset) and the window / cell-core box in that tile's x0.5 frame. No home tile -> tile = ''."""
    reg = json.loads((WORK / "registration.json").read_text())["coarse"]
    p_um = reg["p_nm"] / 1000.0
    trim = ROW_TRIM_UM / p_um                                       # mosaic px
    W = WIN / S
    tl = tiles.set_index(["col", "row"])
    pitch_c = float(np.median(np.diff(sorted(tiles.x.unique())))) if tiles.x.nunique() > 1 else TILE
    xs = tiles.groupby("col").x.median(); ys = tiles.groupby("row").y.median()
    out = []
    for idx, q in enumerate(inst.itertuples()):
        R0, R1 = q.R0, q.R1                                         # along the cell width (GDS x)
        C0, C1 = q.C0 + trim, q.C1 - trim                           # across the row, trimmed to the row pitch
        n = 1 if (R1 - R0) * S <= WIN - 32 else int(np.ceil(((R1 - R0) * S - (WIN - 32)) / STEP)) + 1
        for j in range(n):
            rc = (R0 + R1) / 2 if n == 1 else R0 + (16 + j * STEP + WIN / 2 - 16) / S
            cc = (C0 + C1) / 2
            ci = int(np.argmin(np.abs(xs.values + TILE / 2 - cc))); ri = int(np.argmin(np.abs(ys.values + TILE / 2 - rc)))
            best = None
            for col in (xs.index[ci] - 1, xs.index[ci], xs.index[ci] + 1):
                for row in (ys.index[ri] - 1, ys.index[ri], ys.index[ri] + 1):
                    if (col, row) not in tl.index:
                        continue
                    t = tl.loc[(col, row)]
                    r0 = rc + t.dr_use - t.y - W / 2
                    c0 = cc + t.dc_use - t.x - W / 2
                    if 0 <= r0 and r0 + W <= TILE and 0 <= c0 and c0 + W <= TILE:
                        d = min(r0, TILE - r0 - W, c0, TILE - c0 - W)
                        if best is None or d > best[0]:
                            best = (d, t.file, r0, c0, t.dr_use, t.dc_use, t.y, t.x)
            if best is None:
                out.append(dict(i=idx, j=j, n=n, tile=""))
                continue
            _, f, r0, c0, dr, dc, ty, tx = best
            out.append(dict(i=idx, j=j, n=n, tile=f, win_r0=int(round(r0 * S)), win_c0=int(round(c0 * S)),
                            box_r0=int(np.clip(round((R0 + dr - ty - r0) * S), 0, WIN)), box_r1=int(np.clip(round((R1 + dr - ty - r0) * S), 0, WIN)),
                            box_c0=int(np.clip(round((C0 + dc - tx - c0) * S), 0, WIN)), box_c1=int(np.clip(round((C1 + dc - tx - c0) * S), 0, WIN))))
    return pd.DataFrame(out)


def dry() -> None:
    pos = parse_positions(EXTRACTED / "img-global-positions-0.txt")
    offs = robust_offsets(pd.read_csv(WORK / "tile_offsets.csv"))
    offs.to_csv(WORK / "tile_offsets_robust.csv", index=False)
    inst = pickle.load(open(WORK / "instances_reg.pkl", "rb")).reset_index(drop=True)
    t0 = time.time()
    win = windows_for(inst, offs)
    win.to_pickle(WORK / "windows.pkl")
    gsize = inst.groupby(["cell_type", "orient"]).size()
    inst["group_n"] = [gsize[(a, b)] for a, b in zip(inst.cell_type, inst.orient)]
    has_home = win.groupby("i").tile.apply(lambda s: bool((s != "").all()))
    inst["all_windows_home"] = has_home.reindex(inst.index).fillna(False).values
    inst.to_pickle(WORK / "instances_final.pkl")
    rng = np.random.default_rng(SEED)
    refs = {}
    elig = inst[(inst.group_n >= MIN_GROUP) & inst.all_windows_home]
    for (ct, orr), g in elig.groupby(["cell_type", "orient"]):
        if len(g) >= 3:
            refs[(ct, orr)] = sorted(rng.choice(g.index.values, size=min(K_REF, len(g)), replace=False).tolist())
    pickle.dump(refs, open(WORK / "reference_choice.pkl", "wb"))
    stats = dict(instances=len(inst), windows=len(win), windows_without_home=int((win.tile == "").sum()),
                 instances_in_groups_ge4=int((inst.group_n >= MIN_GROUP).sum()), instances_all_windows_home=int(inst.all_windows_home.sum()),
                 instances_verifiable=int(inst.index.isin(elig.index).sum() if len(refs) else 0), groups_with_reference=len(refs),
                 filler_verifiable=int(elig.filler.sum()), functional_verifiable=int((~elig.filler).sum()),
                 tile_offset_source=offs.offset_source.value_counts().to_dict(), seconds=round(time.time() - t0))
    elig_groups = set(refs)
    stats["instances_verifiable"] = int(sum((a, b) in elig_groups for a, b in zip(elig.cell_type, elig.orient)))
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "dry_stats.json").write_text(json.dumps(stats, indent=1))
    print(json.dumps(stats, indent=1))


# --------------------------------------------------------------------------- run (guarded)
def check_prereg() -> str:
    if not _git("log", "-1", "--format=%H", "--", str(PREREG)):
        sys.exit(f"refusing to run: {PREREG} is not committed")
    if _git("status", "--porcelain", "--", str(PREREG)):
        sys.exit(f"refusing to run: {PREREG} has uncommitted changes")
    if (OUT / "RUN_DONE").exists():
        sys.exit(f"refusing to run: {OUT / 'RUN_DONE'} exists (Rung 3 runs once)")
    return _git("rev-parse", "HEAD")


def _ref_windows_task(args):
    tile_file, rows = args
    out = []
    cache: dict = {}
    with zipfile.ZipFile(ZIP) as z:
        for w in rows:
            out.append((w["key"], w["j"], load_window(z, tile_file, pd.Series(w), cache).astype(np.float32)))
    return out


def _build_ref_task(args):
    from nanovate.refcell import build_cell_reference
    key, j, wins = args
    _, _, rcfg = frozen()
    return key, j, build_cell_reference(wins, rcfg)


_REFS: dict = {}


def _verify_tile_task(args):
    from nanovate.refcell import verify_cell
    tile_file, rows = args
    global _REFS
    if not _REFS:
        _REFS.update(pickle.load(open(WORK / "references.pkl", "rb")))
    pc, rc, rcfg = frozen()
    win = Window(WIN, 16)
    out = []
    cache: dict = {}
    with zipfile.ZipFile(ZIP) as z:
        for w in rows:
            ref = _REFS.get((w["key"], w["j"]))
            if ref is None:
                out.append(dict(i=w["i"], j=w["j"], tile=tile_file, verdict="NO_REFERENCE"))
                continue
            im = load_window(z, tile_file, pd.Series(w), cache)
            t0 = time.perf_counter()
            r, d = verify_cell(im, ref, (w["box_r0"], w["box_r1"], w["box_c0"], w["box_c1"]), win, pc, rc, rcfg)
            z_ = r.reason
            out.append(dict(i=w["i"], j=w["j"], tile=tile_file, verdict=r.verdict, d_beta0=z_["delta_betti"][0], d_beta1=z_["delta_betti"][1],
                            n_bridge_events=z_["n_bridge_events"], n_open_events=z_["n_open_events"], bottleneck=z_["bottleneck_max"],
                            max_sdf_energy=z_["max_sdf_energy"], peak_z=z_["peak_z"], align_rejected=z_["align_rejected"],
                            cell_shift_r=d[0], cell_shift_c=d[1], ms=(time.perf_counter() - t0) * 1e3))
    return out


def summarise(res: pd.DataFrame, inst: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    r = res[res.verdict != "NO_REFERENCE"].copy()
    r["killer"] = r.verdict.str.startswith("KILLER")
    r["near_miss"] = r.verdict == "NEAR_MISS"
    r["unverified"] = r.verdict == "UNVERIFIED_CALIBRATION"
    r["topo_change"] = (r.d_beta0.fillna(0) != 0) | (r.d_beta1.fillna(0) != 0)
    agg = r.groupby("i").agg(killer=("killer", "any"), near_miss=("near_miss", "any"), unverified=("unverified", "all"),
                             topo_change=("topo_change", "any"), max_energy=("max_sdf_energy", "max"),
                             abs_dbeta=("d_beta0", lambda s: float(np.abs(s).sum())), home_tile=("tile", "first"), n_windows=("j", "size"))
    agg = agg.join(inst[["cell_type", "orient", "filler", "x0", "y0", "x1", "y1"]])
    agg["near_miss_only"] = agg.near_miss & ~agg.killer
    agg["verified"] = ~agg.unverified
    rows = []
    for scope, g in (("all verifiable instances", agg), ("functional cells", agg[~agg.filler]), ("filler cells", agg[agg.filler])):
        for col, name in (("killer", "KILLER (any window)"), ("near_miss_only", "NEAR_MISS (no killer window)"),
                          ("topo_change", "topology change (dbeta != 0 in any window)"), ("verified", "verified (calibration reliable)")):
            m, lo, hi, n, ni = _cluster_ci(g.rename(columns={"home_tile": "image_id"}).assign(**{col: g[col].astype(bool)}), col)
            rows.append(dict(scope=scope, metric=name, mean=m, ci_lo=lo, ci_hi=hi, n_instances=n, n_tiles=ni))
    rows.append(dict(scope="all instances", metric="coverage = verified instances / all GDS instances", mean=float(agg.verified.sum()) / len(inst),
                     ci_lo=np.nan, ci_hi=np.nan, n_instances=len(inst), n_tiles=np.nan))
    return pd.DataFrame(rows), agg


def run(workers: int) -> None:
    commit = check_prereg()
    t0 = time.time()
    inst = pd.read_pickle(WORK / "instances_final.pkl")
    win = pd.read_pickle(WORK / "windows.pkl")
    refs = pickle.load(open(WORK / "reference_choice.pkl", "rb"))
    key_of = {i: (ct, o) for i, ct, o in zip(inst.index, inst.cell_type, inst.orient)}
    win = win[win.tile != ""].copy()
    win["key"] = [key_of[i] for i in win.i]
    # pass A: reference windows
    want = []
    for key, ids in refs.items():
        for i in ids:
            want.append(win[win.i == i])
    W = pd.concat(want)
    tasks = [(f, g.to_dict("records")) for f, g in W.groupby("tile")]
    with ProcessPoolExecutor(workers) as ex:
        got = [x for part in ex.map(_ref_windows_task, tasks, chunksize=2) for x in part]
    by = {}
    for key, j, w in got:
        by.setdefault((key, j), []).append(w)
    with ProcessPoolExecutor(workers) as ex:
        built = list(ex.map(_build_ref_task, [(k[0], k[1], v) for k, v in by.items() if len(v) >= 3], chunksize=4))
    REFS = {(k, j): r for k, j, r in built}
    pickle.dump(REFS, open(WORK / "references.pkl", "wb"))
    t_a = time.time() - t0
    # pass B: verify every instance window on its home tile
    todo = win[win.key.isin(set(refs))]
    tasks = [(f, g.to_dict("records")) for f, g in todo.groupby("tile")]
    with ProcessPoolExecutor(workers) as ex:
        res = pd.DataFrame([x for part in ex.map(_verify_tile_task, tasks) for x in part])
    OUT.mkdir(parents=True, exist_ok=True)
    res.to_csv(OUT / "windows.csv.gz", index=False)
    s, agg = summarise(res, inst)
    s.to_csv(OUT / "summary.csv", index=False)
    agg.to_csv(OUT / "instances.csv.gz")
    cand = agg[agg.verified].sort_values(["killer", "abs_dbeta", "topo_change", "max_energy"], ascending=False).head(200)
    cand.to_csv(OUT / "candidates_top200.csv")
    rel = {f"{k[0][0]}|{k[0][1]}|{k[1]}": bool(r.rcal.reliable) for k, r in REFS.items()}
    (OUT / "config.json").write_text(json.dumps(dict(data="REAL backside SEM, 90 nm chip (Puschner et al.)", reference="median of same-type instances (no polygon CAD)",
                                                     prereg=str(PREREG), prereg_commit=commit, scale=S, window=WIN, k_ref=K_REF, min_group=MIN_GROUP,
                                                     references_built=len(REFS), references_reliable=int(sum(rel.values())),
                                                     wall_s=time.time() - t0, pass_a_s=t_a), indent=2))
    (OUT / "RUN_DONE").write_text(json.dumps(dict(commit=commit, finished=time.strftime("%Y-%m-%d %H:%M:%S")), indent=2))
    print(s.round(4).to_string())


def main() -> None:
    ap = argparse.ArgumentParser()
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument("--register", action="store_true")
    m.add_argument("--dry", action="store_true")
    m.add_argument("--run", action="store_true")
    ap.add_argument("--workers", type=int, default=6)
    a = ap.parse_args()
    if a.register:
        register(a.workers)
        return
    if a.dry:
        dry()
        return
    run(a.workers)


if __name__ == "__main__":
    main()


def load_window(z, tile_file: str, w, cache: dict | None = None) -> np.ndarray:
    """The x0.5 window for one windows-row (block-mean downsampled tile, cached per tile)."""
    if cache is not None and tile_file in cache:
        d = cache[tile_file]
    else:
        a = read_tile(z, tile_file)
        d = a.reshape(TILE // 2, 2, TILE // 2, 2).mean((1, 3))
        if cache is not None:
            cache.clear(); cache[tile_file] = d
    r0, c0 = int(w.win_r0), int(w.win_c0)
    return d[r0:r0 + WIN, c0:c0 + WIN]
