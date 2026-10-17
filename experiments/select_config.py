"""Select the verifier configuration on Manhattan *val* only, then freeze it (AGENTS.md §5.2, PLAN §4).

Pre-registered criterion (PLAN §4.1): among candidates {field option} x {region mode} x {tau}, maximise
killer F1 on Manhattan val pooled over doses 20/50/200, subject to nuisance suppression >= the
suppression of the best-F1 percentile-field (option a) candidate; ties -> fewer fake bridges on
nuisance samples, then lower tau. Then d*, e* (near-miss rule, PLAN §4.2 fallback): the largest
near-miss recall (notch / filament flagged NEAR_MISS) such that NEAR_MISS fires on <= 5% of nuisance
samples. Pre-smoothing (1 px), region_tol (8) and the SDF constants are fixed, not searched.

v2 (post-hoc, after the v1 test results were seen; see docs/V2_FIXES.md): (1) holes are counted with
dielectric outside the relevance region (``h1_fill='dielectric'``, removes splice-seam holes), in the
candidate grid as in the pipeline; (2) the near-miss rule adds the label-free peak-z statistic
(``z_star``) and its thresholds are chosen so that NEAR_MISS fires on <= 5% of val nuisance in EVERY
val nuisance cell (LER sigma 0.5/1.0/1.5 nm x grain 0/15/30%, doses pooled) instead of at the nominal
nuisance only. Both use Manhattan val only; targets are unchanged.
Writes results/[v2/]calib/{candidates.csv, near_miss_grid.csv, decisions.json}. Synthetic data.
Usage: python -m experiments.select_config [--quick]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.common import ExpCfg, f1_score, write_results
from nanovate.config import TestbedCfg
from nanovate.modules.align_sdf import align_to_template
from nanovate.modules.calibration import calibrate, features
from nanovate.modules.fields import build_field
from nanovate.modules.localpers import local_persistence
from nanovate.modules.topology import (
    NUISANCE, KILLER_BRIDGE, betti_from_intervals, classify_verdict, compute_persistence_intervals,
)
from nanovate.pipeline import PipelineCfg, verify_patch
from nanovate.testbed.dataset import KINDS, get_layout, make_sample, window
from nanovate.testbed.scribe import make_scribe_patches
from nanovate.topo_utils import relevance_region

SEL_DOSES = (20, 50, 200)
FIELDS = ("percentile", "calibrated", "template")
MODES = ("splice", "pairs")
TAUS = tuple(round(t, 2) for t in np.arange(0.30, 0.701, 0.05))
PRE_SMOOTH, REGION_TOL = 1.0, 8
NM_FA = 0.05          # pre-registered near-miss false-alarm budget on nuisance samples
KILLERS = ("bridge", "open", "particle")
NM_CELLS_SIGMA, NM_CELLS_GRAIN = (0.5, 1.0, 1.5), (0.0, 0.15, 0.30)   # v2 robust near-miss selection (val)


def results_root(cfg: ExpCfg) -> Path:
    """results/[v2/] (full) or results/[v2/]quick/; NANOVATE_RESULTS_ROOT overrides the base (used by tests).

    v1 (the first, pre-registered run) lives in results/; v2 writes beside it in results/v2/ and never over it.
    """
    base = Path(os.environ.get("NANOVATE_RESULTS_ROOT", "results"))
    if cfg.version != "v1":
        base = base / cfg.version
    return base / "quick" if cfg.quick else base


def h1_fill(cfg: ExpCfg) -> str:
    return "cad" if cfg.version == "v1" else "dielectric"


def with_dose(base: TestbedCfg, dose: float) -> TestbedCfg:
    return dataclasses.replace(base, sem=dataclasses.replace(base.sem, dose=float(dose)))


def candidate_rows(cfg: ExpCfg) -> pd.DataFrame:
    """Per (patch, field, mode, tau): the Delta-beta verdict, for every candidate configuration."""
    base = TestbedCfg()
    win = window(base)
    rows = []
    for dose in SEL_DOSES:
        tcfg = with_dose(base, dose)
        cal = calibrate(make_scribe_patches(tcfg, seed=cfg.seed))
        contrast = cal.mu_cond - cal.mu_diel
        for lay in range(cfg.val_layouts):
            L = get_layout("manhattan", "val", lay, tcfg, cfg.seed)
            cadf = L.cad.astype(float)
            reg = relevance_region(L.cad, win, tol=REGION_TOL)
            T = cal.predict(L.cad)
            cs = features(L.cad, *cal.sigmas)[..., 1]
            d0c, d1c = compute_persistence_intervals(cadf)
            if h1_fill(cfg) == "dielectric":
                d1c = compute_persistence_intervals(np.where(reg, cadf, 0.0))[1]
            ref_splice = {t: betti_from_intervals(d0c, d1c, t) for t in TAUS}
            ref_pairs = local_persistence(cadf, reg)
            for die in range(cfg.val_dies):
                for kind in KINDS:
                    s = make_sample("manhattan", "val", lay, die, kind, tcfg, cfg.seed, layout=L)
                    aligned, _ = align_to_template(s.image, T)
                    for fopt in FIELDS:
                        kw = dict(template=T, cad_smooth=cs, contrast=contrast) if fopt == "template" else {}
                        field, _ = build_field(aligned, fopt, levels=cal.levels, pre_smooth=PRE_SMOOTH, **kw)
                        d0, d1 = compute_persistence_intervals(np.where(reg, field, cadf))
                        if h1_fill(cfg) == "dielectric":
                            d1 = compute_persistence_intervals(np.where(reg, field, 0.0))[1]
                        lp, lpr = local_persistence(field, reg), ref_pairs
                        for t in TAUS:
                            for mode in MODES:
                                if mode == "splice":
                                    b, r = betti_from_intervals(d0, d1, t), ref_splice[t]
                                else:
                                    b, r = lp.betti(t), lpr.betti(t)
                                db = (b[0] - r[0], b[1] - r[1])
                                rows.append((dose, lay, die, kind, fopt, mode, t, db[0], db[1], classify_verdict(db)))
        print(f"  candidates: dose {dose} done", flush=True)
    return pd.DataFrame(rows, columns=["dose", "layout", "die", "kind", "field", "mode", "tau", "d_b0", "d_b1", "verdict"])


def summarise(rows: pd.DataFrame) -> pd.DataFrame:
    out = []
    for (fopt, mode, tau), g in rows.groupby(["field", "mode", "tau"]):
        y = g.kind.isin(KILLERS).values
        pred = g.verdict.str.startswith("KILLER").values
        nu = g[g.kind == "nuisance"]
        out.append(dict(field=fopt, mode=mode, tau=tau, n=len(g), n_killer=int(y.sum()), n_nuisance=len(nu),
                        f1=f1_score(pred, y), suppression=float((nu.verdict == NUISANCE).mean()),
                        fake_bridges=int((nu.verdict == KILLER_BRIDGE).sum())))
    return pd.DataFrame(out)


def choose(cands: pd.DataFrame) -> tuple[pd.Series, float]:
    """Pre-registered rule. Returns (chosen row, suppression floor from the best option-(a) candidate)."""
    a = cands[cands.field == "percentile"].sort_values(["f1", "suppression"], ascending=False).iloc[0]
    ok = cands[cands.suppression >= a.suppression - 1e-12]
    best = ok.sort_values(["f1", "fake_bridges", "tau"], ascending=[False, True, True]).iloc[0]
    return best, float(a.suppression)


def near_miss_thresholds(cfg: ExpCfg, pc: PipelineCfg) -> tuple[float, float, pd.DataFrame]:
    """Grid (d*, e*) on val with the chosen config; largest near-miss recall at <= NM_FA nuisance alarms."""
    if cfg.version != "v1":
        raise ValueError("v2 uses near_miss_thresholds_robust")
    base = TestbedCfg()
    win = window(base)
    rec = []
    for dose in SEL_DOSES:
        tcfg = with_dose(base, dose)
        cal = calibrate(make_scribe_patches(tcfg, seed=cfg.seed))
        for lay in range(cfg.val_layouts):
            L = get_layout("manhattan", "val", lay, tcfg, cfg.seed)
            for die in range(cfg.val_dies):
                for kind in ("nuisance", "notch", "filament"):
                    s = make_sample("manhattan", "val", lay, die, kind, tcfg, cfg.seed, layout=L)
                    r = verify_patch(s.image, s.cad, cal, win, pc).reason
                    rec.append(dict(dose=dose, kind=kind, d_b=tuple(r["delta_betti"]), bott=r["bottleneck_max"], energy=r["max_sdf_energy"]))
        print(f"  near-miss: dose {dose} done", flush=True)
    df = pd.DataFrame(rec)
    zero = df.d_b.map(lambda t: t == (0, 0)).values
    nu, nm = (df.kind == "nuisance").values, df.kind.isin(["notch", "filament"]).values
    grid = []
    for d in np.unique(np.quantile(df.bott[nu & zero], np.linspace(0, 0.98, 25))):
        for e in np.unique(np.quantile(df.energy[nu & zero], np.linspace(0, 0.98, 25))):
            alarm = zero & (df.bott.values >= d) & (df.energy.values >= e)
            grid.append(dict(d_star=float(d), e_star=float(e), nuisance_alarm=float(alarm[nu].mean()),
                             nearmiss_recall=float(alarm[nm].mean()), n_nuisance=int(nu.sum()), n_nearmiss=int(nm.sum())))
    G = pd.DataFrame(grid)
    ok = G[G.nuisance_alarm <= NM_FA + 1e-12]
    if ok.empty:  # no threshold pair meets the budget: turn the rule off (honest: recall 0)
        return float("inf"), float("inf"), G
    best = ok.sort_values(["nearmiss_recall", "d_star", "e_star"], ascending=[False, False, False]).iloc[0]
    return float(best.d_star), float(best.e_star), G


def _nm_records(cfg: ExpCfg, pc: PipelineCfg, tcfg: TestbedCfg, kinds: tuple, **tags) -> list[dict]:
    win = window(tcfg)
    cal = calibrate(make_scribe_patches(tcfg, seed=cfg.seed))
    rec = []
    for lay in range(cfg.val_layouts):
        L = get_layout("manhattan", "val", lay, tcfg, cfg.seed)
        for die in range(cfg.val_dies):
            for kind in kinds:
                s = make_sample("manhattan", "val", lay, die, kind, tcfg, cfg.seed, layout=L)
                r = verify_patch(s.image, s.cad, cal, win, pc).reason
                rec.append(dict(dose=tcfg.sem.dose, kind=kind, d_b=tuple(r["delta_betti"]), bott=r["bottleneck_max"],
                                energy=r["max_sdf_energy"], z=r["peak_z"], **tags))
    return rec


def near_miss_thresholds_robust(cfg: ExpCfg, pc: PipelineCfg) -> tuple[float, float, float, pd.DataFrame]:
    """v2: grid (d*, e*, z*) on Manhattan val. Objective as v1 (near-miss recall at the nominal nuisance,
    doses pooled); constraint: NEAR_MISS on <= NM_FA of nuisance samples in every (LER sigma, grain) val cell."""
    from experiments.harness import nuisance_cfg
    base = TestbedCfg()
    rec = []
    for dose in SEL_DOSES:
        nom = with_dose(base, dose)
        rec += _nm_records(cfg, pc, nom, ("notch", "filament"), cell="nominal")
        for sg in NM_CELLS_SIGMA:
            for gr in NM_CELLS_GRAIN:
                rec += _nm_records(cfg, pc, with_dose(nuisance_cfg(base, sg, gr), dose), ("nuisance",), cell=f"s{sg}_g{gr}")
        print(f"  near-miss (robust): dose {dose} done", flush=True)
    df = pd.DataFrame(rec)
    zero = df.d_b.map(lambda t: t == (0, 0)).values
    nu, nm = (df.kind == "nuisance").values, df.kind.isin(["notch", "filament"]).values
    cells = df.cell.values

    def qs(x, k):
        return np.unique(np.quantile(x, np.linspace(0, 0.995, k)))
    grid = []
    for d in qs(df.bott[nu & zero], 20):
        for e in qs(df.energy[nu & zero], 30):
            for z in np.r_[-np.inf, qs(df.z[nu & zero], 20)]:
                alarm = zero & (df.bott.values >= d) & (df.energy.values >= e) & (df.z.values >= z)
                worst = max(float(alarm[nu & (cells == c)].mean()) for c in np.unique(cells[nu]))
                grid.append(dict(d_star=float(d), e_star=float(e), z_star=float(z), worst_cell_nuisance_alarm=worst,
                                 nuisance_alarm=float(alarm[nu].mean()), nearmiss_recall=float(alarm[nm].mean()),
                                 n_nuisance=int(nu.sum()), n_nearmiss=int(nm.sum()), n_cells=len(np.unique(cells[nu]))))
    G = pd.DataFrame(grid)
    ok = G[G.worst_cell_nuisance_alarm <= NM_FA + 1e-12]
    if ok.empty:
        return float("inf"), float("inf"), float("-inf"), G
    best = ok.sort_values(["nearmiss_recall", "worst_cell_nuisance_alarm", "d_star", "e_star", "z_star"],
                          ascending=[False, True, False, False, False]).iloc[0]
    return float(best.d_star), float(best.e_star), float(best.z_star), G


def run(cfg: ExpCfg) -> dict:
    root = results_root(cfg) / "calib"
    print("selecting verifier config on Manhattan val ...", flush=True)
    rows = candidate_rows(cfg)
    cands = summarise(rows)
    best, floor = choose(cands)
    pc = PipelineCfg(field_option=best.field, tau=float(best.tau), pre_smooth=PRE_SMOOTH, region_mode=best["mode"], region_tol=REGION_TOL,
                     h1_fill=h1_fill(cfg))
    if cfg.version == "v1":
        d_star, e_star, grid = near_miss_thresholds(cfg, pc)
        z_star = float("-inf")
    else:
        d_star, e_star, z_star, grid = near_miss_thresholds_robust(cfg, pc)
    pc = dataclasses.replace(pc, d_star=d_star, e_star=e_star, z_star=z_star)
    decisions = {
        "pipeline_cfg": {k: (None if isinstance(v, float) and np.isinf(v) else v) for k, v in pc.to_dict().items()},
        "criterion": "max killer F1 on Manhattan val (doses 20/50/200) s.t. nuisance suppression >= best percentile-field candidate",
        "suppression_floor": floor, "chosen": {k: (v.item() if hasattr(v, "item") else v) for k, v in best.to_dict().items()},
        "near_miss_budget": NM_FA, "version": cfg.version,
        "near_miss_constraint": "nominal nuisance" if cfg.version == "v1" else f"every val cell sigma {NM_CELLS_SIGMA} x grain {NM_CELLS_GRAIN}",
        "n_val_patches_per_dose": cfg.val_layouts * cfg.val_dies * len(KINDS), "quick": cfg.quick, "seed": cfg.seed,
    }
    write_results(root, "candidates", cands, {"exp_cfg": cfg.to_dict(), "taus": TAUS, "doses": SEL_DOSES})
    grid.to_csv(root / "near_miss_grid.csv", index=False)
    (root / "decisions.json").write_text(json.dumps(decisions, indent=2, default=str))
    print(f"chosen: {decisions['chosen']}\n d*={d_star}, e*={e_star}, z*={z_star}", flush=True)
    return decisions


def load_pipeline_cfg(cfg: ExpCfg) -> PipelineCfg:
    """Frozen verifier config from decisions.json (null thresholds -> rule off)."""
    d = json.loads((results_root(cfg) / "calib" / "decisions.json").read_text())["pipeline_cfg"]
    d = {k: (float("inf") if v is None and k in ("d_star", "e_star") else v) for k, v in d.items()}
    d = {k: (float("-inf") if v is None and k == "z_star" else v) for k, v in d.items()}
    return PipelineCfg(**d)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    run(ExpCfg.quick() if a.quick else ExpCfg.full())
