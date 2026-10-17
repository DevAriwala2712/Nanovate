"""Select the verifier configuration on Manhattan *val* only, then freeze it (AGENTS.md §5.2, PLAN §4).

Pre-registered criterion (PLAN §4.1): among candidates {field option} x {region mode} x {tau}, maximise
killer F1 on Manhattan val pooled over doses 20/50/200, subject to nuisance suppression >= the
suppression of the best-F1 percentile-field (option a) candidate; ties -> fewer fake bridges on
nuisance samples, then lower tau. Then d*, e* (near-miss rule, PLAN §4.2 fallback): the largest
near-miss recall (notch / filament flagged NEAR_MISS) such that NEAR_MISS fires on <= 5% of nuisance
samples. Pre-smoothing (1 px), region_tol (8) and the SDF constants are fixed, not searched.
Writes results/calib/{candidates.csv, near_miss_grid.csv, decisions.json}. Synthetic data.
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


def results_root(cfg: ExpCfg) -> Path:
    """results/ (full) or results/quick/; NANOVATE_RESULTS_ROOT overrides the base (used by tests)."""
    base = Path(os.environ.get("NANOVATE_RESULTS_ROOT", "results"))
    return base / "quick" if cfg.quick else base


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


def run(cfg: ExpCfg) -> dict:
    root = results_root(cfg) / "calib"
    print("selecting verifier config on Manhattan val ...", flush=True)
    rows = candidate_rows(cfg)
    cands = summarise(rows)
    best, floor = choose(cands)
    pc = PipelineCfg(field_option=best.field, tau=float(best.tau), pre_smooth=PRE_SMOOTH, region_mode=best["mode"], region_tol=REGION_TOL)
    d_star, e_star, grid = near_miss_thresholds(cfg, pc)
    pc = dataclasses.replace(pc, d_star=d_star, e_star=e_star)
    decisions = {
        "pipeline_cfg": {k: (None if isinstance(v, float) and np.isinf(v) else v) for k, v in pc.to_dict().items()},
        "criterion": "max killer F1 on Manhattan val (doses 20/50/200) s.t. nuisance suppression >= best percentile-field candidate",
        "suppression_floor": floor, "chosen": {k: (v.item() if hasattr(v, "item") else v) for k, v in best.to_dict().items()},
        "near_miss_budget": NM_FA, "n_val_patches_per_dose": cfg.val_layouts * cfg.val_dies * len(KINDS), "quick": cfg.quick, "seed": cfg.seed,
    }
    write_results(root, "candidates", cands, {"exp_cfg": cfg.to_dict(), "taus": TAUS, "doses": SEL_DOSES})
    grid.to_csv(root / "near_miss_grid.csv", index=False)
    (root / "decisions.json").write_text(json.dumps(decisions, indent=2, default=str))
    print(f"chosen: {decisions['chosen']}\n d*={d_star}, e*={e_star}", flush=True)
    return decisions


def load_pipeline_cfg(cfg: ExpCfg) -> PipelineCfg:
    """Frozen verifier config from decisions.json (null thresholds -> rule off)."""
    d = json.loads((results_root(cfg) / "calib" / "decisions.json").read_text())["pipeline_cfg"]
    d = {k: (float("inf") if v is None and k in ("d_star", "e_star") else v) for k, v in d.items()}
    return PipelineCfg(**d)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    run(ExpCfg.quick() if a.quick else ExpCfg.full())
