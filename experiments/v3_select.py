"""v3: select the located-event rule on Manhattan VAL only, freeze it (docs/V3_PHASE2_DESIGN.md). Synthetic data.

Selection set: Manhattan val layouts 0..7 x 2 dies x 6 kinds, doses 20/50/200, every (LER sigma, grain) cell of
v2's robust near-miss selection (sigma 0.5/1.0/1.5 nm x grain 0/15/30%) -> 2,592 patches. Evidence is extracted
once per patch; the rule grid is applied vectorised.

Pre-registered criterion (design note): maximise killer F1 pooled over the selection set, subject to nuisance
suppression (no killer event) on the nominal cell (sigma 1.0, grain 15%, doses pooled) >= v2's floor (0.708,
results/v2/calib/decisions.json). Ties -> fewer options switched on, then larger m_merge, larger m_norm,
larger m_cut, larger e_edge (deterministic). Then d*, e*, z* re-selected with v2's robust procedure
(near-miss recall at the nominal cell s.t. NEAR_MISS <= 5% of nuisance in every cell), "no killer event"
replacing "Delta-beta = 0". tau, field, region and pre-smoothing stay v2's. pair_dist and body_radius are
fixed a priori. The test split is never generated here.
Writes results/v3/calib/{rule_grid.csv, near_miss_grid.csv, val_records.csv, decisions.json}.
Usage: python -m experiments.v3_select [--quick]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.common import f1_score
from experiments.harness import nuisance_cfg
from experiments.select_config import NM_FA, with_dose
from nanovate.config import TestbedCfg
from nanovate.modules.calibration import calibrate
from nanovate.modules.events import EventRuleCfg
from nanovate.pipeline import PipelineCfg
from nanovate.pipeline_v3 import patch_evidence
from nanovate.testbed.dataset import KINDS, get_layout, make_sample, window
from nanovate.testbed.scribe import make_scribe_patches

V2_DECISIONS = Path("results/v2/calib/decisions.json")
KILLERS = ("bridge", "open", "particle")
M_MERGE = tuple(round(x, 2) for x in np.arange(0.20, 0.801, 0.02))
M_NORM = (None, 0.6, 0.7, 0.8, 0.9)
M_CUT = (None, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55, 0.60)
E_EDGE = (None, 0.30, 0.40, 0.50, 0.60)


@dataclass(frozen=True)
class V3Cfg:
    """v3 run settings (kept apart from ExpCfg so v1/v2 configs stay byte-identical)."""

    quick: bool = False
    seed: int = 0                      # val, scribe, D2D rules, CNN threshold: as v2
    test_seed: int = 20261018          # fresh test seed: no version has generated test data with it (used after STOP POINT 3)
    val_layouts: int = 8
    val_dies: int = 2
    sel_doses: tuple = (20, 50, 200)
    cells_sigma: tuple = (0.5, 1.0, 1.5)
    cells_grain: tuple = (0.0, 0.15, 0.30)
    nominal: tuple = (1.0, 0.15)
    workers: int = 6
    n_boot: int = 1000

    @classmethod
    def quick_cfg(cls) -> "V3Cfg":
        return cls(quick=True, val_layouts=2, val_dies=1, sel_doses=(50,), cells_sigma=(1.0,), cells_grain=(0.15, 0.30), workers=4, n_boot=200)

    def to_dict(self) -> dict:
        return asdict(self)


def v2_pc() -> PipelineCfg:
    return PipelineCfg(**json.loads(V2_DECISIONS.read_text())["pipeline_cfg"])


def root(cfg: V3Cfg) -> Path:
    return Path("results/v3/quick" if cfg.quick else "results/v3")


_CAL: dict = {}


def _cal(tcfg: TestbedCfg, seed: int):
    key = (tcfg.sem.dose, tcfg.nuisance.ler_sigma, tcfg.nuisance.grain_contrast)
    if key not in _CAL:
        _CAL[key] = calibrate(make_scribe_patches(tcfg, seed=seed))
    return _CAL[key]


def _record(spec: dict):
    tcfg = with_dose(nuisance_cfg(TestbedCfg(), spec["sigma"], spec["grain"]), spec["dose"])
    pc = PipelineCfg(**spec["pc"])
    L = get_layout("manhattan", "val", spec["layout"], tcfg, spec["seed"])
    s = make_sample("manhattan", "val", spec["layout"], spec["die"], spec["kind"], tcfg, spec["seed"], layout=L)
    img = s.expected if spec.get("noise_free") else s.image
    ev, sc, t = patch_evidence(img, s.cad, _cal(tcfg, spec["seed"]), window(tcfg), pc)
    meta = {k: spec[k] for k in ("dose", "sigma", "grain", "layout", "die", "kind")} | dict(killer=s.killer, noise_free=bool(spec.get("noise_free")))
    meta.update({f"p_{k}": float(v) for k, v in s.params.get("defect", {}).items() if isinstance(v, (int, float))})
    return meta, ev, sc


def collect(cfg: V3Cfg, specs: list[dict]) -> list:
    with ProcessPoolExecutor(cfg.workers) as ex:
        return list(ex.map(_record, specs, chunksize=4))


def selection_specs(cfg: V3Cfg, pc: PipelineCfg) -> list[dict]:
    return [dict(seed=cfg.seed, pc=pc.to_dict(), dose=float(d), sigma=sg, grain=gr, layout=lay, die=die, kind=k)
            for d in cfg.sel_doses for sg in cfg.cells_sigma for gr in cfg.cells_grain
            for lay in range(cfg.val_layouts) for die in range(cfg.val_dies) for k in KINDS]


# ------------------------------------------------------------------ vectorised rule grid
def merge_combos():
    return [(m, n) for n in M_NORM for m in M_MERGE]


def open_combos():
    return [(c, e) for c in M_CUT for e in E_EDGE]


def bridge_flags(ev, sc_tau: float, combos) -> np.ndarray:
    """(len(combos),) bool: any bridge event (pair merge or new hole) under each merge combo."""
    from nanovate.modules.events import _hole_events
    new, _ = _hole_events(ev, sc_tau)
    if new:
        return np.ones(len(combos), bool)
    if not ev.pairs:
        return np.zeros(len(combos), bool)
    s = np.array([p["s"] for p in ev.pairs]); e = np.array([p["e"] for p in ev.pairs]); b = np.array([p["body"] for p in ev.pairs])
    ok = ~(np.isnan(s) | np.isnan(e))
    s, e, b = s[ok], e[ok], b[ok]
    out = np.zeros(len(combos), bool)
    for k, (m, n) in enumerate(combos):
        hit = (s - e) >= m
        if n is not None:
            hit &= (s / np.maximum(b, 1e-9)) >= n
        out[k] = bool(hit.any())
    return out


def open_flags(ev, tau: float, combos) -> np.ndarray:
    from nanovate.modules.events import v3_events
    return np.array([bool(v3_events(ev, EventRuleCfg(m_merge=np.inf, m_cut=c, e_edge=e), tau)["open"]) for c, e in combos])


def grid_table(meta: pd.DataFrame, B: np.ndarray, O: np.ndarray, mc, oc, nominal_mask: np.ndarray) -> pd.DataFrame:
    y = meta.killer.values
    nu = (meta.kind == "nuisance").values
    rows = []
    for i, (m, n) in enumerate(mc):
        flag = B[:, i][:, None] | O                                # (patches, open combos)
        tp = (flag & y[:, None]).sum(0); fp = (flag & ~y[:, None]).sum(0); fn = (~flag & y[:, None]).sum(0)
        f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1)
        supp = (~flag[nu & nominal_mask]).mean(0)
        for j, (c, e) in enumerate(oc):
            rows.append(dict(m_merge=m, m_norm=n, m_cut=c, e_edge=e, f1=float(f1[j]), suppression_nominal=float(supp[j]),
                             n=len(y), n_killer=int(y.sum()), n_nuisance_nominal=int((nu & nominal_mask).sum()),
                             n_options=int(n is not None) + int(c is not None) + int(e is not None)))
    return pd.DataFrame(rows)


def choose(G: pd.DataFrame, floor: float) -> pd.Series:
    ok = G[G.suppression_nominal >= floor - 1e-12].copy()
    if ok.empty:
        raise RuntimeError("no rule meets the v2 suppression floor on val")
    for c in ("m_norm", "m_cut", "e_edge"):
        ok[c + "_key"] = ok[c].fillna(-1.0)
    ok["f1_r"] = ok.f1.round(12)
    return ok.sort_values(["f1_r", "n_options", "m_merge", "m_norm_key", "m_cut_key", "e_edge_key"],
                          ascending=[False, True, False, False, False, False]).iloc[0]


def near_miss(meta: pd.DataFrame, killer_flag: np.ndarray, sc: pd.DataFrame, nominal_mask: np.ndarray) -> tuple[float, float, float, pd.DataFrame]:
    """v2's robust (d*, e*, z*) procedure with 'no killer event' as the zero condition."""
    zero = ~killer_flag
    nu = (meta.kind == "nuisance").values
    nm = meta.kind.isin(["notch", "filament"]).values & nominal_mask
    cells = (meta.sigma.astype(str) + "_" + meta.grain.astype(str)).values
    bott, en, z = sc.bottleneck.values, sc.energy.values, sc.peak_z.values

    def qs(x, k):
        return np.unique(np.quantile(x, np.linspace(0, 0.995, k)))
    grid = []
    for d in qs(bott[nu & zero], 20):
        for e in qs(en[nu & zero], 30):
            for zz in np.r_[-np.inf, qs(z[nu & zero], 20)]:
                alarm = zero & (bott >= d) & (en >= e) & (z >= zz)
                worst = max(float(alarm[nu & (cells == c)].mean()) for c in np.unique(cells[nu]))
                grid.append(dict(d_star=float(d), e_star=float(e), z_star=float(zz), worst_cell_nuisance_alarm=worst,
                                 nuisance_alarm=float(alarm[nu].mean()), nearmiss_recall=float(alarm[nm].mean()),
                                 n_nuisance=int(nu.sum()), n_nearmiss=int(nm.sum()), n_cells=len(np.unique(cells[nu]))))
    G = pd.DataFrame(grid)
    ok = G[G.worst_cell_nuisance_alarm <= NM_FA + 1e-12]
    if ok.empty:
        return float("inf"), float("inf"), float("-inf"), G
    best = ok.sort_values(["nearmiss_recall", "worst_cell_nuisance_alarm", "d_star", "e_star", "z_star"],
                          ascending=[False, True, False, False, False]).iloc[0]
    return float(best.d_star), float(best.e_star), float(best.z_star), G


def run(cfg: V3Cfg) -> dict:
    out = root(cfg) / "calib"
    out.mkdir(parents=True, exist_ok=True)
    v2 = json.loads(V2_DECISIONS.read_text())
    pc = v2_pc()
    floor = float(v2["suppression_floor"])
    print(f"collecting val evidence ({len(selection_specs(cfg, pc))} patches) ...", flush=True)
    res = collect(cfg, selection_specs(cfg, pc))
    meta = pd.DataFrame([m for m, _, _ in res])
    sc = pd.DataFrame([{k: v for k, v in s.items() if k != "shift"} for _, _, s in res])
    evs = [e for _, e, _ in res]
    nominal = ((meta.sigma == cfg.nominal[0]) & (meta.grain == cfg.nominal[1])).values
    mc, oc = merge_combos(), open_combos()
    B = np.stack([bridge_flags(ev, pc.tau, mc) for ev in evs])
    O = np.stack([open_flags(ev, pc.tau, oc) for ev in evs])
    G = grid_table(meta, B, O, mc, oc, nominal)
    best = choose(G, floor)
    nn = lambda v: None if pd.isna(v) else float(v)  # noqa: E731
    rc = EventRuleCfg(m_merge=float(best.m_merge), normalise=best.m_norm is not None and not pd.isna(best.m_norm),
                      m_norm=nn(best.m_norm) or 0.0, m_cut=nn(best.m_cut), e_edge=nn(best.e_edge))
    i, j = mc.index((rc.m_merge, None if not rc.normalise else rc.m_norm)), oc.index((rc.m_cut, rc.e_edge))
    kflag = B[:, i] | O[:, j]
    d_star, e_star, z_star, NG = near_miss(meta, kflag, sc, nominal)
    pc3 = dataclasses.replace(pc, d_star=d_star, e_star=e_star, z_star=z_star)
    decisions = {
        "version": "v3", "data": "synthetic", "split": "val (Manhattan)", "seed": cfg.seed, "test_seed": cfg.test_seed,
        "pipeline_cfg": {k: (None if isinstance(v, float) and np.isinf(v) else v) for k, v in pc3.to_dict().items()},
        "event_rule_cfg": rc.to_dict(),
        "criterion": "max killer F1 pooled over val doses x (LER sigma, grain) cells s.t. nominal-cell suppression >= v2 floor",
        "suppression_floor_v2": floor,
        "chosen": {k: (None if (isinstance(v, float) and np.isnan(v)) else (v.item() if hasattr(v, "item") else v))
                   for k, v in best.drop(labels=[c for c in best.index if c.endswith("_key") or c == "f1_r"]).to_dict().items()},
        "near_miss_budget": NM_FA, "near_miss_constraint": f"every val cell sigma {cfg.cells_sigma} x grain {cfg.cells_grain}",
        "grids": {"m_merge": M_MERGE, "m_norm": M_NORM, "m_cut": M_CUT, "e_edge": E_EDGE},
        "fixed_a_priori": {"pair_dist_px": rc.pair_dist, "body_radius_px": rc.body_radius, "tau": pc.tau, "field": pc.field_option,
                           "pre_smooth_px": pc.pre_smooth, "region_tol_px": pc.region_tol},
        "v3_cfg": cfg.to_dict(), "n_val_patches": len(meta),
    }
    G.to_csv(out / "rule_grid.csv", index=False)
    NG.to_csv(out / "near_miss_grid.csv", index=False)
    rec = meta.join(sc.drop(columns=["delta_betti_v2"]))
    rec["v3_killer_flag"] = kflag
    rec["v3_bridge_flag"] = B[:, i]
    rec["v2_d_b0"] = [d[0] for d in sc.delta_betti_v2]
    rec["v2_d_b1"] = [d[1] for d in sc.delta_betti_v2]
    rec.to_csv(out / "val_records.csv", index=False)
    (out / "decisions.json").write_text(json.dumps(decisions, indent=2, default=str))
    print("chosen:", decisions["chosen"], "\nnear-miss:", d_star, e_star, z_star, flush=True)
    return decisions


def load_frozen(cfg: V3Cfg) -> tuple[PipelineCfg, EventRuleCfg]:
    d = json.loads((root(cfg) / "calib" / "decisions.json").read_text())
    p = d["pipeline_cfg"]
    p = {k: (float("inf") if v is None and k in ("d_star", "e_star") else float("-inf") if v is None and k == "z_star" else v) for k, v in p.items()}
    return PipelineCfg(**p), EventRuleCfg(**d["event_rule_cfg"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); a = ap.parse_args()
    run(V3Cfg.quick_cfg() if a.quick else V3Cfg())
