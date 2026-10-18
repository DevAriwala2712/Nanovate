"""v3 STOP POINT 3: frozen v3 rule vs frozen v2 on Manhattan VAL only (no test data). Synthetic data.

Two kinds of val evidence, labelled as such:
  * in-selection: the 2,592 selection patches (val layouts 0..7), from results/v3/calib/val_records.csv. Optimistic
    for v3, because the v3 rule was chosen on them (v2 was chosen on a subset of them too).
  * held-out val: val layouts 8..23 x 2 dies x 6 kinds at the nominal nuisance, dose 50, noisy and noise-free
    (not used by either selection), plus the cut / neck detection limit on noise-free val layouts 0..23.
Every number has n and a 95% percentile-bootstrap CI; v3 - v2 F1 differences are paired bootstraps.
Writes results/v3/val_check/*.csv and the generated block of docs/V3_PHASE2_VAL.md.
Usage: python -m experiments.v3_val_report [--quick]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from experiments.common import bootstrap_ci, bootstrap_stat, f1_score
from experiments.report import _md, update_block
from experiments.v3_select import V3Cfg, collect, load_frozen, root, v2_pc
from nanovate.modules.events import decide
from nanovate.modules.topology import NUISANCE, classify_verdict
from nanovate.testbed.dataset import KINDS

KILLERS = ("bridge", "open", "particle")
DOC = Path("docs/V3_PHASE2_VAL.md")
MARK = ("<!-- V3VAL:START -->", "<!-- V3VAL:END -->")


def v2_verdicts(df: pd.DataFrame, pc2) -> np.ndarray:
    return np.array([classify_verdict((int(r.v2_d_b0), int(r.v2_d_b1)), r.bottleneck, r.energy, pc2.d_star, pc2.e_star, r.peak_z, pc2.z_star)
                     for r in df.itertuples()])


def v3_from_flags(df: pd.DataFrame, pc3) -> np.ndarray:
    nm = (df.bottleneck >= pc3.d_star) & (df.energy >= pc3.e_star) & (df.peak_z >= pc3.z_star)
    return np.where(df.v3_bridge_flag, "KILLER_BRIDGE", np.where(df.v3_killer_flag, "KILLER_OPEN/PARTICLE", np.where(nm, "NEAR_MISS", NUISANCE)))


def _f1ci(flag, y, cfg):
    return bootstrap_stat([flag, y], f1_score, cfg.n_boot, cfg.seed)


def _diff(a, b, y, cfg):
    return bootstrap_stat([a, b, y], lambda p, q, t: f1_score(p, t) - f1_score(q, t), cfg.n_boot, cfg.seed)


def f1_rows(df: pd.DataFrame, v2: np.ndarray, v3: np.ndarray, cfg: V3Cfg, what: str) -> dict:
    y = df.killer.values
    k2, k3 = np.char.startswith(v2.astype(str), "KILLER"), np.char.startswith(v3.astype(str), "KILLER")
    a, b = _f1ci(k2, y, cfg), _f1ci(k3, y, cfg)
    d = _diff(k3, k2, y, cfg)
    row = dict(what=what, n=len(y), n_killer=int(y.sum()), f1_v2=a[0], f1_v2_lo=a[1], f1_v2_hi=a[2],
               f1_v3=b[0], f1_v3_lo=b[1], f1_v3_hi=b[2], diff=d[0], diff_lo=d[1], diff_hi=d[2])
    for k in KINDS:
        sel = (df.kind == k).values
        for name, v, kf in (("v2", v2, k2), ("v3", v3, k3)):
            x = kf[sel] if k in KILLERS else ((v[sel] == NUISANCE) if k == "nuisance" else kf[sel])
            row[f"{name}_{k}"] = float(np.mean(x)) if sel.any() else np.nan
        row[f"n_{k}"] = int(sel.sum())
    return row


def cell_rows(df: pd.DataFrame, v2, v3, cfg: V3Cfg) -> pd.DataFrame:
    out = []
    for (dose, sg, gr), g in df.groupby(["dose", "sigma", "grain"]):
        idx = g.index.values
        for name, v in (("v2", v2), ("v3", v3)):
            nu = (g.kind == "nuisance").values
            br = (g.kind == "bridge").values
            s = bootstrap_ci((v[idx][nu] == NUISANCE).astype(float), cfg.n_boot, cfg.seed)
            r = bootstrap_ci(np.char.startswith(v[idx][br].astype(str), "KILLER").astype(float), cfg.n_boot, cfg.seed)
            nm = g.kind.isin(["notch", "filament"]).values
            out.append(dict(dose=dose, sigma=sg, grain=gr, rule=name, suppression=s[0], supp_lo=s[1], supp_hi=s[2], n_nuisance=int(nu.sum()),
                            bridge_recall=r[0], br_lo=r[1], br_hi=r[2], n_bridge=int(br.sum()),
                            nearmiss_recall=float((v[idx][nm] == "NEAR_MISS").mean()), n_nearmiss=int(nm.sum())))
    return pd.DataFrame(out)


def heldout(cfg: V3Cfg, pc2, pc3, rc, noise_free: bool, layouts) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    specs = [dict(seed=cfg.seed, pc=pc2.to_dict(), dose=50.0, sigma=cfg.nominal[0], grain=cfg.nominal[1], layout=lay, die=die, kind=k,
                  noise_free=noise_free) for lay in layouts for die in range(2) for k in KINDS]
    res = collect(cfg, specs)
    meta = pd.DataFrame([m for m, _, _ in res])
    v2, v3 = [], []
    for _, ev, sc in res:
        v2.append(classify_verdict(sc["delta_betti_v2"], sc["bottleneck"], sc["energy"], pc2.d_star, pc2.e_star, sc["peak_z"], pc2.z_star))
        v3.append(decide(ev, rc, pc3.tau, sc["bottleneck"], sc["energy"], sc["peak_z"], pc3.d_star, pc3.e_star, pc3.z_star)[0])
    return meta, np.array(v2), np.array(v3)


def limit_rows(meta: pd.DataFrame, v3: np.ndarray, cfg: V3Cfg) -> pd.DataFrame:
    kf = np.char.startswith(v3.astype(str), "KILLER")
    out = []
    for k, col, edges in (("open", "p_length", (3, 4, 5, 6, 7, 8.01)), ("notch", "p_remaining", (1.5, 2.5, 3.5, 4.5))):
        for a, b in zip(edges[:-1], edges[1:]):
            sel = ((meta.kind == k) & (meta[col] >= a) & (meta[col] < b)).values
            x = kf[sel] if k == "open" else ~kf[sel]
            m, lo, hi = bootstrap_ci(x.astype(float), cfg.n_boot, cfg.seed)
            out.append(dict(kind=k, size=col[2:], lo_nm=a, hi_nm=b, metric="recall" if k == "open" else "not_called_killer",
                            mean=m, ci_lo=lo, ci_hi=hi, n=int(sel.sum())))
    t = pd.DataFrame(out)
    # pre-registered definition: shortest cut length from which every bin has recall >= 90%
    o = t[(t.kind == "open") & (t.n > 0)].sort_values("lo_nm")
    lim = None
    for i in range(len(o)):
        if (o.iloc[i:]["mean"] >= 0.9).all():
            lim = float(o.iloc[i].lo_nm)
            break
    t.attrs["open_limit_nm"] = lim
    return t


def run(cfg: V3Cfg) -> None:
    out = root(cfg) / "val_check"
    out.mkdir(parents=True, exist_ok=True)
    pc2 = v2_pc()
    pc3, rc = load_frozen(cfg)
    rec = pd.read_csv(root(cfg) / "calib" / "val_records.csv")
    v2s, v3s = v2_verdicts(rec, pc2), v3_from_flags(rec, pc3)
    rows = []
    nom = rec[(rec.sigma == cfg.nominal[0]) & (rec.grain == cfg.nominal[1])]
    for dose, g in nom.groupby("dose"):
        rows.append(f1_rows(g, v2s[g.index], v3s[g.index], cfg, f"in-selection: nominal cell, dose {dose:g}"))
    rows.append(f1_rows(rec, v2s, v3s, cfg, "in-selection: all cells and doses pooled"))
    cells = cell_rows(rec, v2s, v3s, cfg)
    lay_ho = range(cfg.val_layouts, cfg.val_layouts + (2 if cfg.quick else 16))
    for nf in (False, True):
        m, a, b = heldout(cfg, pc2, pc3, rc, nf, lay_ho)
        rows.append(f1_rows(m, a, b, cfg, f"held-out val layouts {lay_ho.start}-{lay_ho.stop - 1}, dose 50, {'noise-free' if nf else 'noisy'}"))
    mlim, _, v3lim = heldout(cfg, pc2, pc3, rc, True, range(0, 2 if cfg.quick else 24))
    lim = limit_rows(mlim, v3lim, cfg)
    f1t = pd.DataFrame(rows)
    f1t.to_csv(out / "f1.csv", index=False); cells.to_csv(out / "cells.csv", index=False); lim.to_csv(out / "detection_limit.csv", index=False)
    (out / "config.json").write_text(json.dumps({"v3_cfg": cfg.to_dict(), "pipeline_cfg_v3": pc3.to_dict(), "event_rule_cfg": rc.to_dict(),
                                                 "pipeline_cfg_v2": pc2.to_dict(), "data": "synthetic", "split": "val",
                                                 "open_limit_nm": lim.attrs["open_limit_nm"]}, indent=2, default=str))
    if not cfg.quick:
        update_block(str(DOC), render(f1t, cells, lim, pc3, rc), MARK)


def _ci(m, lo, hi, n=None, pct=False):
    s = f"{100 * m:.0f}% [{100 * lo:.0f}, {100 * hi:.0f}]" if pct else f"{m:.2f} [{lo:.2f}, {hi:.2f}]"
    return s + (f" (n={int(n)})" if n is not None else "")


def render(f1t, cells, lim, pc3, rc) -> str:
    parts = ["_Generated by `experiments/v3_val_report.py` from `results/v3/`. **Synthetic data, Manhattan val only; no test data.**_",
             f"_Frozen v3 rule: m_merge {rc.m_merge}, normalise {rc.normalise} (m_norm {rc.m_norm}), m_cut {rc.m_cut}, e_edge {rc.e_edge}, "
             f"pair_dist {rc.pair_dist} px, body_radius {rc.body_radius} px; tau {pc3.tau}; near-miss d* {pc3.d_star}, e* {pc3.e_star}, z* {pc3.z_star}._"]
    rows = [[r.what, _ci(r.f1_v2, r.f1_v2_lo, r.f1_v2_hi, r.n), _ci(r.f1_v3, r.f1_v3_lo, r.f1_v3_hi, r.n), f"{r.diff:+.3f} [{r.diff_lo:+.3f}, {r.diff_hi:+.3f}]"]
            for r in f1t.itertuples()]
    parts.append("#### Killer F1, v2 vs v3 (paired bootstrap of the difference)\n\n" + _md(["Val set", "v2", "v3", "v3 - v2"], rows))
    rows = [[r.what] + [f"{100 * getattr(r, f'v2_{k}'):.0f}% -> {100 * getattr(r, f'v3_{k}'):.0f}% (n={getattr(r, f'n_{k}')})" for k in KINDS] for r in f1t.itertuples()]
    parts.append("#### Per kind, v2 -> v3 (killers: recall; nuisance: NUISANCE verdict; notch / filament: called killer)\n\n"
                 + _md(["Val set", *KINDS], rows))
    c = cells[cells.dose == 50.0]
    rows = []
    for (sg, gr), g in c.groupby(["sigma", "grain"]):
        a, b = g[g.rule == "v2"].iloc[0], g[g.rule == "v3"].iloc[0]
        rows.append([f"{sg:g} nm", f"{gr:.0%}", _ci(a.suppression, a.supp_lo, a.supp_hi, a.n_nuisance, True), _ci(b.suppression, b.supp_lo, b.supp_hi, b.n_nuisance, True),
                     _ci(a.bridge_recall, a.br_lo, a.br_hi, a.n_bridge, True), _ci(b.bridge_recall, b.br_lo, b.br_hi, b.n_bridge, True)])
    parts.append("#### Exp 1 analogue on val (dose 50, in-selection cells; n = 16 per cell, so CIs are wide)\n\n"
                 + _md(["LER sigma", "Grain", "Suppression v2", "Suppression v3", "Bridge recall v2", "Bridge recall v3"], rows))
    rows = [[r.kind, r.size, f"[{r.lo_nm:g}, {r.hi_nm:g})", r.metric.replace("_", " "), _ci(r.mean, r.ci_lo, r.ci_hi, r.n, True)] for r in lim.itertuples()]
    L = lim.attrs.get("open_limit_nm")
    parts.append("#### Cut / neck detection limit (v3, noise-free val layouts 0-23, dose 50)\n\n" + _md(["Kind", "Size", "Bin (nm)", "Metric", "Rate"], rows)
                 + f"\n\nPre-registered definition: the shortest cut length from which every bin has recall >= 90%: "
                 + (f"**{L:g} nm**." if L is not None else "**none in the tested range** (3-8 nm)."))
    return "\n\n".join(parts)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); a = ap.parse_args()
    run(V3Cfg.quick_cfg() if a.quick else V3Cfg())
