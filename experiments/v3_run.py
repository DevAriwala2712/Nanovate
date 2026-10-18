"""v3: Experiments 1-6 ONCE on the fresh test seed with the frozen v3 rule, plus paired variant tables. Synthetic data.

Primary = frozen v3 rule (results/v3/calib/decisions.json). Secondary = pre-declared variant with the per-neck cut
margin off (results/v3/calib/secondary.json, committed before this run). "v2 rule" = v2's Delta-beta ladder and
thresholds on the same fresh test patches (paired). Nothing is selected here.

Writes results/v3/exp1..exp6 (the standard experiment outputs, verifier = primary), results/v3/all_rows.csv,
results/v3/variants.csv, results/v3/phase2_targets.csv, results/v3/detection_limit.csv, results/v3/run_config.json,
and rewrites the README "Results (v3)" block (full run only). Usage: python -m experiments.v3_run [--quick]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd

from experiments import exp1_nuisance, exp2_dose, exp3_geometry, exp4_misalignment, exp5_nearmiss, exp6_latency, report
from experiments.common import ExpCfg, bootstrap_ci, bootstrap_stat, f1_score
from experiments.select_config import results_root
from experiments.v3_harness import V3Context

EXPERIMENTS = {"exp1": exp1_nuisance, "exp2": exp2_dose, "exp3": exp3_geometry, "exp4": exp4_misalignment, "exp5": exp5_nearmiss, "exp6": exp6_latency}
RULES = {"primary": "verdict", "secondary": "verdict_sec", "v2 rule": "verdict_v2rule"}
RULE_NAMES = {"primary": "v3 primary (frozen)", "secondary": "v3 secondary (cut margin off, pre-declared)", "v2 rule": "v2 rule, same patches"}
TARGETS = {"noise_free_f1": 0.92, "exp1_suppression": 0.90, "exp1_bridge_recall": 0.90, "manhattan_f1": 0.88}
V3_HEADER = "## Results (v3: per-gap merge rule, fresh test seed)"


def _killer(v: pd.Series) -> np.ndarray:
    return v.astype(str).str.startswith("KILLER").values


def variants(log: pd.DataFrame, cfg: ExpCfg) -> pd.DataFrame:
    """Key metrics per rule on the same fresh test patches (paired). Exp 4 is reported for the primary only."""
    out = []
    nb, sd = cfg.n_boot, cfg.seed
    t = log[(log.split == "test") & (log.tag.fillna("") == "") & (log.die < cfg.dies) & (log.layout < cfg.layouts)]
    t = t.drop_duplicates(["family", "layout", "die", "kind", "dose", "ler_sigma", "grain", "noise_free"])
    nominal = (~t.noise_free) & (t.dose == cfg.nominal_dose) & (t.ler_sigma == 1.0) & (t.grain == 0.15)

    def add(metric, rule, m, lo, hi, n, cell=""):
        out.append(dict(metric=metric, rule=rule, mean=m, ci_lo=lo, ci_hi=hi, n=int(n), cell=cell))

    for rule, col in RULES.items():
        e1 = t[(t.family == "manhattan") & (~t.noise_free) & (t.dose == cfg.nominal_dose) & (t.ler_sigma <= 1.5) & t.grain.isin(cfg.exp1_grains)
               & t.ler_sigma.isin(cfg.exp1_sigmas)]
        cells = []
        for (sg, gr), g in e1.groupby(["ler_sigma", "grain"]):
            nu, br = g[g.kind == "nuisance"], g[g.kind == "bridge"]
            if len(nu) and len(br):
                cells.append((sg, gr, (nu[col] == "NUISANCE").values.astype(float), _killer(br[col]).astype(float)))
        if cells:
            ws = min(cells, key=lambda c: c[2].mean()); wb = min(cells, key=lambda c: c[3].mean())
            add("exp1 worst-cell suppression (LER <= 1.5 nm)", rule, *bootstrap_ci(ws[2], nb, sd), len(ws[2]), f"sigma {ws[0]:g} nm, grain {ws[1]:.0%}")
            add("exp1 worst-cell bridge recall (LER <= 1.5 nm)", rule, *bootstrap_ci(wb[3], nb, sd), len(wb[3]), f"sigma {wb[0]:g} nm, grain {wb[1]:.0%}")
        for fam in ("manhattan", "curvy", "via"):
            g = t[nominal & (t.family == fam)]
            if len(g):
                add(f"killer F1, {fam}, dose {cfg.nominal_dose:g}", rule, *bootstrap_stat([_killer(g[col]), g.killer.values], f1_score, nb, sd), len(g))
        nf = t[t.noise_free & (t.family == "manhattan")]
        if len(nf):
            add(f"killer F1, manhattan, noise-free (dose {cfg.nominal_dose:g})", rule, *bootstrap_stat([_killer(nf[col]), nf.killer.values], f1_score, nb, sd), len(nf))
        m5 = t[nominal & (t.family == "manhattan")]
        for k in ("bridge", "open", "particle"):
            g = m5[m5.kind == k]
            add(f"recall {k}, manhattan, dose {cfg.nominal_dose:g}", rule, *bootstrap_ci(_killer(g[col]).astype(float), nb, sd), len(g))
        for k in ("notch", "filament"):
            g = m5[m5.kind == k]
            add(f"{k} called killer, manhattan, dose {cfg.nominal_dose:g}", rule, *bootstrap_ci(_killer(g[col]).astype(float), nb, sd), len(g))
    return pd.DataFrame(out)


def targets(var: pd.DataFrame) -> pd.DataFrame:
    p = var[var.rule == "primary"].set_index("metric")
    nf = [m for m in p.index if m.startswith("killer F1, manhattan, noise-free")]
    s1 = [m for m in p.index if m.startswith("exp1 worst-cell suppression")]
    b1 = [m for m in p.index if m.startswith("exp1 worst-cell bridge recall")]
    mf = [m for m in p.index if m.startswith("killer F1, manhattan, dose")]
    rows = []
    for name, key, thr in (("Noise-free F1 >= 0.92", nf, TARGETS["noise_free_f1"]), ("Exp 1 suppression >= 90% (worst cell, LER <= 1.5 nm)", s1, TARGETS["exp1_suppression"]),
                           ("Exp 1 bridge recall >= 90% (worst cell, LER <= 1.5 nm)", b1, TARGETS["exp1_bridge_recall"]), ("Manhattan F1 >= 0.88", mf, TARGETS["manhattan_f1"])):
        if key:
            r = p.loc[key[0]]
            rows.append(dict(target=name, metric=key[0], cell=r.get("cell", ""), threshold=thr, mean=r["mean"], ci_lo=r.ci_lo, ci_hi=r.ci_hi, n=int(r.n),
                             met=bool(r["mean"] >= thr)))
    return pd.DataFrame(rows)


def detection_limit(log: pd.DataFrame, cfg: ExpCfg) -> pd.DataFrame:
    t = log[(log.split == "test") & (log.family == "manhattan") & log.noise_free].drop_duplicates(["layout", "die", "kind"])
    out = []
    for k, col, edges in (("open", "p_length", (3, 4, 5, 6, 7, 8.01)), ("notch", "p_remaining", (1.5, 2.5, 3.5, 4.5))):
        for a, b in zip(edges[:-1], edges[1:]):
            g = t[(t.kind == k) & (t[col] >= a) & (t[col] < b)] if col in t else t.iloc[:0]
            for rule, c in RULES.items():
                x = _killer(g[c]) if k == "open" else ~_killer(g[c])
                m, lo, hi = bootstrap_ci(x.astype(float), cfg.n_boot, cfg.seed)
                out.append(dict(kind=k, size=col[2:], lo_nm=a, hi_nm=b, rule=rule, metric="recall" if k == "open" else "not_called_killer",
                                mean=m, ci_lo=lo, ci_hi=hi, n=len(g)))
    return pd.DataFrame(out)


def open_limit(lim: pd.DataFrame, rule: str) -> float | None:
    o = lim[(lim.kind == "open") & (lim.rule == rule) & (lim.n > 0)].sort_values("lo_nm")
    for i in range(len(o)):
        if (o.iloc[i:]["mean"] >= 0.9).all():
            return float(o.iloc[i].lo_nm)
    return None


def _ci(m, lo, hi, n, metric, cell=""):
    if pd.isna(m):
        return f"n/a (n={int(n)})"
    pct = not metric.startswith("killer F1")
    s = f"{100 * m:.0f}% [{100 * lo:.0f}, {100 * hi:.0f}]" if pct else f"{m:.2f} [{lo:.2f}, {hi:.2f}]"
    return f"{s} (n={int(n)})" + (f", {cell}" if isinstance(cell, str) and cell else "")


def render_v3(root: Path) -> str:
    base = report.render_tables(root)
    var, tg, lim = (pd.read_csv(root / f) for f in ("variants.csv", "phase2_targets.csv", "detection_limit.csv"))
    rc = json.loads((root / "run_config.json").read_text())
    parts = [base]
    rows = [[r.target, _ci(r.mean, r.ci_lo, r.ci_hi, r.n, r.metric, r.cell if isinstance(r.cell, str) else ""), "yes" if r.met else "**NO**"] for r in tg.itertuples()]
    parts.append("#### Phase 2 pre-registered targets (hypotheses; verifier = v3 primary)\n\n" + _md(["Target", "Measured", "Met"], rows))
    metrics = list(dict.fromkeys(var.metric))
    def cellv(m, k):
        r = var[(var.metric == m) & (var.rule == k)]
        if not len(r):
            return "n/a"
        r = r.iloc[0]
        return _ci(r["mean"], r.ci_lo, r.ci_hi, r.n, m, r.cell if "cell" in r else "")
    rows = [[m] + [cellv(m, k) for k in RULES] for m in metrics]
    parts.append("#### Rule variants on the same fresh test patches (paired)\n\n"
                 "The secondary variant was declared before this run (`results/v3/calib/secondary.json`). \"v2 rule\" is v2's Delta-beta ladder "
                 "and thresholds re-run on these patches, so it is comparable with v3 here (the v2 table below used a different test seed).\n\n"
                 + _md(["Metric", *[RULE_NAMES[k] for k in RULES]], rows))
    rows = []
    for (k, a, b), g in lim.groupby(["kind", "lo_nm", "hi_nm"], sort=False):
        rows.append([k, f"{g['size'].iloc[0]} [{a:g}, {b:g}) nm", g.metric.iloc[0].replace("_", " ")]
                    + [_ci(*g[g.rule == r][["mean", "ci_lo", "ci_hi", "n"]].iloc[0], "x") for r in RULES])
    lims = "; ".join(f"{RULE_NAMES[r]}: " + (f"{open_limit(lim, r):g} nm" if open_limit(lim, r) is not None else "none in 3-8 nm") for r in RULES)
    parts.append(f"#### Cut / neck detection limit (Manhattan test, noise-free, dose {rc['nominal_dose']:g}; small n per bin)\n\n"
                 + _md(["Kind", "Size bin", "Metric", *[RULE_NAMES[r] for r in RULES]], rows)
                 + "\n\nShortest cut length from which every bin has open recall >= 90% (definition fixed before the run): " + lims + ".")
    return "\n\n".join(parts)


def _md(h, r):
    return report._md(h, r)


def update_readme(path: str, body: str) -> None:
    text = Path(path).read_text()
    if report.V3_START not in text:
        anchor = "## Results (v2: post-hoc fixes)"
        assert anchor in text
        text = text.replace(anchor, f"{V3_HEADER}\n\n{report.V3_START}\n{report.V3_END}\n\n{anchor}", 1)
        Path(path).write_text(text)
    report.update_block(path, body, (report.V3_START, report.V3_END))


def run(quick: bool = False, only: str = "") -> Path:
    cfg = ExpCfg.quick("v3") if quick else ExpCfg.full("v3")
    root = results_root(cfg)
    root.mkdir(parents=True, exist_ok=True)
    ctx = V3Context(cfg)
    (root / "run_config.json").write_text(json.dumps({"exp_cfg": cfg.to_dict(), "test_seed": ctx.test_seed, "val_seed": cfg.seed,
                                                      "nominal_dose": cfg.nominal_dose, "pipeline_cfg": ctx.pc.to_dict(),
                                                      "event_rule_primary": ctx.rc.to_dict(), "event_rule_secondary": ctx.rc_sec.to_dict(),
                                                      "pipeline_cfg_v2": ctx.pc_v2.to_dict(), "data": "synthetic"}, indent=2, default=str))
    t0 = time.time()
    for name, mod in EXPERIMENTS.items():
        if only and name not in only.split(","):
            continue
        t1 = time.time()
        print(f"== {name} ==", flush=True)
        mod.run(cfg, ctx)
        print(f"   {name} done in {time.time() - t1:.0f}s", flush=True)
    log = pd.DataFrame(ctx.log)
    log.to_csv(root / "all_rows.csv", index=False)
    var = variants(log, cfg)
    var.to_csv(root / "variants.csv", index=False)
    targets(var).to_csv(root / "phase2_targets.csv", index=False)
    detection_limit(log, cfg).to_csv(root / "detection_limit.csv", index=False)
    if not quick:
        update_readme("README.md", render_v3(root))
        print("README Results (v3) block updated from", root)
    else:
        (root / "tables.md").write_text(render_v3(root))
        print("quick run: README untouched; preview ->", root / "tables.md")
    print(f"all done in {time.time() - t0:.0f}s")
    return root


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); ap.add_argument("--only", default="")
    a = ap.parse_args()
    run(a.quick, a.only)
