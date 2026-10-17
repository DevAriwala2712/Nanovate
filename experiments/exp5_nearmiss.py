"""Exp 5: near-miss capture. Notches / filaments that stop 1-3 nm short, at a fixed false-alarm budget.

Manhattan *test*, nominal dose. Two rules on the same patches:
  delta-beta alone : alarm iff Delta-beta != (0, 0)
  full rule        : alarm iff verdict != NUISANCE (Delta-beta != 0, or NEAR_MISS from the bottleneck + SDF-energy
                     rule with d*, e* frozen on Manhattan val so that NEAR_MISS fires on <= 5% of nuisance samples)
Reports near-miss recall (alarm rate on notch / filament) and the nuisance alarm rate of each rule, plus the
strict NEAR_MISS class rate. Stated target (hypothesis): recall >= 80% at the fixed false-alarm rate, vs delta-beta alone.
Usage: python -m experiments.exp5_nearmiss [--quick]
"""
from __future__ import annotations

import argparse

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.common import ExpCfg, bootstrap_ci, write_results  # noqa: E402
from experiments.harness import Context  # noqa: E402
from experiments.select_config import results_root, with_dose  # noqa: E402

KINDS5 = ("nuisance", "notch", "filament")


def run(cfg: ExpCfg, ctx: Context | None = None) -> pd.DataFrame:
    ctx = ctx or Context(cfg)
    dose = cfg.nominal_dose
    tcfg = with_dose(ctx.base, dose)
    specs = [dict(family="manhattan", split="test", layout_id=lay, die=die, kind=kind, tcfg=tcfg, methods=("verifier",))
             for lay in range(cfg.layouts) for die in range(cfg.exp5_dies) for kind in KINDS5]
    df = ctx.rows(specs, "exp5")
    df["alarm_dbeta"] = ((df.d_b0 != 0) | (df.d_b1 != 0)).values
    df["alarm_full"] = (df.verdict != "NUISANCE").values
    df["near_miss_class"] = (df.verdict == "NEAR_MISS").values
    rows = []
    for kind in KINDS5:
        g = df[df.kind == kind]
        for rule, col in (("delta_beta_alone", "alarm_dbeta"), ("full_rule", "alarm_full"), ("near_miss_class_only", "near_miss_class")):
            mean, lo, hi = bootstrap_ci(g[col].values.astype(float), cfg.n_boot, cfg.seed)
            rows.append(dict(kind=kind, rule=rule, metric="alarm_rate" if kind == "nuisance" else "recall", mean=mean, ci_lo=lo, ci_hi=hi, n=len(g)))
    nm = df[df.kind.isin(["notch", "filament"])]
    for rule, col in (("delta_beta_alone", "alarm_dbeta"), ("full_rule", "alarm_full")):
        mean, lo, hi = bootstrap_ci(nm[col].values.astype(float), cfg.n_boot, cfg.seed)
        rows.append(dict(kind="notch+filament", rule=rule, metric="recall", mean=mean, ci_lo=lo, ci_hi=hi, n=len(nm)))
    tab = pd.DataFrame(rows)
    tab["target_met"] = np.where((tab.kind == "notch+filament") & (tab.rule == "full_rule"), tab["mean"] >= 0.8, np.nan)
    out = results_root(cfg) / "exp5"
    write_results(out, "near_miss", tab, {"exp_cfg": cfg.to_dict(), "pipeline_cfg": ctx.pc.to_dict(), "dose": dose, "data": "synthetic",
                                         "target": "near-miss recall >= 80% at the fixed nuisance false-alarm rate"})
    df.to_csv(out / "samples.csv", index=False)
    if "p_gap" in df:  # recall by filament gap (nm)
        f = df[df.kind == "filament"].assign(gap=lambda d: d.p_gap.round().astype(int))
        by = f.groupby("gap").agg(n=("alarm_full", "size"), recall_full=("alarm_full", "mean"), recall_dbeta=("alarm_dbeta", "mean")).reset_index()
        by.to_csv(out / "filament_recall_by_gap.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    d = tab[(tab.rule != "near_miss_class_only")]
    for i, (r, c) in enumerate(zip(("delta_beta_alone", "full_rule"), ("C1", "C0"))):
        x = d[d.rule == r]
        ax.bar(np.arange(len(x)) + (i - 0.5) * 0.35, x["mean"], 0.35, yerr=[x["mean"] - x.ci_lo, x.ci_hi - x["mean"]], capsize=2, label=r, color=c)
    ax.set_xticks(range(len(x))); ax.set_xticklabels([f"{k}\n({m})" for k, m in zip(x.kind, x.metric)], fontsize=8)
    ax.axhline(0.8, ls=":", c="gray"); ax.set_ylabel("alarm rate (95% CI)"); ax.legend(fontsize=8)
    ax.set_title("Exp 5 - near-miss capture (synthetic data)"); fig.tight_layout(); fig.savefig(out / "near_miss.png", dpi=120); plt.close(fig)
    return tab


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); a = ap.parse_args()
    run(ExpCfg.quick() if a.quick else ExpCfg.full())
