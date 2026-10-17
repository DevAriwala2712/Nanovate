"""Exp 1: nuisance stress. LER sigma x grain; nuisance suppression % and bridge recall (synthetic data).

Manhattan *test* layouts at the nominal dose. Suppression = fraction of nuisance-only samples that are not
flagged (verifier: verdict NUISANCE; NEAR_MISS counts as a false alarm). Secondary: not_killer = fraction of
nuisance samples not flagged as a KILLER (NEAR_MISS allowed). Bridge recall = fraction of bridge
samples flagged as a killer. Stated target (hypothesis): >= 90% / >= 90% at sigma <= 1.5 nm, vs die-to-die.
Baseline rules were fitted on Manhattan val at the nominal nuisance (sigma 1.0, grain 15%) and are NOT
re-fitted per cell. Usage: python -m experiments.exp1_nuisance [--quick]
"""
from __future__ import annotations

import argparse

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.common import ExpCfg, bootstrap_ci, write_results  # noqa: E402
from experiments.harness import Context, killer_flags, not_flagged, nuisance_cfg  # noqa: E402
from experiments.select_config import results_root, with_dose  # noqa: E402

METHODS = ("verifier", "d2d_single", "d2d_double", "cnn")


def run(cfg: ExpCfg, ctx: Context | None = None) -> pd.DataFrame:
    ctx = ctx or Context(cfg)
    dose = cfg.nominal_dose
    rows, samples = [], []
    for sig in cfg.exp1_sigmas:
        for gr in cfg.exp1_grains:
            tcfg = with_dose(nuisance_cfg(ctx.base, sig, gr), dose)
            specs = [dict(family="manhattan", split="test", layout_id=lay, die=die, kind=kind, tcfg=tcfg, rule_dose=dose)
                     for lay in range(cfg.layouts) for die in range(cfg.dies) for kind in ("nuisance", "bridge")]
            df = ctx.rows(specs)
            samples.append(df)
            supp, flag = not_flagged(df), killer_flags(df)
            nu, br = (df.kind == "nuisance").values, (df.kind == "bridge").values
            for m in METHODS:
                for metric, vec in (("suppression", supp[m][nu]), ("not_killer", ~flag[m][nu]), ("bridge_recall", flag[m][br])):
                    mean, lo, hi = bootstrap_ci(vec.astype(float), cfg.n_boot, cfg.seed)
                    rows.append(dict(ler_sigma=sig, grain=gr, method=m, metric=metric, mean=mean, ci_lo=lo, ci_hi=hi, n=len(vec),
                                     mean_resamples=float(df.resamples[br if metric == "bridge_recall" else nu].mean())))
                if m == "verifier":
                    strict = (df.verdict[br] == "KILLER_BRIDGE").values.astype(float)
                    mean, lo, hi = bootstrap_ci(strict, cfg.n_boot, cfg.seed)
                    rows.append(dict(ler_sigma=sig, grain=gr, method="verifier", metric="bridge_recall_strict_class", mean=mean,
                                     ci_lo=lo, ci_hi=hi, n=len(strict), mean_resamples=float(df.resamples[br].mean())))
        print(f"  exp1: sigma {sig} done", flush=True)
    tab = pd.DataFrame(rows)
    ok = tab[(tab.method == "verifier") & (tab.ler_sigma <= 1.5) & tab.metric.isin(["suppression", "bridge_recall"])]
    tab["target_met"] = np.where((tab.method == "verifier") & (tab.ler_sigma <= 1.5) & tab.metric.isin(["suppression", "bridge_recall"]), tab["mean"] >= 0.9, np.nan)
    out = results_root(cfg) / "exp1"
    write_results(out, "nuisance_stress", tab, {"exp_cfg": cfg.to_dict(), "pipeline_cfg": ctx.pc.to_dict(), "dose": dose,
                                                "data": "synthetic", "target": ">=90% suppression and >=90% bridge recall at sigma<=1.5 nm"})
    pd.concat(samples).to_csv(out / "samples.csv", index=False)
    _plot(tab, cfg, out / "nuisance_stress.png", ok)
    return tab


def _plot(tab, cfg, path, _ok):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for ax, metric, title in zip(axes, ("suppression", "bridge_recall"), ("Nuisance suppression", "Bridge recall")):
        g0 = min(tab.grain)
        for m, col in zip(METHODS, ("C0", "C1", "C3", "C2")):
            d = tab[(tab.method == m) & (tab.metric == metric) & (tab.grain == g0)].sort_values("ler_sigma")
            ax.errorbar(d.ler_sigma, d["mean"], yerr=[d["mean"] - d.ci_lo, d.ci_hi - d["mean"]], marker="o", capsize=3, label=m, color=col)
        ax.axhline(0.9, ls=":", c="gray"); ax.set_xlabel("LER sigma (nm)"); ax.set_title(f"{title} (grain {g0:.0%}, n per point in CSV)")
    axes[0].set_ylabel("rate (95% bootstrap CI)"); axes[0].legend(fontsize=8)
    fig.suptitle("Exp 1 - nuisance stress (synthetic data)"); fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); a = ap.parse_args()
    run(ExpCfg.quick() if a.quick else ExpCfg.full())
