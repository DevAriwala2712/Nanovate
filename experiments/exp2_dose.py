"""Exp 2: dose cliff. F1 vs dose (killer vs not-killer) for the verifier, die-to-die and the CNN (synthetic data).

Manhattan *test*, all six kinds (killers: bridge / open / particle; notch, filament and nuisance are
negatives; 50% killer prevalence). Baseline thresholds are per-dose rules fitted on Manhattan val.
Stated target (hypothesis): verifier F1 >= 0.9 at lambda >= 50 e-/px. Reports the lowest lambda with
F1 >= 0.9 per method, or "none". Usage: python -m experiments.exp2_dose [--quick]
"""
from __future__ import annotations

import argparse

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.common import ExpCfg, bootstrap_stat, f1_score, write_results  # noqa: E402
from experiments.harness import Context, killer_flags, not_flagged  # noqa: E402
from experiments.select_config import results_root, with_dose  # noqa: E402
from nanovate.testbed.dataset import KINDS  # noqa: E402

METHODS = ("verifier", "d2d_single", "d2d_double", "cnn")


def run(cfg: ExpCfg, ctx: Context | None = None) -> pd.DataFrame:
    ctx = ctx or Context(cfg)
    rows, samples = [], []
    for dose in cfg.doses:
        tcfg = with_dose(ctx.base, dose)
        specs = [dict(family="manhattan", split="test", layout_id=lay, die=die, kind=kind, tcfg=tcfg, rule_dose=dose)
                 for lay in range(cfg.layouts) for die in range(cfg.dies) for kind in KINDS]
        df = ctx.rows(specs)
        samples.append(df)
        y, flag, supp = df.killer.values, killer_flags(df), not_flagged(df)
        nu = (df.kind == "nuisance").values
        for m in METHODS:
            f1, lo, hi = bootstrap_stat([flag[m], y], f1_score, cfg.n_boot, cfg.seed)
            rows.append(dict(dose=dose, method=m, f1=f1, ci_lo=lo, ci_hi=hi, n=len(y), n_killer=int(y.sum()),
                             recall=float(flag[m][y].mean()), nuisance_suppression=float(supp[m][nu].mean())))
        print(f"  exp2: dose {dose} done", flush=True)
    tab = pd.DataFrame(rows)
    low = []
    for m in METHODS:
        ok = tab[(tab.method == m) & (tab.f1 >= 0.9)].dose
        low.append(dict(method=m, lowest_dose_f1_ge_0p9=float(ok.min()) if len(ok) else "none"))
    out = results_root(cfg) / "exp2"
    write_results(out, "dose_cliff", tab, {"exp_cfg": cfg.to_dict(), "pipeline_cfg": ctx.pc.to_dict(), "data": "synthetic",
                                           "target": "verifier F1 >= 0.9 at lambda >= 50 e-/px"})
    pd.DataFrame(low).to_csv(out / "lowest_dose.csv", index=False)
    pd.concat(samples).to_csv(out / "samples.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for m, c in zip(METHODS, ("C0", "C1", "C3", "C2")):
        d = tab[tab.method == m].sort_values("dose")
        ax.errorbar(d.dose, d.f1, yerr=[d.f1 - d.ci_lo, d.ci_hi - d.f1], marker="o", capsize=3, label=m, color=c)
    ax.axhline(0.9, ls=":", c="gray"); ax.set_xscale("log"); ax.set_xlabel("dose (e-/px)"); ax.set_ylabel("killer F1 (95% bootstrap CI)")
    ax.set_title("Exp 2 - dose cliff (synthetic data)"); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(out / "dose_cliff.png", dpi=120); plt.close(fig)
    return tab


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); a = ap.parse_args()
    run(ExpCfg.quick() if a.quick else ExpCfg.full())
