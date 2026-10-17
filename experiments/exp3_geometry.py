"""Exp 3: zero-shot geometry. Everything calibrated / trained / frozen on Manhattan (A); test on A, B (curvy), C (via).

Verifier: scribe-calibrated per layer, config frozen on A val. Die-to-die rules and the CNN threshold: Manhattan val.
CNN: trained on Manhattan train only. F1 drop = F1(A) - F1(B or C) in points, with an independent-bootstrap CI.
Stated target (hypothesis): verifier topology drop <= 5 points, vs the Manhattan-trained CNN.
Usage: python -m experiments.exp3_geometry [--quick]
"""
from __future__ import annotations

import argparse

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.common import ExpCfg, bootstrap_diff, bootstrap_stat, f1_score, write_results  # noqa: E402
from experiments.harness import Context, killer_flags, not_flagged  # noqa: E402
from experiments.select_config import results_root, with_dose  # noqa: E402
from nanovate.testbed.dataset import KINDS  # noqa: E402

METHODS = ("verifier", "d2d_single", "d2d_double", "cnn")
FAMILIES = (("manhattan", "A"), ("curvy", "B"), ("via", "C"))


def run(cfg: ExpCfg, ctx: Context | None = None) -> pd.DataFrame:
    ctx = ctx or Context(cfg)
    dose = cfg.nominal_dose
    tcfg = with_dose(ctx.base, dose)
    rows, per_family, samples = [], {}, []
    for fam, label in FAMILIES:
        specs = [dict(family=fam, split="test", layout_id=lay, die=die, kind=kind, tcfg=tcfg, rule_dose=dose)
                 for lay in range(cfg.layouts) for die in range(cfg.dies) for kind in KINDS]
        df = ctx.rows(specs)
        samples.append(df)
        y, flag, supp = df.killer.values, killer_flags(df), not_flagged(df)
        nu = (df.kind == "nuisance").values
        per_family[fam] = (flag, y)
        for m in METHODS:
            f1, lo, hi = bootstrap_stat([flag[m], y], f1_score, cfg.n_boot, cfg.seed)
            rows.append(dict(family=fam, set=label, method=m, f1=f1, ci_lo=lo, ci_hi=hi, n=len(y), n_killer=int(y.sum()),
                             recall=float(flag[m][y].mean()), nuisance_suppression=float(supp[m][nu].mean())))
        print(f"  exp3: {fam} done", flush=True)
    tab = pd.DataFrame(rows)
    drops = []
    for m in METHODS:
        fa, ya = per_family["manhattan"]
        for fam in ("curvy", "via"):
            fb, yb = per_family[fam]
            d, lo, hi = bootstrap_diff([fa[m], ya], [fb[m], yb], f1_score, cfg.n_boot, cfg.seed)
            drops.append(dict(method=m, test_family=fam, drop_points=100 * d, ci_lo_points=100 * lo, ci_hi_points=100 * hi,
                              n_A=len(ya), n_other=len(yb), target_met=(100 * d <= 5.0) if m == "verifier" else None))
    out = results_root(cfg) / "exp3"
    write_results(out, "geometry_f1", tab, {"exp_cfg": cfg.to_dict(), "pipeline_cfg": ctx.pc.to_dict(), "dose": dose, "data": "synthetic",
                                           "target": "verifier F1 drop A->B/C <= 5 points"})
    pd.DataFrame(drops).to_csv(out / "f1_drop.csv", index=False)
    pd.concat(samples).to_csv(out / "samples.csv", index=False)
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    w = 0.2
    for i, (m, c) in enumerate(zip(METHODS, ("C0", "C1", "C3", "C2"))):
        d = tab[tab.method == m]
        ax.bar([k + (i - 1.5) * w for k in range(3)], d.f1, w, yerr=[d.f1 - d.ci_lo, d.ci_hi - d.f1], capsize=2, label=m, color=c)
    ax.set_xticks(range(3)); ax.set_xticklabels(["A manhattan", "B curvy", "C via"]); ax.set_ylabel("killer F1 (95% CI)")
    ax.set_title(f"Exp 3 - zero-shot geometry, dose {dose:g} (synthetic data)"); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(out / "geometry_f1.png", dpi=120); plt.close(fig)
    return tab


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); a = ap.parse_args()
    run(ExpCfg.quick() if a.quick else ExpCfg.full())
