"""Exp 4: misalignment. The same wafer re-acquired with shifts U(-8, 8) px plus a smooth warp (<= 1 px).

% of verdicts unchanged vs the zero-shift acquisition of the same wafer, with and without Module 3
(alignment). Reference acquisition: shift (0, 0), no warp. Kinds: nuisance, bridge, open, particle.
Stated target (hypothesis): >= 95% unchanged with Module 3. Usage: python -m experiments.exp4_misalignment [--quick]
"""
from __future__ import annotations

import argparse
import dataclasses

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.common import ExpCfg, bootstrap_ci, write_results  # noqa: E402
from experiments.harness import Context  # noqa: E402
from experiments.select_config import results_root, with_dose  # noqa: E402

KINDS4 = ("nuisance", "bridge", "open", "particle")


def run(cfg: ExpCfg, ctx: Context | None = None) -> pd.DataFrame:
    ctx = ctx or Context(cfg)
    dose = cfg.nominal_dose
    tcfg = with_dose(ctx.base, dose)
    pcs = {"verifier_with_M3": ctx.pc, "verifier_without_M3": dataclasses.replace(ctx.pc, align=False)}
    n_acq = 1 if cfg.quick else 2
    rng = np.random.default_rng([cfg.seed, 4])
    recs = []
    for lay in range(cfg.layouts):
        for die in range(cfg.dies):
            for kind in KINDS4:
                base_kw = dict(family="manhattan", split="test", layout_id=lay, die=die, kind=kind, tcfg=tcfg, methods=("verifier",))
                refs = {n: ctx.evaluate(**base_kw, shift=(0.0, 0.0), acq_tag="ref", pc=pc, tag=n) for n, pc in pcs.items()}
                for j in range(n_acq):
                    sh = tuple(float(v) for v in rng.uniform(-cfg.exp4_shift, cfg.exp4_shift, 2))
                    for n, pc in pcs.items():
                        r = ctx.evaluate(**base_kw, shift=sh, warp=cfg.exp4_warp, acq_tag=f"s{j}", pc=pc, tag=n)
                        recs.append(dict(method=n, layout=lay, die=die, kind=kind, acq=j, shift_y=sh[0], shift_x=sh[1],
                                         ref_verdict=refs[n]["verdict"], verdict=r["verdict"],
                                         unchanged=refs[n]["verdict"] == r["verdict"],
                                         killer_flag_unchanged=refs[n]["verdict"].startswith("KILLER") == r["verdict"].startswith("KILLER")))
        print(f"  exp4: layout {lay} done", flush=True)
    df = pd.DataFrame(recs)
    rows = []
    for m in pcs:
        for kind in (*KINDS4, "all"):
            g = df[(df.method == m) & ((df.kind == kind) | (kind == "all"))]
            for col in ("unchanged", "killer_flag_unchanged"):
                mean, lo, hi = bootstrap_ci(g[col].values.astype(float), cfg.n_boot, cfg.seed)
                rows.append(dict(method=m, kind=kind, metric=col, mean=mean, ci_lo=lo, ci_hi=hi, n=len(g)))
    tab = pd.DataFrame(rows)
    tab["target_met"] = np.where((tab.method == "verifier_with_M3") & (tab.kind == "all") & (tab.metric == "unchanged"), tab["mean"] >= 0.95, np.nan)
    out = results_root(cfg) / "exp4"
    write_results(out, "misalignment", tab, {"exp_cfg": cfg.to_dict(), "pipeline_cfg": ctx.pc.to_dict(), "dose": dose, "data": "synthetic",
                                            "target": ">=95% verdicts unchanged with Module 3"})
    df.to_csv(out / "samples.csv", index=False)
    a = tab[(tab.metric == "unchanged") & (tab.kind != "all")]
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    for i, (m, c) in enumerate(zip(pcs, ("C0", "C3"))):
        d = a[a.method == m]
        ax.bar(np.arange(len(KINDS4)) + (i - 0.5) * 0.35, d["mean"], 0.35, yerr=[d["mean"] - d.ci_lo, d.ci_hi - d["mean"]], capsize=2, label=m, color=c)
    ax.axhline(0.95, ls=":", c="gray"); ax.set_xticks(range(len(KINDS4))); ax.set_xticklabels(KINDS4); ax.set_ylabel("verdict unchanged vs zero shift")
    ax.set_title("Exp 4 - shifts +-8 px + warp (synthetic data)"); ax.legend(fontsize=8); fig.tight_layout()
    fig.savefig(out / "misalignment.png", dpi=120); plt.close(fig)
    return tab


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true"); a = ap.parse_args()
    run(ExpCfg.quick() if a.quick else ExpCfg.full())
