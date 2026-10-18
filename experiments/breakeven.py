"""Label break-even: fine-tuned CNN F1 vs the training-free verifier on a new family (synthetic data).

Reads the manifest + weights written by scripts/breakeven_train.py (Colab), regenerates each family's pool
to pick the threshold on the run's held-out 25% (``labelled`` rule; k = 0 uses 0.39) and, as a second curve,
uses 0.39 for every k (``fixed_0.39``). Evaluates on the same test patches as Exp 3 (dose 50, test split).
The verifier reference is the v2 Exp 3 F1 (``results/v2/exp3/geometry_f1.csv``); the verifier is not rerun.
Break-even k = smallest k whose seed-mean CNN F1 >= the verifier's F1; clearly-better k = smallest k whose
every seed's CI lower bound > the verifier's F1. Writes results/v2/breakeven/{breakeven.csv, summary.csv,
breakeven.png, config.json}. Usage: python -m experiments.breakeven --weights DIR [--quick]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.common import ExpCfg, bootstrap_stat, f1_score, write_results  # noqa: E402
from experiments.select_config import results_root, with_dose  # noqa: E402
from nanovate.baselines.breakeven import DEFAULT_THRESHOLD, pick_threshold_labelled  # noqa: E402
from nanovate.baselines.cnn import load_cnn, predict_proba  # noqa: E402
from nanovate.baselines.data import make_cnn_train_set  # noqa: E402
from nanovate.config import TestbedCfg  # noqa: E402
from nanovate.testbed.dataset import KINDS, make_sample  # noqa: E402

VERIFIER_CSV = "results/v2/exp3/geometry_f1.csv"


def breakeven_k(f1_by_k: dict[int, float], verifier_f1: float) -> int | str:
    """Smallest k with F1 >= ``verifier_f1``, or "none"."""
    ok = [k for k, f in sorted(f1_by_k.items()) if f >= verifier_f1 - 1e-12]
    return ok[0] if ok else "none"


def _test_set(cfg: ExpCfg, family: str):
    tcfg = with_dose(TestbedCfg(), cfg.nominal_dose)
    s = [make_sample(family, "test", lay, die, kind, tcfg, cfg.seed)
         for lay in range(cfg.layouts) for die in range(cfg.dies) for kind in KINDS]
    return np.stack([x.image for x in s]), np.stack([x.cad for x in s]), np.array([x.killer for x in s])


def run(cfg: ExpCfg, weights_dir: str | Path, verifier_csv: str = VERIFIER_CSV) -> pd.DataFrame:
    wdir = Path(weights_dir)
    man = json.loads((wdir / "manifest.json").read_text())
    base, _, _ = load_cnn(man["base_weights"])
    ver = pd.read_csv(verifier_csv)
    rows = []
    for fam in sorted({r["family"] for r in man["runs"]}):
        X, C, y = _test_set(cfg, fam)
        pool = man["pool"]
        pimg, pcad, py, _ = make_cnn_train_set(pool["layouts"], pool["dies"], seed=pool["seed"], family=fam)
        for r in [r for r in man["runs"] if r["family"] == fam]:
            path = wdir / r["weights"] if r["weights"] else None
            missing = path is not None and not path.exists()
            model = base if path is None else (None if missing else load_cnn(path)[0])
            p_test = None if model is None else predict_proba(model, X, C)
            thr_idx = np.asarray(r["thr_idx"], int)
            for rule in ("labelled", "fixed_0.39"):
                row = dict(family=fam, k=r["k"], seed=r["seed"], threshold_rule=rule, n=len(y), missing=missing,
                           threshold=np.nan, f1=np.nan, ci_lo=np.nan, ci_hi=np.nan)
                if model is not None:
                    if rule == "labelled" and len(thr_idx):
                        thr = pick_threshold_labelled(predict_proba(model, pimg[thr_idx], pcad[thr_idx]), py[thr_idx])
                    else:
                        thr = DEFAULT_THRESHOLD
                    f1, lo, hi = bootstrap_stat([p_test >= thr, y], f1_score, cfg.n_boot, cfg.seed)
                    row.update(threshold=thr, f1=f1, ci_lo=lo, ci_hi=hi)
                rows.append(row)
        print(f"  breakeven: {fam} done", flush=True)
    tab = pd.DataFrame(rows)
    summ = []
    for (fam, rule), g in tab[~tab.missing].groupby(["family", "threshold_rule"]):
        vf = float(ver[(ver.method == "verifier") & (ver.family == fam)].f1.iloc[0])
        mean = g.groupby("k").f1.mean().to_dict()
        clear = [k for k, gg in g.groupby("k") if (gg.ci_lo > vf).all()]
        summ.append(dict(family=fam, threshold_rule=rule, verifier_f1=vf, breakeven_k=breakeven_k(mean, vf),
                         clearly_better_k=min(clear) if clear else "none", n_seeds=int(g.seed.nunique()), n_test=int(g.n.iloc[0])))
    summary = pd.DataFrame(summ)
    out = results_root(cfg) / "breakeven"
    write_results(out, "breakeven", tab, {"exp_cfg": cfg.to_dict(), "manifest": {k: v for k, v in man.items() if k != "runs"},
                                          "verifier_csv": verifier_csv, "data": "synthetic",
                                          "note": "only the CNN baseline is adapted (family train split); verifier and test split untouched"})
    summary.to_csv(out / "summary.csv", index=False)
    _plot(tab, ver, out / "breakeven.png")
    return tab


def _plot(tab: pd.DataFrame, ver: pd.DataFrame, path: Path) -> None:
    fams = sorted(tab.family.unique())
    fig, axes = plt.subplots(1, len(fams), figsize=(5.5 * len(fams), 4), squeeze=False)
    for ax, fam in zip(axes[0], fams):
        v = ver[(ver.method == "verifier") & (ver.family == fam)].iloc[0]
        ax.axhspan(v.ci_lo, v.ci_hi, color="C0", alpha=0.15)
        ax.axhline(v.f1, color="C0", label=f"verifier (no labels) {v.f1:.2f}")
        for rule, c in (("labelled", "C1"), ("fixed_0.39", "C2")):
            g = tab[(tab.family == fam) & (tab.threshold_rule == rule) & ~tab.missing].groupby("k").f1
            ks = np.array(sorted(g.groups)); x = np.where(ks == 0, 6, ks)   # k = 0 drawn at the left edge
            m, lo, hi = g.mean()[ks], g.min()[ks], g.max()[ks]
            ax.errorbar(x, m, yerr=[m - lo, hi - m], marker="o", capsize=3, color=c, label=f"CNN fine-tuned, threshold {rule}")
        ax.set_xscale("log"); ax.set_xlabel("labelled patches of the new family (k; leftmost = 0)")
        ax.set_ylabel("killer F1 (test, dose 50)"); ax.set_title(f"{fam}: labels the CNN needs (synthetic data)"); ax.legend(fontsize=7)
    fig.tight_layout(); fig.savefig(path, dpi=120); plt.close(fig)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--weights", required=True); ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    run(ExpCfg.quick() if a.quick else ExpCfg.full(), a.weights)
