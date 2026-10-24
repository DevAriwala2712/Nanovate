"""README figures: one chart per experiment (v3 CSVs), the real-data rungs, and a testbed strip. Synthetic data unless marked.

  python scripts/make_figures.py      # writes docs/assets/figures/<name>-{light,dark}.png

Every plotted number is read from results/ (never typed); the README embeds the light / dark pair with <picture>.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
V3 = ROOT / "results" / "v3"
REAL = ROOT / "results" / "real"
OUT = ROOT / "docs" / "assets" / "figures"

# Fixed method -> slot mapping (colour follows the method in every chart). Light / dark steps of the same hue.
METHODS = {"verifier": "Verifier (ours)", "cnn": "CNN (Manhattan-trained)", "d2d_single": "Die-to-die (single)",
           "d2d_double": "Die-to-die (double)"}
MARKERS = {"verifier": "o", "cnn": "s", "d2d_single": "^", "d2d_double": "D"}
THEMES = {
    "light": dict(surface="#ffffff", ink="#1f2328", ink2="#59636e", grid="#e6e8eb", target="#59636e",
                  series={"verifier": "#2a78d6", "cnn": "#eb6834", "d2d_single": "#1baf7a", "d2d_double": "#eda100"},
                  seq="Blues", accent="#2a78d6", muted="#c9ced4"),
    "dark": dict(surface="#0d1117", ink="#e6edf3", ink2="#9198a1", grid="#262c33", target="#9198a1",
                 series={"verifier": "#3987e5", "cnn": "#d95926", "d2d_single": "#199e70", "d2d_double": "#c98500"},
                 seq="Blues_r", accent="#3987e5", muted="#3d444d"),
}


def _style(t: dict) -> None:
    plt.rcParams.update({
        "figure.facecolor": t["surface"], "axes.facecolor": t["surface"], "savefig.facecolor": t["surface"],
        "text.color": t["ink"], "axes.labelcolor": t["ink2"], "xtick.color": t["ink2"], "ytick.color": t["ink2"],
        "axes.edgecolor": t["grid"], "axes.grid": True, "grid.color": t["grid"], "grid.linewidth": 0.8,
        "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
        "font.family": "DejaVu Sans", "font.size": 10.5, "axes.titlesize": 12, "axes.titleweight": "bold",
        "axes.titlelocation": "left", "axes.titlepad": 12, "legend.frameon": False, "legend.labelcolor": t["ink"],
        "xtick.major.size": 0, "ytick.major.size": 0,
    })


def _save(fig, name: str, mode: str, t: dict, note: str = "synthetic data · 95% bootstrap CI") -> None:
    fig.text(0.995, -0.01, note, ha="right", va="top", fontsize=8, color=t["ink2"])
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f"{name}-{mode}.png", dpi=160, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)


def _target(ax, y: float, t: dict, label: str, left: bool = False) -> None:
    ax.axhline(y, color=t["target"], lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.annotate(label, (0 if left else 1, y), xycoords=("axes fraction", "data"), xytext=(0, 4), textcoords="offset points",
                ha="left" if left else "right", va="bottom", fontsize=8.5, color=t["ink2"])


def fig_exp1(mode: str, t: dict) -> None:
    df = pd.read_csv(V3 / "exp1" / "nuisance_stress.csv")
    fig, axes = plt.subplots(2, 2, figsize=(10, 7.2), constrained_layout=True)
    for row, m in enumerate(("verifier", "d2d_single")):
        for col, (metric, title) in enumerate((("suppression", "nuisance suppression"), ("bridge_recall", "bridge recall"))):
            ax = axes[row, col]
            d = df[(df.method == m) & (df.metric == metric)].pivot(index="grain", columns="ler_sigma", values="mean")
            ax.imshow(d.values, cmap="Blues", vmin=0, vmax=1.15, aspect="auto", origin="lower")
            for i, g in enumerate(d.index):
                for j, s in enumerate(d.columns):
                    v = d.values[i, j]
                    ax.text(j, i, f"{100 * v:.0f}", ha="center", va="center", fontsize=9.5,
                            color="#ffffff" if v > 0.62 else "#1f2328", fontweight="bold" if v >= 0.9 else "normal")
            ax.set_xticks(range(len(d.columns)), [f"{s:g}" for s in d.columns])
            ax.set_yticks(range(len(d.index)), [f"{100 * g:.0f}%" for g in d.index])
            ax.grid(False)
            ax.set_title(f"{METHODS[m]} · {title}", fontsize=11)
            if row == 1:
                ax.set_xlabel("LER σ (nm)")
            if col == 0:
                ax.set_ylabel("grain contrast")
            n = int(df[(df.method == m) & (df.metric == metric)].n.iloc[0])
            ax.text(1, 1.01, f"n={n} per cell", transform=ax.transAxes, ha="right", va="bottom", fontsize=8, color=t["ink2"])
    fig.suptitle("Exp 1 · nuisance stress (% of samples; bold = meets the 90% target)", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=t["ink"])
    _save(fig, "exp1_nuisance", mode, t, "synthetic data · dose 50 e⁻/px")


def fig_exp2(mode: str, t: dict) -> None:
    df = pd.read_csv(V3 / "exp2" / "dose_cliff.csv")
    fig, ax = plt.subplots(figsize=(9, 5))
    for m in METHODS:
        d = df[df.method == m].sort_values("dose")
        c = t["series"][m]
        ax.fill_between(d.dose, d.ci_lo, d.ci_hi, color=c, alpha=0.14 if m in ("verifier", "cnn") else 0.05, lw=0)
        ax.plot(d.dose, d.f1, color=c, lw=2.2 if m == "verifier" else 1.8, marker=MARKERS[m], ms=6.5,
                markeredgecolor=t["surface"], markeredgewidth=1.5, label=METHODS[m], zorder=3)
        ax.annotate({"verifier": "Verifier", "cnn": "CNN", "d2d_single": "D2D single", "d2d_double": "D2D double"}[m],
                    (d.dose.iloc[-1], d.f1.iloc[-1]),
                    xytext=(8, 0), textcoords="offset points", va="center", fontsize=9, color=t["ink"])
    _target(ax, 0.9, t, "target F1 ≥ 0.9 at λ ≥ 50")
    ax.axvspan(50, 330, color=t["muted"], alpha=0.18, lw=0, zorder=0)
    ax.set_xscale("log")
    ax.set_xticks(sorted(df.dose.unique()), [f"{d:g}" for d in sorted(df.dose.unique())])
    ax.minorticks_off()
    ax.set_xlim(4, 330)
    ax.set_ylim(0.45, 1.0)
    ax.set_xlabel("electron dose λ (e⁻/px, log scale)")
    ax.set_ylabel("killer F1")
    ax.set_title(f"Exp 2 · killer F1 vs dose (n={int(df.n.iloc[0])} per point)")
    ax.legend(loc="lower right", ncol=2, fontsize=9)
    _save(fig, "exp2_dose", mode, t)


def _grouped_bars(ax, groups: list[str], series: dict, t: dict, colors: dict, labels: dict) -> None:
    k = len(series)
    w = 0.8 / k
    x = np.arange(len(groups))
    for i, (key, (vals, lo, hi)) in enumerate(series.items()):
        pos = x - 0.4 + w * (i + 0.5)
        ax.bar(pos, vals, w * 0.86, color=colors[key], label=labels[key], zorder=2)
        ax.errorbar(pos, vals, yerr=[np.array(vals) - lo, np.array(hi) - vals], fmt="none", ecolor=t["ink2"],
                    elinewidth=1, capsize=2.5, zorder=3)
        for p, v in zip(pos, vals):
            ax.text(p, 0.02, f"{v:.2f}" if v <= 1.0001 and max(vals) <= 1.0001 else f"{v:.0f}", ha="center",
                    va="bottom", fontsize=7.5, color="#ffffff", rotation=90, zorder=4)
    ax.set_xticks(x, groups)
    ax.grid(axis="x", visible=False)


def fig_exp3(mode: str, t: dict) -> None:
    df = pd.read_csv(V3 / "exp3" / "geometry_f1.csv")
    fams = ["manhattan", "curvy", "via"]
    series = {}
    for m in METHODS:
        d = df[df.method == m].set_index("family").loc[fams]
        series[m] = (d.f1.values, d.ci_lo.values, d.ci_hi.values)
    fig, ax = plt.subplots(figsize=(9, 5))
    _grouped_bars(ax, ["Manhattan\n(CNN's training layout)", "curvy\n(unseen)", "via / pillar\n(unseen)"], series, t,
                  t["series"], METHODS)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("killer F1")
    ax.set_title(f"Exp 3 · zero-shot geometry: calibrated / trained on Manhattan only (n={int(df.n.iloc[0])} per bar)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=4, fontsize=9)
    _save(fig, "exp3_geometry", mode, t)


def fig_exp4(mode: str, t: dict) -> None:
    df = pd.read_csv(V3 / "exp4" / "misalignment.csv")
    kinds = ["nuisance", "bridge", "open", "particle", "all"]
    fig, ax = plt.subplots(figsize=(9, 4.6))
    cols = {"verifier_with_M3": t["series"]["verifier"], "verifier_without_M3": t["muted"]}
    labs = {"verifier_with_M3": "with Module 3 (sub-pixel alignment)", "verifier_without_M3": "without Module 3"}
    series = {}
    for m in cols:
        d = df[(df.method == m) & (df.metric == "unchanged")].set_index("kind").loc[kinds]
        series[m] = (d["mean"].values, d.ci_lo.values, d.ci_hi.values)
    _grouped_bars(ax, [k if k != "all" else "all kinds" for k in kinds], series, t, cols, labs)
    _target(ax, 0.95, t, "target ≥ 95% unchanged")
    ax.set_ylim(0, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("verdict class unchanged")
    n_all = int(df[(df.kind == "all")].n.iloc[0])
    ax.set_title(f"Exp 4 · verdict stability under ±8 px shift + ≤1 px warp (n={n_all} overall)")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=2, fontsize=9)
    _save(fig, "exp4_misalignment", mode, t)


def fig_exp5(mode: str, t: dict) -> None:
    df = pd.read_csv(V3 / "exp5" / "near_miss.csv")
    kinds = ["notch", "filament", "notch+filament"]
    rules = {"delta_beta_alone": "Δβ rule alone", "full_rule": "full rule (Δβ + bottleneck + SDF energy)"}
    cols = {"delta_beta_alone": t["muted"], "full_rule": t["series"]["verifier"]}
    series = {}
    for r in rules:
        d = df[(df.rule == r) & (df.metric == "recall")].set_index("kind").loc[kinds]
        series[r] = (d["mean"].values, d.ci_lo.values, d.ci_hi.values)
    fig, ax = plt.subplots(figsize=(9, 4.6))
    _grouped_bars(ax, ["notch\n(constriction, not severed)", "filament\n(stops 1–3 nm short)", "both"], series, t, cols, rules)
    _target(ax, 0.8, t, "target recall ≥ 80%", left=True)
    alarm = df[(df.kind == "nuisance") & (df.rule == "full_rule")].iloc[0]
    ax.set_ylim(0, 1.08)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.set_ylabel("near-miss recall")
    ax.set_title(f"Exp 5 · near-miss capture at a fixed false-alarm rate ({100 * alarm['mean']:.0f}% on nuisance, n={int(alarm.n)})")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=9)
    _save(fig, "exp5_nearmiss", mode, t)


def fig_exp6(mode: str, t: dict) -> None:
    df = pd.read_csv(V3 / "exp6" / "latency.csv")
    rows = [
        ("calibration (3 scribe patches, with registration)", "Calibration, once per layer", "verifier"),
        ("verifier: Module 3 alignment", "Verifier · M3 alignment", "verifier"),
        ("verifier: Module 3 SDF energy", "Verifier · M3 SDF energy", "verifier"),
        ("verifier: Module 4 topology (+ field)", "Verifier · M4 topology", "verifier"),
        ("verifier: total per patch (singleton path)", "Verifier · total per patch", "verifier"),
        ("die-to-die single (align + GAT + stats)", "Die-to-die (single) per patch", "d2d_single"),
        ("cnn batch 1 (default threads)", "CNN per patch (batch 1)", "cnn"),
    ]
    d = df.set_index("what")
    fig, ax = plt.subplots(figsize=(9, 4.4))
    y = np.arange(len(rows))[::-1]
    for yi, (what, lab, m) in zip(y, rows):
        v = d.loc[what, "median"]
        ax.barh(yi, v, 0.62, color=t["series"][m], alpha=1.0 if "total" in lab or m != "verifier" or "Calibration" in lab else 0.55, zorder=2)
        ax.text(v, yi, f"  {v:.1f} ms  (n={int(d.loc[what, 'n'])})", va="center", fontsize=9, color=t["ink"])
    ax.set_yticks(y, [r[1] for r in rows])
    ax.axvline(50, color=t["target"], lw=1.2, ls=(0, (4, 3)), zorder=1)
    ax.annotate("calibration target < 50 ms", (50, y[0] + 0.5), xytext=(4, 0), textcoords="offset points", fontsize=8.5,
                color=t["ink2"], va="center")
    ax.set_xlim(0, d.loc[rows[4][0], "median"] * 1.45)
    ax.set_xlabel("median latency (ms, single CPU process)")
    ax.grid(axis="y", visible=False)
    ax.set_title("Exp 6 · calibration and per-patch latency")
    _save(fig, "exp6_latency", mode, t, "synthetic data · this CPU, not bit-reproducible")


def fig_real(mode: str, t: dict) -> None:
    """Track C real-SEM rungs: KILLER / NEAR_MISS rate on clean (defect-free) tiles, read from each rung's summary.csv."""
    r1 = pd.read_csv(REAL / "rung1" / "summary.csv")
    r1 = r1[r1.scope == "arm C, all ZMW groups"].set_index("metric")
    pts = {"KILLER": {"orig": [], "fresh": []}, "NEAR_MISS": {"orig": [], "fresh": []}}
    pts["KILLER"]["orig"].append(("1", r1.loc["false alarm: KILLER on clean tiles"]))
    pts["NEAR_MISS"]["orig"].append(("1", r1.loc["NEAR_MISS on clean tiles"]))
    b = pd.read_csv(REAL / "rung1b" / "summary.csv")
    b = b[b.variant == "primary"]
    for pop in ("orig", "fresh"):
        pts["KILLER"][pop].append(("1b", b[(b.population == pop) & (b.metric == "KILLER on clean tiles (UNVERIFIED counts as not KILLER)")].iloc[0]))
    for rung in ("1c", "1d"):
        s = pd.read_csv(REAL / f"rung{rung}" / "summary.csv")
        for pop in ("orig", "fresh"):
            for v in ("KILLER", "NEAR_MISS"):
                pts[v][pop].append((rung, s[(s.population == pop) & (s.metric == f"{v} on clean tiles")].iloc[0]))
    order = ["1", "1b", "1c", "1d"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4), constrained_layout=True, sharey=True)
    pop_style = {"orig": (t["series"]["cnn"], "s", "original set (32 images)"),
                 "fresh": (t["series"]["verifier"], "o", "held-out fresh set (17 images)")}
    for ax, (verdict, target) in zip(axes, (("KILLER", 0.05), ("NEAR_MISS", 0.05))):
        for pop, (c, mk, lab) in pop_style.items():
            p = pts[verdict][pop]
            if not p:
                continue
            x = [order.index(r) + (0.08 if pop == "fresh" else -0.08) for r, _ in p]
            m = np.array([row["mean"] for _, row in p])
            lo, hi = np.array([row.ci_lo for _, row in p]), np.array([row.ci_hi for _, row in p])
            ax.errorbar(x, m, yerr=[m - lo, hi - m], color=c, marker=mk, ms=7, lw=0, elinewidth=2, capsize=3,
                        markeredgecolor=t["surface"], markeredgewidth=1.5, label=lab, zorder=3)
            for xi, v in zip(x, m):
                ax.annotate(f"{100 * v:.0f}%", (xi, v), xytext=(7, 4), textcoords="offset points", fontsize=8.5, color=t["ink"])
        _target(ax, target, t, "target ≤ 5%", left=True)
        ax.set_xticks(range(len(order)), [f"Rung {r}" for r in order])
        ax.set_xlim(-0.3, len(order) - 0.6)
        ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
        ax.set_title(f"{verdict} called on clean tiles")
    axes[1].text(order.index("1b"), 0.5, "not in the\nRung 1b summary", ha="center", va="center", fontsize=8.5, color=t["ink2"])
    axes[0].set_ylim(-0.02, 1.0)
    axes[0].legend(loc="upper right", fontsize=9)
    fig.suptitle("Track C · real SEM images of defect-free nanowell arrays (lower is better)", x=0.01, ha="left",
                 fontsize=13, fontweight="bold", color=t["ink"])
    _save(fig, "real_rungs", mode, t, "REAL SEM data (Zenodo 10.5281/zenodo.8060099) · fitted reference, not CAD · 95% bootstrap CI over images")


def fig_testbed(mode: str, t: dict) -> None:
    from nanovate.config import TestbedCfg
    from nanovate.testbed.dataset import make_sample
    from skimage.segmentation import find_boundaries

    kinds = [("nuisance", "roughness only"), ("bridge", "bridge · killer"), ("open", "open · killer"),
             ("particle", "particle · killer"), ("notch", "notch · near-miss"), ("filament", "filament · near-miss")]
    cfg = TestbedCfg()
    fig, axes = plt.subplots(2, 6, figsize=(13, 4.8), constrained_layout=True)
    for j, (kind, lab) in enumerate(kinds):
        s = make_sample("manhattan", "val", 0, j, kind, cfg, 0)
        img = s.image.astype(float)
        lo, hi = np.percentile(img, [1, 99])
        if s.mask.any():  # centre the 96-px crop on the defect
            cy, cx = (int(v) for v in np.argwhere(s.mask).mean(0))
        else:
            cy, cx = s.W.shape[0] // 2, s.W.shape[1] // 2
        y0, x0 = (int(np.clip(c - 48, 0, n - 96)) for c, n in zip((cy, cx), s.W.shape))
        sl = np.s_[y0:y0 + 96, x0:x0 + 96]
        axes[0, j].imshow(np.clip((img - lo) / (hi - lo), 0, 1)[sl], cmap="gray", interpolation="nearest")
        ov = np.zeros((*s.W.shape, 4))
        ov[s.W] = (0.85, 0.88, 0.92, 1)
        ov[~s.W] = (0.06, 0.09, 0.12, 1)
        edge = find_boundaries(s.mask, mode="outer")
        ov[edge] = (0.18, 0.83, 0.75, 1)
        ov[s.mask & s.W] = (0.18, 0.83, 0.75, 1)
        axes[1, j].imshow(ov[sl], interpolation="nearest")
        axes[0, j].set_title(lab, fontsize=10.5, loc="center")
        for ax in axes[:, j]:
            ax.set_xticks([]), ax.set_yticks([]), ax.grid(False)
            for sp in ax.spines.values():
                sp.set_visible(False)
    axes[0, 0].set_ylabel("simulated SEM\n(50 e⁻/px)", fontsize=9.5)
    axes[1, 0].set_ylabel("ground-truth wafer\n(defect in teal)", fontsize=9.5)
    _save(fig, "testbed_strip", mode, t, "synthetic data · Manhattan val, 96×96 nm crops around the defect")


FIGS = [fig_exp1, fig_exp2, fig_exp3, fig_exp4, fig_exp5, fig_exp6, fig_real, fig_testbed]


def main() -> None:
    for mode, t in THEMES.items():
        _style(t)
        for f in FIGS:
            f(mode, t)
            print(f"{f.__name__} ({mode})", flush=True)
    print("figures ->", OUT)


if __name__ == "__main__":
    main()
