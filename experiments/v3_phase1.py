"""v3 Phase 1: diagnose the noise-free ceiling of the frozen v2 verifier (Manhattan VAL only; synthetic data).

Question: are the noise-free errors caused by resolution (information lost) or by the decision logic
(information present, wrongly called)? Nothing here selects or changes a parameter: the frozen v2 config
(results/v2/calib/decisions.json) is applied as is, and the test split is never generated.

  1a oracle tau   per sample, Delta-beta at every tau of a fine grid (persistence computed once); a miss is
                  recoverable if SOME tau gives exactly the wafer's Delta-beta (strict: nothing else in the
                  patch breaks) or at least the right verdict class (loose).
  1b sampling     the same wafers re-rendered at 1x / 2x / 3x finer pixel sampling, blur / bloom / defects fixed
                  in nm; pixel-defined thresholds rescaled (nanovate.diagnostics docstring lists each one).
                  Supplementary arm (not in the brief): 2x sampling with beam + bloom widths halved in nm.
  1c miss log     every wrong call with kind, size (nm), Delta-beta, saddle field value at the defect, margin to tau.
Writes results/v3/phase1/. Usage: python -m experiments.v3_phase1 [--quick]
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict, dataclass
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from experiments.common import bootstrap_ci, bootstrap_stat, f1_score  # noqa: E402
from nanovate.config import TestbedCfg  # noqa: E402
from nanovate.diagnostics import defect_saddle, delta_beta_curve, margin, scaled_window, verify_scaled  # noqa: E402
from nanovate.modules.calibration import calibrate  # noqa: E402
from nanovate.pipeline import PipelineCfg  # noqa: E402
from nanovate.testbed.dataset import KINDS, get_layout, window  # noqa: E402
from nanovate.testbed.resample import make_fine_sample, make_fine_scribe_patches  # noqa: E402
from experiments.select_config import with_dose  # noqa: E402
from scipy import ndimage as ndi  # noqa: E402

KILLERS = ("bridge", "open", "particle")
FROZEN = Path("results/v2/calib/decisions.json")


@dataclass(frozen=True)
class Phase1Cfg:
    quick: bool = False
    seed: int = 0
    family: str = "manhattan"
    split: str = "val"                 # never 'test' in Phase 1
    layouts: int = 24                  # val layouts 0..23 (v2 selection used 0..7; all are val)
    dies: int = 2
    dose: float = 50.0                 # e-/px; images are the noise-free expected image at this dose
    scales: tuple = (1, 2, 3)          # sub-pixels per nm
    blur_arms: tuple = ((2, 0.5),)     # supplementary: (scale, beam+bloom width factor in nm)
    tau_grid: tuple = tuple(round(t, 2) for t in np.arange(1, 100) / 100)
    n_boot: int = 1000
    workers: int = 6

    @classmethod
    def quick_cfg(cls) -> "Phase1Cfg":
        return cls(quick=True, layouts=2, dies=1, scales=(1, 2), blur_arms=(), n_boot=200, workers=4)


def frozen_pc() -> PipelineCfg:
    return PipelineCfg(**json.loads(FROZEN.read_text())["pipeline_cfg"])


_CAL: dict = {}


def _cal(tcfg: TestbedCfg, seed: int, s: int, blur: float):
    key = (s, blur)
    if key not in _CAL:
        _CAL[key] = calibrate(make_fine_scribe_patches(tcfg, seed, s, blur), sigmas=(1.5 * s, 4.0 * s))
    return _CAL[key]


def _ok_masks(kind: str, curve: np.ndarray, true_db: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """(strict, loose) correctness of the call at each tau."""
    d0, d1 = curve[:, 0], curve[:, 1]
    bridge = (d0 < 0) | (d1 > 0)
    openp = ~bridge & ((d0 > 0) | (d1 < 0))
    strict = (d0 == true_db[0]) & (d1 == true_db[1])
    if kind == "bridge":
        loose = bridge
    elif kind in ("open", "particle"):
        loose = openp
    else:
        loose = (d0 == 0) & (d1 == 0)
    return strict, loose


def evaluate(spec: dict) -> tuple[dict, np.ndarray]:
    cfg = Phase1Cfg(**spec["cfg"])
    s, blur, lay, die, kind = spec["s"], spec["blur"], spec["layout"], spec["die"], spec["kind"]
    tcfg = with_dose(TestbedCfg(), cfg.dose)
    pc = PipelineCfg(**spec["pc"])
    L = get_layout(cfg.family, cfg.split, lay, tcfg, cfg.seed)
    smp = make_fine_sample(cfg.family, cfg.split, lay, die, kind, tcfg, cfg.seed, s, blur, layout=L)
    cal = _cal(tcfg, cfg.seed, s, blur)
    win = scaled_window(window(tcfg), s)
    r = verify_scaled(smp.expected, smp.cad, cal, win, pc, s)
    taus = np.asarray(cfg.tau_grid)
    curve = delta_beta_curve(r.diagrams, taus)
    true_db = (int(smp.topo["d_beta0"]), int(smp.topo["d_beta1"]))
    strict, loose = _ok_masks(kind, curve, true_db)
    sad = defect_saddle(kind, r.field, smp.cad, smp.W, smp.mask, smp.params.get("defect", {}), tcfg.layout.pad, s)
    row = dict(layout=lay, die=die, kind=kind, label=smp.label, killer=smp.killer, s=s, blur_scale=blur,
               true_db0=true_db[0], true_db1=true_db[1], verdict=r.verdict, db0=r.delta_beta[0], db1=r.delta_beta[1],
               flag_killer=r.verdict.startswith("KILLER"), bottleneck=r.bottleneck, energy=r.energy, peak_z=r.peak_z,
               saddle=sad["saddle"], peak=sad["peak"], geom=sad["geom"], margin=margin(kind, sad, pc.tau),
               cal_mu_diel=cal.mu_diel, cal_mu_cond=cal.mu_cond)
    for k, v in smp.params.get("defect", {}).items():
        if isinstance(v, (int, float)):
            row[f"p_{k}"] = float(v)
    if kind == "particle" and smp.mask.any():  # clearance to the nearest CAD conductor, nm
        row["p_clearance"] = float(ndi.distance_transform_edt(~smp.cad)[smp.mask].min() / s - 1.0 / s)
    ok_at_tau = bool(strict[np.isclose(taus, pc.tau)][0])
    row.update(correct_strict=ok_at_tau, correct_flag=(r.verdict.startswith("KILLER") == smp.killer),
               n_tau_strict=int(strict.sum()), n_tau_loose=int(loose.sum()),
               tau_ok_min=float(taus[strict].min()) if strict.any() else np.nan,
               tau_ok_max=float(taus[strict].max()) if strict.any() else np.nan,
               tau_ok_nearest=float(taus[strict][np.argmin(np.abs(taus[strict] - pc.tau))]) if strict.any() else np.nan)
    flags = (curve != 0).any(1)  # killer flag at each tau (Delta-beta != 0)
    return row, np.stack([flags, strict, loose], 1)


def run(cfg: Phase1Cfg) -> dict:
    out = Path("results/v3/quick/phase1" if cfg.quick else "results/v3/phase1")
    out.mkdir(parents=True, exist_ok=True)
    pc = frozen_pc()
    arms = [(s, 1.0) for s in cfg.scales] + list(cfg.blur_arms)
    specs = [dict(cfg=asdict(cfg), pc=pc.to_dict(), s=s, blur=b, layout=lay, die=die, kind=kind)
             for s, b in arms for lay in range(cfg.layouts) for die in range(cfg.dies) for kind in KINDS]
    # heavy (3x) arms first so the pool stays busy; results are re-sorted, so order does not matter
    specs.sort(key=lambda sp: -sp["s"])
    with ProcessPoolExecutor(cfg.workers) as ex:
        res = list(ex.map(evaluate, specs, chunksize=2))
    rows = pd.DataFrame([r for r, _ in res])
    taus = np.asarray(cfg.tau_grid)
    sweep = []
    for (r, m) in res:
        for j, t in enumerate(taus):
            sweep.append((r["s"], r["blur_scale"], r["layout"], r["die"], r["kind"], r["killer"], t, bool(m[j, 0]), bool(m[j, 1]), bool(m[j, 2])))
    sweep = pd.DataFrame(sweep, columns=["s", "blur_scale", "layout", "die", "kind", "killer", "tau", "flag", "ok_strict", "ok_loose"])
    rows = rows.sort_values(["blur_scale", "s", "layout", "die", "kind"], ascending=[False, True, True, True, True]).reset_index(drop=True)
    rows.to_csv(out / "samples.csv", index=False)
    sweep.to_csv(out / "tau_sweep.csv.gz", index=False, compression={"method": "gzip", "mtime": 0})
    tabs = summarise(rows, sweep, cfg, pc)
    tabs.update(analyse(rows, cfg))
    for name, t in tabs.items():
        t.to_csv(out / f"{name}.csv", index=False)
    (out / "config.json").write_text(json.dumps({"phase1_cfg": asdict(cfg), "pipeline_cfg_frozen_v2": pc.to_dict(),
                                                 "data": "synthetic", "split": cfg.split,
                                                 "rescaled_px_params": "pre_smooth*s, region_tol*s, margin*s, cal sigmas*s, energy smooth*s, gamma/s, cap*s, peak-z radius*s"},
                                                indent=2, sort_keys=True, default=str))
    _plot(rows, pc, out)
    return tabs


def _f1(df: pd.DataFrame, cfg: Phase1Cfg, col: str = "flag_killer") -> tuple[float, float, float]:
    return bootstrap_stat([df[col].values, df.killer.values], f1_score, cfg.n_boot, cfg.seed)


def summarise(rows: pd.DataFrame, sweep: pd.DataFrame, cfg: Phase1Cfg, pc: PipelineCfg) -> dict[str, pd.DataFrame]:
    sampling, oracle, glob, size = [], [], [], []
    for (b, s), d in rows.groupby(["blur_scale", "s"], sort=False):
        f, lo, hi = _f1(d, cfg)
        rec = dict(s=s, pixel_nm=round(1 / s, 3), blur_scale=b, f1=f, ci_lo=lo, ci_hi=hi, n=len(d), n_killer=int(d.killer.sum()))
        for k in KINDS:
            dk = d[d.kind == k]
            x = dk.flag_killer.values if k in KILLERS else (dk.verdict == "NUISANCE").values if k == "nuisance" else dk.flag_killer.values
            m, l_, h_ = bootstrap_ci(x.astype(float), cfg.n_boot, cfg.seed)
            name = f"recall_{k}" if k in KILLERS else ("suppression_nuisance" if k == "nuisance" else f"false_killer_{k}")
            rec.update({name: m, f"{name}_lo": l_, f"{name}_hi": h_, f"n_{k}": len(dk)})
        sampling.append(rec)
        # 1a oracle tau, per kind: wrong at frozen tau -> recoverable by some tau?
        sw = sweep[(sweep.s == s) & (sweep.blur_scale == b)]
        for k in KINDS:
            dk = d[d.kind == k]
            wrong = dk[~dk.correct_flag]
            rs, rl = (wrong.n_tau_strict > 0).values.astype(float), (wrong.n_tau_loose > 0).values.astype(float)
            ms, ls, hs = bootstrap_ci(rs, cfg.n_boot, cfg.seed)
            ml, ll, hl = bootstrap_ci(rl, cfg.n_boot, cfg.seed)
            oracle.append(dict(s=s, blur_scale=b, kind=k, n=len(dk), n_wrong_at_frozen_tau=len(wrong),
                               recoverable_strict=ms, strict_lo=ls, strict_hi=hs, n_recoverable_strict=int(rs.sum()),
                               recoverable_loose=ml, loose_lo=ll, loose_hi=hl, n_recoverable_loose=int(rl.sum()),
                               within_0p1_of_frozen=int(((wrong.tau_ok_nearest - pc.tau).abs() <= 0.1).sum())))
        # global tau sweep (in-sample, optimistic) and per-sample oracle upper bound
        for t, dt in sw.groupby("tau"):
            glob.append(dict(s=s, blur_scale=b, tau=t, f1=f1_score(dt.flag.values, dt.killer.values), n=len(dt)))
        per = sw.groupby(["layout", "die", "kind"]).agg(any_ok=("ok_strict", "any"), killer=("killer", "first")).reset_index()
        pred = np.where(per.any_ok, per.killer, ~per.killer)  # oracle: correct whenever some tau is correct
        fo, lo_, ho = bootstrap_stat([pred, per.killer.values], f1_score, cfg.n_boot, cfg.seed)
        g = pd.DataFrame([x for x in glob if x["s"] == s and x["blur_scale"] == b])
        best = g.loc[g.f1.idxmax()]
        sampling[-1].update(oracle_global_tau=float(best.tau), oracle_global_f1=float(best.f1),
                            oracle_per_sample_f1=fo, oracle_per_sample_lo=lo_, oracle_per_sample_hi=ho)
    # recall vs physical size (1x, nominal optics)
    d = rows[(rows.s == 1) & (rows.blur_scale == 1.0)]
    for k, col, edges in (("bridge", "p_gap", (0, 8, 12, 16, 26)), ("bridge", "p_radius", (3, 5, 7, 10.01)),
                          ("open", "p_length", (3, 4, 5, 6, 8.01)), ("particle", "p_clearance", (0, 3, 4, 6, 40)),
                          ("particle", "p_radius", (2, 3, 4, 6.01)), ("notch", "p_remaining", (1.5, 2.5, 3.5, 4.5)),
                          ("filament", "p_gap", (0, 1.5, 2.5, 4))):
        dk = d[d.kind == k]
        if col not in dk:
            continue
        for a, z in zip(edges[:-1], edges[1:]):
            sel = dk[(dk[col] >= a) & (dk[col] < z)]
            x = (sel.flag_killer if k in KILLERS else ~sel.flag_killer).values.astype(float)
            m, l_, h_ = bootstrap_ci(x, cfg.n_boot, cfg.seed)
            size.append(dict(kind=k, size=col[2:], lo_nm=a, hi_nm=z, metric="recall" if k in KILLERS else "not_called_killer",
                             mean=m, ci_lo=l_, ci_hi=h_, n=len(sel)))
    miss = rows[~rows.correct_flag].copy()
    keep = ["s", "blur_scale", "layout", "die", "kind", "label", "verdict", "true_db0", "true_db1", "db0", "db1", "saddle", "peak",
            "margin", "geom", "tau_ok_min", "tau_ok_max", "tau_ok_nearest", "n_tau_strict", "bottleneck", "energy", "peak_z"]
    keep += [c for c in rows.columns if c.startswith("p_") and c not in ("p_p", "p_q")]
    return {"sampling": pd.DataFrame(sampling), "oracle_tau": pd.DataFrame(oracle), "global_tau": pd.DataFrame(glob),
            "size_recall": pd.DataFrame(size), "miss_log": miss[[c for c in keep if c in miss]]}


def mechanism(r) -> str:
    """Why a call is wrong (rows from samples.csv). Margin > 0: the field at the defect shows the change."""
    called_bridge, called_split = r.db0 < 0 or r.db1 > 0, not (r.db0 < 0 or r.db1 > 0) and (r.db0 > 0 or r.db1 < 0)
    if r.kind == "nuisance":
        return "spurious_event_elsewhere"
    if pd.isna(r.margin):
        return "unmeasured"
    if r.killer:
        return "defect_seen_but_cancelled_or_masked" if r.margin > 0 else "defect_signal_below_tau"
    if r.kind == "filament" and called_split:
        return "filament_read_as_split"
    own = (r.kind == "notch" and called_split) or (r.kind == "filament" and called_bridge)
    return "near_miss_signal_past_tau" if own and r.margin > 0 else "spurious_event_elsewhere"


def separability(df: pd.DataFrame, pos: str, neg: str, change_if_high: bool) -> dict:
    """Can ANY single threshold on the defect saddle separate a killer kind from its near-miss look-alike?

    Returns n, the minimum number of errors over all thresholds, and the AUC (labels used: a diagnostic only).
    """
    x = df[df.kind.isin([pos, neg]) & df.saddle.notna()]
    v = (x.saddle.values if change_if_high else -x.saddle.values)
    y = (x.kind == pos).values
    best = max(int(((v >= t) == y).sum()) for t in np.unique(v))
    auc = float(np.mean([(a > b) + 0.5 * (a == b) for a in v[y] for b in v[~y]]))
    return dict(pair=f"{pos} vs {neg}", n=len(x), n_pos=int(y.sum()), min_errors_any_threshold=len(x) - best, auc=auc)


def analyse(rows: pd.DataFrame, cfg: Phase1Cfg) -> dict[str, pd.DataFrame]:
    mech, sep = [], []
    for (b, s), d in rows.groupby(["blur_scale", "s"], sort=False):
        w = d[~d.correct_flag]
        for (k, m), g in w.assign(mechanism=[mechanism(r) for r in w.itertuples()]).groupby(["kind", "mechanism"]):
            mech.append(dict(s=s, blur_scale=b, kind=k, mechanism=m, n=len(g), n_kind=int((d.kind == k).sum())))
        for pos, neg, hi in (("bridge", "filament", True), ("open", "notch", False)):
            sep.append(dict(s=s, blur_scale=b, **separability(d, pos, neg, hi)))
    return {"error_mechanism": pd.DataFrame(mech), "separability": pd.DataFrame(sep)}


def _plot(rows: pd.DataFrame, pc: PipelineCfg, out: Path) -> None:
    d = rows[(rows.s == 1) & (rows.blur_scale == 1.0) & rows.margin.notna()]
    fig, ax = plt.subplots(figsize=(7, 3.8))
    for i, k in enumerate(("bridge", "filament", "open", "notch", "particle")):
        dk = d[d.kind == k]
        ok = dk.correct_flag.values
        y = dk.margin.values
        ax.scatter(i + np.linspace(-0.2, 0.2, ok.sum()), y[ok], s=12, c="C0", label="correct call" if i == 0 else None)
        ax.scatter(i + np.linspace(-0.2, 0.2, (~ok).sum()), y[~ok], s=18, c="C3", marker="x", label="wrong call" if i == 0 else None)
    ax.axhline(0, ls=":", c="gray")
    ax.set_xticks(range(5)); ax.set_xticklabels(["bridge", "filament\n(near-miss)", "open", "notch\n(near-miss)", "particle"])
    ax.set_ylabel("margin toward a change call\n(field units; > 0 = change seen)")
    ax.set_title(f"Phase 1: field margin at the defect vs frozen tau {pc.tau}, noise-free, 1 nm/px (Manhattan val, synthetic)", fontsize=8)
    ax.legend(fontsize=8); fig.tight_layout(); fig.savefig(out / "defect_margin.png", dpi=120); plt.close(fig)

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--quick", action="store_true")
    ap.add_argument("--analyse-only", action="store_true", help="recompute error_mechanism / separability from samples.csv")
    a = ap.parse_args()
    c = Phase1Cfg.quick_cfg() if a.quick else Phase1Cfg()
    if a.analyse_only:
        out = Path("results/v3/quick/phase1" if c.quick else "results/v3/phase1")
        for name, tab in analyse(pd.read_csv(out / "samples.csv"), c).items():
            tab.to_csv(out / f"{name}.csv", index=False)
            print(tab.to_string())
        raise SystemExit
    t = run(c)
    pd.set_option("display.width", 250)
    for k in ("sampling", "oracle_tau"):
        print(t[k].to_string())
