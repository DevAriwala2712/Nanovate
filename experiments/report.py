"""README results tables, generated from the experiment CSVs only (never typed by hand).

``render_tables(root)`` reads results/expN/*.csv (+ results/calib/decisions.json) and returns markdown;
``update_readme(path, root)`` rewrites the text between the RESULTS markers (idempotent). Every table is
labelled synthetic data and carries n beside each number.
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

START, END = "<!-- RESULTS:START -->", "<!-- RESULTS:END -->"
V1_START, V1_END = "<!-- RESULTS_V1:START -->", "<!-- RESULTS_V1:END -->"
DAY0_START, DAY0_END = "<!-- DAY0:START -->", "<!-- DAY0:END -->"
METHOD_NAMES = {"verifier": "Verifier (ours)", "d2d_single": "Die-to-die (single)", "d2d_double": "Die-to-die (double)", "cnn": "CNN (Manhattan-trained)"}


def _ci(m, lo, hi, n, pct=False, nd=2) -> str:
    if pct:
        return f"{100 * m:.0f}% [{100 * lo:.0f}, {100 * hi:.0f}] (n={int(n)})"
    return f"{m:.{nd}f} [{lo:.{nd}f}, {hi:.{nd}f}] (n={int(n)})"


def _md(headers: list[str], rows: list[list[str]]) -> str:
    out = ["| " + " | ".join(headers) + " |", "|" + "|".join("---" for _ in headers) + "|"]
    out += ["| " + " | ".join(str(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def _read(root: Path, exp: str, name: str) -> pd.DataFrame | None:
    p = root / exp / f"{name}.csv"
    return pd.read_csv(p) if p.exists() else None


def _met(flag) -> str:
    return "yes" if bool(flag) else "**NO**"


def render_tables(root: str | Path) -> str:
    root = Path(root)
    parts: list[str] = []
    dec = root / "calib" / "decisions.json"
    mode = "quick (small n, smoke-level)" if "quick" in root.parts else "full"
    version = "v2" if "v2" in root.parts else "v1"
    head = f"_All numbers below are **synthetic data** ({mode} run, {version}; 95% percentile-bootstrap CIs in brackets, n beside every number)._"
    if version == "v2":
        head += ("\n\n_v2 is a **post-hoc revision**: its fixes were designed after the v1 test results were seen "
                 "(splice-seam holes, LER-robust near-miss selection, faster calibration; see `docs/V2_FIXES.md`). "
                 "Every v2 choice was made on Manhattan val only and the targets are unchanged; v1 numbers are kept below._")
    if dec.exists():
        d = json.loads(dec.read_text())
        c = d["pipeline_cfg"]
        head += (f"\n\n_Verifier config frozen on Manhattan val: field `{c['field_option']}`, region `{c['region_mode']}`, tau {c['tau']}, "
                 f"pre-smooth {c['pre_smooth']} px, d* {c['d_star']}, e* {c['e_star']}"
                 + (f", z* {c.get('z_star')}, holes `{c.get('h1_fill', 'cad')}`" if version == "v2" else "")
                 + f" (val killer F1 {d['chosen']['f1']:.2f}, n={d['chosen']['n']})._")
    parts.append(head)

    e1, e2, e3 = _read(root, "exp1", "nuisance_stress"), _read(root, "exp2", "dose_cliff"), _read(root, "exp3", "geometry_f1")
    d3, e4, e5, e6 = _read(root, "exp3", "f1_drop"), _read(root, "exp4", "misalignment"), _read(root, "exp5", "near_miss"), _read(root, "exp6", "latency")

    # ---- targets vs measured -------------------------------------------------------
    tv = []
    if e1 is not None:
        v = e1[(e1.method == "verifier") & (e1.ler_sigma <= 1.5) & e1.metric.isin(["suppression", "bridge_recall"])]
        if len(v):
            s, r = v[v.metric == "suppression"]["mean"].min(), v[v.metric == "bridge_recall"]["mean"].min()
            tv.append(["1 Nuisance stress", ">= 90% suppression and >= 90% bridge recall at LER sigma <= 1.5 nm",
                       f"worst cell: suppression {100 * s:.0f}%, bridge recall {100 * r:.0f}%", _met(s >= 0.9 and r >= 0.9)])
    if e2 is not None:
        v = e2[(e2.method == "verifier")]
        hi = v[v.dose >= 50]
        if len(hi):
            tv.append(["2 Dose cliff", "F1 >= 0.9 at lambda >= 50 e-/px", f"F1 at lambda >= 50: min {hi.f1.min():.2f} (max {hi.f1.max():.2f})", _met(hi.f1.min() >= 0.9)])
    if d3 is not None:
        v = d3[d3.method == "verifier"]
        if len(v):
            tv.append(["3 Zero-shot geometry", "F1 drop A -> B/C <= 5 points", "; ".join(f"{r.test_family}: {r.drop_points:+.1f} pts" for r in v.itertuples()), _met((v.drop_points <= 5).all())])
    if e4 is not None:
        v = e4[(e4.method == "verifier_with_M3") & (e4.kind == "all") & (e4.metric == "unchanged")]
        if len(v):
            tv.append(["4 Misalignment", ">= 95% verdicts unchanged (shift +-8 px + warp)", f"{100 * v['mean'].iloc[0]:.0f}% with Module 3", _met(v["mean"].iloc[0] >= 0.95)])
    if e5 is not None:
        v = e5[(e5.kind == "notch+filament") & (e5.rule == "full_rule")]
        if len(v):
            tv.append(["5 Near-miss capture", "recall >= 80% at the fixed false-alarm rate", f"{100 * v['mean'].iloc[0]:.0f}% recall", _met(v['mean'].iloc[0] >= 0.8)])
    if e6 is not None:
        v = e6[e6.what.str.startswith("calibration (3 scribe patches, with")]
        if len(v):
            tv.append(["6 Calibration time", "< 50 ms for 3 scribe patches", f"{v['median'].iloc[0]:.0f} ms (median of {int(v.n.iloc[0])})", _met(v["median"].iloc[0] < 50)])
    if tv:
        parts.append("#### Pre-registered targets vs measured (verifier)\n\n" + _md(["Exp", "Target (hypothesis)", "Measured", "Met"], tv))

    # ---- per-experiment tables -----------------------------------------------------
    if e1 is not None:
        sig = e1[e1.ler_sigma <= 1.5].ler_sigma.max()
        gr = e1.grain.max()
        rows = []
        for m, name in METHOD_NAMES.items():
            cells = []
            for metric in ("suppression", "not_killer", "bridge_recall"):
                r = e1[(e1.method == m) & (e1.metric == metric) & (e1.ler_sigma == sig) & (e1.grain == gr)]
                cells.append(_ci(r["mean"].iloc[0], r.ci_lo.iloc[0], r.ci_hi.iloc[0], r.n.iloc[0], pct=True) if len(r) else "n/a")
            rows.append([name, *cells])
        parts.append(f"#### Exp 1 - nuisance stress (LER sigma {sig:g} nm, grain {gr:.0%}; full grid in `{root.as_posix()}/exp1/`)\n\n" + _md(["Method", "Nuisance suppression (NUISANCE verdict)", "Nuisance not flagged KILLER (secondary)", "Bridge recall"], rows))
    if e2 is not None:
        doses = sorted(e2.dose.unique())
        rows = []
        for m, name in METHOD_NAMES.items():
            row = [name]
            for dz in doses:
                r = e2[(e2.method == m) & (e2.dose == dz)]
                row.append(_ci(r.f1.iloc[0], r.ci_lo.iloc[0], r.ci_hi.iloc[0], r.n.iloc[0]) if len(r) else "n/a")
            rows.append(row)
        low = _read(root, "exp2", "lowest_dose")
        extra = ""
        if low is not None:
            extra = "\n\nLowest dose with F1 >= 0.9: " + "; ".join(f"{METHOD_NAMES.get(r.method, r.method)}: {r.lowest_dose_f1_ge_0p9}" for r in low.itertuples())
        parts.append("#### Exp 2 - killer F1 vs dose (e-/px)\n\n" + _md(["Method", *[f"{dz:g}" for dz in doses]], rows) + extra)
        ce = _read(root, "exp2", "ceiling")
        if ce is not None:
            crow = [[r.what, _ci(r.f1, r.ci_lo, r.ci_hi, r.n), f"{100 * r.nuisance_suppression:.0f}%",
                     " / ".join(f"{100 * getattr(r, 'recall_' + k):.0f}%" for k in ("bridge", "open", "particle"))] for r in ce.itertuples()]
            last = ce.iloc[-1]
            sub = "; ".join(f"{k} {int(last['subres_' + k])}/{int(last['n_' + k])}" for k in ("bridge", "open", "particle") if f"subres_{k}" in ce)
            parts.append("#### Exp 2 diagnostic - verifier ceiling without shot noise (not a target)\n\n"
                         + _md(["Input", "Killer F1", "Nuisance suppression", "Recall bridge / open / particle"], crow)
                         + (f"\n\nKillers missed even noise-free (below this verifier's optical resolution): {sub}." if sub else ""))
    if e3 is not None:
        fams = list(e3.family.unique())
        rows = [[name, *[_ci(*(lambda r: (r.f1.iloc[0], r.ci_lo.iloc[0], r.ci_hi.iloc[0], r.n.iloc[0]))(e3[(e3.method == m) & (e3.family == f)])) for f in fams]]
                for m, name in METHOD_NAMES.items()]
        t = _md(["Method", *fams], rows)
        if d3 is not None:
            t += "\n\nF1 drop vs Manhattan (points, 95% CI): " + "; ".join(
                f"{METHOD_NAMES.get(r.method, r.method)} -> {r.test_family}: {r.drop_points:+.1f} [{r.ci_lo_points:+.1f}, {r.ci_hi_points:+.1f}]" for r in d3.itertuples())
        parts.append("#### Exp 3 - zero-shot geometry (killer F1)\n\n" + t)
    if e4 is not None:
        rows = []
        for m in ("verifier_with_M3", "verifier_without_M3"):
            r = e4[(e4.method == m) & (e4.kind == "all")]
            u, k = r[r.metric == "unchanged"], r[r.metric == "killer_flag_unchanged"]
            rows.append([m.replace("_", " "), _ci(u["mean"].iloc[0], u.ci_lo.iloc[0], u.ci_hi.iloc[0], u.n.iloc[0], pct=True),
                         _ci(k["mean"].iloc[0], k.ci_lo.iloc[0], k.ci_hi.iloc[0], k.n.iloc[0], pct=True)])
        parts.append("#### Exp 4 - verdicts unchanged vs zero shift (shift +-8 px + warp <= 1 px)\n\n" + _md(["Pipeline", "Verdict class unchanged", "Killer flag unchanged"], rows))
    if e5 is not None:
        rows = [[r.kind, r.rule.replace("_", " "), r.metric, _ci(r.mean, r.ci_lo, r.ci_hi, r.n, pct=True)] for r in e5.itertuples() if r.rule != "near_miss_class_only"]
        parts.append("#### Exp 5 - near-miss capture\n\n" + _md(["Kind", "Rule", "Metric", "Rate"], rows))
    if e6 is not None:
        rows = [[r.what, f"{r.median:.1f}", _ci(r.mean, r.ci_lo, r.ci_hi, r.n, nd=1)] for r in e6.itertuples()]
        parts.append("#### Exp 6 - latency (ms, this CPU; not bit-reproducible)\n\n" + _md(["What", "Median", "Mean [95% CI] (n)"], rows))
    return "\n\n".join(parts)


def update_block(path: str | Path, body: str, markers: tuple[str, str]) -> str:
    """Replace the text between ``markers`` in ``path`` with ``body`` (append a new block if absent). Idempotent."""
    start, end = markers
    path = Path(path)
    text = path.read_text()
    block = f"{start}\n{body}\n{end}"
    if start in text and end in text:
        head, rest = text.split(start, 1)
        _, tail = rest.split(end, 1)
        new = head + block + tail
    else:
        new = text.rstrip("\n") + "\n\n## Results\n\n" + block + "\n"
    path.write_text(new)
    return new


def update_readme(readme: str | Path, root: str | Path, markers: tuple[str, str] = (START, END)) -> str:
    """Replace the text between ``markers`` with freshly rendered tables. Returns the new text."""
    return update_block(readme, render_tables(root), markers)


DAY0_METHODS = {"verifier": "Verifier (no labels)", "cnn": "CNN (Manhattan-trained)", "d2d_single": "Die-to-die (single)",
                "d2d_double": "Die-to-die (double)"}


def render_day0(root: str | Path) -> str:
    """The day-0 use-case block: advantages vs disadvantages, the review-queue projection, the label break-even.

    Reads results/v2/{exp3/geometry_f1.csv, exp6/latency.csv, day0_scenario/scenario.csv, breakeven/summary.csv};
    any missing file drops its part (break-even says "not yet run"). Synthetic data throughout.
    """
    root = Path(root)
    parts = ["_All numbers are **synthetic data**; the review queue is a **projection** from measured per-kind rates "
             "(via family, dose 50) with an assumed site mix._"]
    geo, lat = _read(root, "exp3", "geometry_f1"), _read(root, "exp6", "latency")
    sc, be = _read(root, "day0_scenario", "scenario"), _read(root, "breakeven", "summary")
    if geo is not None:
        def f1(m, fam):
            r = geo[(geo.method == m) & (geo.family == fam)]
            return _ci(r.f1.iloc[0], r.ci_lo.iloc[0], r.ci_hi.iloc[0], r.n.iloc[0]) if len(r) else "n/a"
        lat_of = {}
        if lat is not None:
            for key, what in (("cal", "calibration (3 scribe patches, with registration)"), ("ver", "verifier: total per patch (singleton path)"),
                              ("cnn", "cnn batch 1 (default threads)"), ("d2d", "die-to-die single (align + GAT + stats)")):
                r = lat[lat.what == what]
                lat_of[key] = f"{r['median'].iloc[0]:.0f} ms" if len(r) else "n/a"
        rows = [["Labelled images needed before first use", "none (3 scribe images per layer)", "a labelled training set (Manhattan here)", "labelled val per dose, to fit the rule"],
                ["F1 on a new layout: via", f1("verifier", "via"), f1("cnn", "via"), f1("d2d_single", "via")],
                ["F1 on a new layout: curvy", f1("verifier", "curvy"), f1("cnn", "curvy"), f1("d2d_single", "curvy")],
                ["F1 on the layout the CNN was trained on (Manhattan)", f1("verifier", "manhattan"), f1("cnn", "manhattan"), f1("d2d_single", "manhattan")],
                ["Setup per layer", lat_of.get("cal", "n/a") + " calibration", "training run (GPU)", "rule fit"],
                ["Time per patch (this CPU)", lat_of.get("ver", "n/a"), lat_of.get("cnn", "n/a"), lat_of.get("d2d", "n/a")],
                ["Says why it flagged a site", "yes: delta-beta, bottleneck, SDF energy", "no (a probability)", "no (a difference map)"]]
        parts.append("#### Day 0 on a new layer: what each option gives you\n\n" + _md(["", "Verifier (ours)", "CNN (Manhattan-trained)", "Die-to-die (single)"], rows)
                     + "\n\nWhere the verifier loses: on a layout the CNN was trained on, the CNN is more accurate, and it is much faster per patch.")
    if sc is not None:
        methods = [m for m in DAY0_METHODS if m in set(sc.method)]

        def cell(m, metric, p):
            r = sc[(sc.method == m) & (sc.metric == metric) & (sc.prevalence == p)]
            return f"{r['mean'].iloc[0]:.0f} [{max(r.ci_lo.iloc[0], 0):.0f}, {r.ci_hi.iloc[0]:.0f}]" if len(r) else "n/a"
        p = 0.05
        labels = [("killers_caught", "Killers caught (flagged KILLER)"), ("killers_flagged_near_miss", "Killers flagged NEAR_MISS (still reviewed)"),
                  ("killers_missed", "Killers missed (passed as harmless)"), ("harmless_sent_as_killer", "Harmless sites flagged KILLER"),
                  ("near_miss_flagged", "Near-misses flagged NEAR_MISS (reliability bin)"), ("nuisance_flagged_near_miss", "Harmless sites flagged NEAR_MISS"),
                  ("sites_to_engineer", "All sites sent to an engineer")]
        rows = [[lab, *[cell(m, k, p) for m in methods]] for k, lab in labels]
        n_k = sc[(sc.metric == "killers_total") & (sc.prevalence == p)]["mean"].iloc[0]
        parts.append(f"#### Review queue: 1,000 flagged sites on a new via layer, {100 * p:.0f}% real killers ({n_k:.0f} sites)\n\n"
                     + _md(["Sites (95% CI)", *[DAY0_METHODS[m] for m in methods]], rows)
                     + "\n\nSite mix assumed: killers split equally bridge / open / particle; harmless sites 90% roughness-only, 5% notch, 5% filament "
                       "(near-misses). Only the verifier has a NEAR_MISS bin; the others must call a near-miss killer or harmless.")
        rows = [[f"{100 * q:.0f}%", *[f"{cell(m, 'killers_missed', q)} / {cell(m, 'harmless_sent_as_killer', q)}" for m in methods]] for q in sorted(set(sc.prevalence)) if q != p]
        if rows:
            parts.append("Other prevalences (killers missed / harmless sites flagged KILLER):\n\n" + _md(["Real killers", *[DAY0_METHODS[m] for m in methods]], rows))
    if be is None:
        parts.append("#### Label break-even\n\nThe label break-even experiment (how many labelled images of a new layout the CNN needs to match the verifier) is **not yet run**.")
    else:
        rows = [[r.family, r.threshold_rule, f"{r.verifier_f1:.2f}", str(r.breakeven_k), str(r.clearly_better_k), f"{int(r.n_seeds)} seeds, n={int(r.n_test)}"]
                for r in be.itertuples()]
        parts.append("#### Label break-even: labelled images of the new layout the CNN needs to match the verifier\n\n"
                     + _md(["Family", "CNN threshold", "Verifier F1", "Break-even k (seed-mean F1 >= verifier)", "Clearly better k (every seed's CI above)", "n"], rows))
    return "\n\n".join(parts)
