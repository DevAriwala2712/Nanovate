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
DAY0_V3_START, DAY0_V3_END = "<!-- DAY0_V3:START -->", "<!-- DAY0_V3:END -->"
STATUS_START, STATUS_END = "<!-- STATUS:START -->", "<!-- STATUS:END -->"
REAL_RUNG1 = Path("results/real/rung1/summary.csv")
V3_START, V3_END = "<!-- RESULTS_V3:START -->", "<!-- RESULTS_V3:END -->"
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
    version = "v3" if "v3" in root.parts else "v2" if "v2" in root.parts else "v1"
    head = f"_All numbers below are **synthetic data** ({mode} run, {version}; 95% percentile-bootstrap CIs in brackets, n beside every number)._"
    if version == "v2":
        head += ("\n\n_v2 is a **post-hoc revision**: its fixes were designed after the v1 test results were seen "
                 "(splice-seam holes, LER-robust near-miss selection, faster calibration; see `docs/V2_FIXES.md`). "
                 "Every v2 choice was made on Manhattan val only and the targets are unchanged; v1 numbers are kept below._")
    if version == "v3":
        head += ("\n\n_v3 replaces the single global tau for **merges** with a per-gap rule against the template-expected merge level, "
                 "and calls a patch from its **located events** (no Delta-beta sum, so a seen bridge is never cancelled); cuts and necks keep "
                 "v2's rule (an optical limit, see `docs/V3_PHASE1_DIAGNOSIS.md`). Every v3 choice was made on Manhattan val only and evaluated "
                 "once on a **fresh test seed**; targets are unchanged; v2 and v1 numbers are kept below._")
    if dec.exists() and version == "v3":
        d = json.loads(dec.read_text())
        c, r = d["pipeline_cfg"], d["event_rule_cfg"]
        head += (f"\n\n_Verifier config frozen on Manhattan val: v2 field / region / tau {c['tau']}; merge margin {r['m_merge']}, "
                 f"grain normalisation {'m_norm ' + str(r['m_norm']) if r['normalise'] else 'off'}, per-neck cut margin {r['m_cut']}, "
                 f"protrusion edge level {r['e_edge']}; d* {c['d_star']}, e* {c['e_star']}, z* {c['z_star']} "
                 f"(val killer F1 {d['chosen']['f1']:.2f}, n={int(d['chosen']['n'])}); test seed {d['test_seed']}._")
    elif dec.exists():
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
            if version == "v3":  # report the change (B/C minus A): positive = improved, so "-7.3 pts" no longer reads as a loss
                meas = "; ".join(f"{r.test_family}: {-r.drop_points:+.1f} pts ({'improved' if r.drop_points < 0 else 'dropped'})" for r in v.itertuples())
            else:
                meas = "; ".join(f"{r.test_family}: {r.drop_points:+.1f} pts" for r in v.itertuples())
            tv.append(["3 Zero-shot geometry", "F1 drop A -> B/C <= 5 points", meas, _met((v.drop_points <= 5).all())])
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
        if d3 is not None and version == "v3":
            t += "\n\nF1 change vs Manhattan (points, 95% CI; positive = better on the new layout): " + "; ".join(
                f"{METHOD_NAMES.get(r.method, r.method)} -> {r.test_family}: {-r.drop_points:+.1f} [{-r.ci_hi_points:+.1f}, {-r.ci_lo_points:+.1f}]" for r in d3.itertuples())
        elif d3 is not None:
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


def render_day0(root: str | Path, scenario_dir: str = "day0_scenario", version: str = "") -> str:
    """The day-0 use-case block: advantages vs disadvantages, the review-queue projection, the label break-even.

    Reads <root>/{exp3/geometry_f1.csv, exp6/latency.csv, <scenario_dir>/scenario.csv, breakeven/summary.csv}
    (v2: results/v2, scenario_dir day0_scenario; v3: results/v3, scenario_dir day0); any missing file drops its part
    (break-even says "not yet run"). ``version`` (e.g. "v2") labels every heading. Synthetic data throughout.
    """
    root = Path(root)
    tag = f" ({version})" if version else ""
    parts = [f"_{version + ': ' if version else ''}All numbers are **synthetic data**; the review queue is a **projection** from measured per-kind rates "
             f"(via family, dose 50{f', `results/{version}/exp3/samples.csv`' if version else ''}) with an assumed site mix._"]
    geo, lat = _read(root, "exp3", "geometry_f1"), _read(root, "exp6", "latency")
    sc, be = _read(root, scenario_dir, "scenario"), _read(root, "breakeven", "summary")
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
                ["Says why it flagged a site", "yes: located merge/cut events, bottleneck, SDF energy" if version == "v3" else "yes: delta-beta, bottleneck, SDF energy",
                 "no (a probability)", "no (a difference map)"]]
        parts.append(f"#### Day 0 on a new layer{tag}: what each option gives you\n\n" + _md(["", "Verifier (ours)", "CNN (Manhattan-trained)", "Die-to-die (single)"], rows)
                     + "\n\nWhere the verifier loses: on a layout the CNN was trained on, the CNN is more accurate, and it is much faster per patch.")
    if sc is not None:
        methods = [m for m in DAY0_METHODS if m in set(sc.method)]

        def cell(m, metric, p):
            r = sc[(sc.method == m) & (sc.metric == metric) & (sc.prevalence == p)]
            return f"{r['mean'].iloc[0]:.0f} [{max(r.ci_lo.iloc[0], 0):.0f}, {r.ci_hi.iloc[0]:.0f}]" if len(r) else "n/a"
        p = 0.05
        labels = [("killers_caught", "Killers caught (flagged KILLER)"), ("killers_flagged_near_miss", "Killers flagged NEAR_MISS (still reviewed)"),
                  ("killers_missed", "Killers missed (passed as harmless)"), ("harmless_sent_as_killer", "Non-killer sites flagged KILLER (incl. near-misses)"),
                  ("near_miss_flagged", "Near-misses flagged NEAR_MISS (reliability bin)"), ("nuisance_flagged_near_miss", "Harmless sites flagged NEAR_MISS"),
                  ("sites_to_engineer", "All sites sent to an engineer")]
        rows = [[lab, *[cell(m, k, p) for m in methods]] for k, lab in labels]
        n_k = sc[(sc.metric == "killers_total") & (sc.prevalence == p)]["mean"].iloc[0]
        parts.append(f"#### Review queue{tag}: 1,000 flagged sites on a new via layer, {100 * p:.0f}% real killers ({n_k:.0f} sites)\n\n"
                     + _md(["Sites (95% CI)", *[DAY0_METHODS[m] for m in methods]], rows)
                     + "\n\nSite mix assumed: killers split equally bridge / open / particle; harmless sites 90% roughness-only, 5% notch, 5% filament "
                       "(near-misses). Only the verifier has a NEAR_MISS bin; the others must call a near-miss killer or harmless.")
        rows = [[f"{100 * q:.0f}%", *[f"{cell(m, 'killers_missed', q)} / {cell(m, 'harmless_sent_as_killer', q)}" for m in methods]] for q in sorted(set(sc.prevalence)) if q != p]
        if rows:
            parts.append("Other prevalences (killers missed / non-killer sites flagged KILLER):\n\n" + _md(["Real killers", *[DAY0_METHODS[m] for m in methods]], rows))
    if be is None:
        parts.append(f"#### Label break-even{tag}\n\nThe label break-even experiment (how many labelled images of a new layout the CNN needs to match the verifier) is **not yet run**"
                     + (f" for {version} (the v2 break-even compares the CNN against the v2 verifier)." if version and version != "v2" else "."))
    else:
        rows = [[r.family, r.threshold_rule, f"{r.verifier_f1:.2f}", str(r.breakeven_k), str(r.clearly_better_k), f"{int(r.n_seeds)} seeds, n={int(r.n_test)}"]
                for r in be.itertuples()]
        parts.append(f"#### Label break-even{tag}: labelled images of the new layout the CNN needs to match the verifier\n\n"
                     + _md(["Family", "CNN threshold", "Verifier F1", "Break-even k (seed-mean F1 >= verifier)", "Clearly better k (every seed's CI above)", "n"], rows))
    return "\n\n".join(parts)


def render_status(v3_root: str | Path = "results/v3", real_csv: str | Path = REAL_RUNG1, app_default: str | None = None) -> str:
    """The README status box (directly under the title): every number is read from the v3 / Track C result CSVs.

    Reads <v3_root>/{exp3/geometry_f1.csv, day0/scenario.csv, phase2_targets.csv, exp2/lowest_dose.csv, exp6/latency.csv}
    and results/real/rung1/summary.csv; ``app_default`` defaults to ``nanovate.app.engine.DEFAULT_VERSION``.
    """
    root = Path(v3_root)
    if app_default is None:
        from nanovate.app.engine import DEFAULT_VERSION as app_default  # noqa: N811 (the app's own default)
    geo, sc, lat = _read(root, "exp3", "geometry_f1"), _read(root, "day0", "scenario"), _read(root, "exp6", "latency")
    tg = pd.read_csv(root / "phase2_targets.csv") if (root / "phase2_targets.csv").exists() else None
    low = _read(root, "exp2", "lowest_dose")
    real = pd.read_csv(real_csv) if Path(real_csv).exists() else None

    def f1(m, fam):
        r = geo[(geo.method == m) & (geo.family == fam)] if geo is not None else []
        return _ci(r.f1.iloc[0], r.ci_lo.iloc[0], r.ci_hi.iloc[0], r.n.iloc[0]) if len(r) else "n/a"

    def missed(m):
        r = sc[(sc.method == m) & (sc.metric == "killers_missed") & (sc.prevalence == 0.05)] if sc is not None else []
        return f"{r['mean'].iloc[0]:.0f} [{max(r.ci_lo.iloc[0], 0):.0f}, {r.ci_hi.iloc[0]:.0f}]" if len(r) else "n/a"

    def target(prefix):
        r = tg[tg.target.str.startswith(prefix)] if tg is not None else []
        if not len(r):
            return "n/a"
        r = r.iloc[0]
        return f"{_ci(r['mean'], r.ci_lo, r.ci_hi, r.n, pct=True)} at {r.cell}"

    def ms(what):
        r = lat[lat.what == what] if lat is not None else []
        return f"{r['median'].iloc[0]:.1f} ms" if len(r) else "n/a"

    def lowest(m):
        r = low[low.method == m] if low is not None else []
        v = str(r.iloc[0, 1]) if len(r) else "n/a"
        return "none" if v == "none" else f"{float(v):g} e⁻/px" if v != "n/a" else v

    n_k = sc[(sc.metric == "killers_total") & (sc.prevalence == 0.05)]["mean"].iloc[0] if sc is not None else float("nan")
    if real is not None:
        r = real[(real.metric == "false alarm: KILLER on clean tiles") & (real.scope == "arm C, all ZMW groups")].iloc[0]
        ok = r["mean"] <= 0.05          # pre-registered target (docs/REAL_DATA_PREREG.md): false KILLER <= 5% of clean tiles
        real_line = (f"**Real SEM data (Track C Rung 1): {'target met' if ok else 'not yet working'}.** On real images of defect-free hole arrays the frozen v3 verifier "
                     f"calls {100 * r['mean']:.1f}% [{100 * r.ci_lo:.1f}, {100 * r.ci_hi:.1f}] (n={int(r.n_tiles)} tiles) KILLER ({int(r.n_images)} images; pre-registered target "
                     f"≤ 5%, {'met' if ok else '**not met**'}; [docs/REAL_DATA_RUNG1.md](docs/REAL_DATA_RUNG1.md))."
                     + ("" if ok else " Fixing this is the next step."))
    else:
        real_line = "**Real SEM data (Track C Rung 1):** not run yet."
    lines = [
        "**Synthetic-data prototype.** Every number below except the real-data line is measured on the simulated SEM testbed "
        "(v3: rule frozen on Manhattan val, Experiments 1–6 run once on a fresh test seed); 95% bootstrap CIs, n beside each.",
        f"**v3 headline (synthetic):** killer F1 on an unseen via layout {f1('verifier', 'via')} vs the Manhattan-trained CNN {f1('cnn', 'via')}; "
        f"curvy {f1('verifier', 'curvy')} vs {f1('cnn', 'curvy')}. Review-queue projection (via, {n_k:.0f} killers in 1,000 sites): "
        f"{missed('verifier')} killers missed vs {missed('cnn')} for the CNN, n=144 samples per method.",
        f"**v3 nuisance stress (synthetic):** worst cell at LER ≤ 1.5 nm, suppression {target('Exp 1 suppression')}, "
        f"bridge recall {target('Exp 1 bridge recall')} (target ≥ 90% / ≥ 90%: "
        f"{'met' if tg is not None and tg[tg.target.str.startswith('Exp 1')].met.all() else '**not met**'}).",
        f"**Where the CNN wins (synthetic):** on its training layout (Manhattan) F1 {f1('cnn', 'manhattan')} vs {f1('verifier', 'manhattan')}; "
        f"at low dose (lowest dose with F1 ≥ 0.9: CNN {lowest('cnn')}, verifier {lowest('verifier')}); and speed, "
        f"{ms('cnn batch 1 (default threads)')} vs {ms('verifier: total per patch (singleton path)')} per patch (median, this CPU).",
        real_line,
        f"**Demo app** (`python -m nanovate.app`) runs pipeline **{app_default}** by default; `--pipeline-version v2` switches to v2.",
    ]
    return "> **Status** _(generated from the result CSVs by `scripts/reproduce.py`; not typed by hand)_\n>\n" + "\n".join(f"> - {ln}" for ln in lines)


def update_status(readme: str | Path = "README.md", **kw) -> str:
    """Write the status box between STATUS markers, inserting the markers under the title on first use."""
    path = Path(readme)
    text = path.read_text()
    if STATUS_START not in text:
        title, rest = text.split("\n", 1)
        path.write_text(f"{title}\n\n{STATUS_START}\n{STATUS_END}\n{rest}")
    return update_block(path, render_status(**kw), (STATUS_START, STATUS_END))
