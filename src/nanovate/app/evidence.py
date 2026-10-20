"""/results: the README result tables and the day-0 tables, generated from the CSVs by experiments/report.py.

The app never formats a result number itself: it calls ``render_tables`` / ``render_day0`` (the functions that
write the README blocks), records which files they read, and splits their markdown into sections. The few facts
the Evidence tab states in words (where the verifier loses) are read from the same CSVs, with their source rows.
Only the selected pipeline version's tables are returned: v2 -> results/v2 (README "Results (v2)" and v2 day-0
blocks); v3 -> results/v3 (``experiments.v3_run.render_v3`` and the v3 day-0 block). Synthetic data throughout.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from experiments import report
from experiments.common import ExpCfg
from experiments.select_config import results_root


def _parse(md: str) -> list[dict]:
    """Markdown from report.py -> [{title, notes, headers, rows, after}] (one per table; prose before / after it)."""
    sections: list[dict] = []
    cur = {"title": "", "notes": [], "headers": [], "rows": [], "after": []}

    def flush():
        if cur["headers"] or cur["notes"]:
            sections.append(dict(cur))

    for block in re.split(r"\n\s*\n", md.strip()):
        lines = block.strip().splitlines()
        if not lines:
            continue
        if lines[0].startswith("#### "):
            flush()
            cur = {"title": lines[0][5:].strip(), "notes": [], "headers": [], "rows": [], "after": []}
            lines = lines[1:]
            if not lines:
                continue
        if all(ln.startswith("|") for ln in lines):
            cells = [[c.strip() for c in ln.strip().strip("|").split("|")] for ln in lines]
            if cur["headers"]:                        # a second table under one heading: new section, same title
                flush()
                cur = {"title": cur["title"], "notes": [], "headers": [], "rows": [], "after": []}
            cur["headers"] = cells[0]
            cur["rows"] = [r for r in cells[2:]]      # cells[1] is the --- separator
        else:
            text = " ".join(lines)
            if cur["headers"] and text.rstrip().endswith(":"):   # prose ending in ':' introduces the next table
                flush()
                cur = {"title": cur["title"], "notes": [], "headers": [], "rows": [], "after": []}
            (cur["after"] if cur["headers"] else cur["notes"]).append(text)
    flush()
    return sections


def _render_with_sources(fn, root: Path) -> tuple[str, list[str]]:
    """Run a report.py renderer and record every CSV it read (by wrapping report._read)."""
    read: list[str] = []
    orig = report._read

    def spy(r, exp, name):
        p = Path(r) / exp / f"{name}.csv"
        if p.exists():
            read.append(str(p))
        return orig(r, exp, name)

    report._read = spy
    try:
        md = fn(root)
    finally:
        report._read = orig
    return md, sorted(set(read))


def _readme_block(readme: Path, markers: tuple[str, str]) -> str | None:
    text = readme.read_text() if readme.exists() else ""
    start, end = markers
    if start not in text or end not in text:
        return None
    return text.split(start, 1)[1].split(end, 1)[0].strip()


def _losses(root: Path) -> dict:
    """Facts the Evidence tab states in words, each with the CSV rows it comes from (no number is typed)."""
    out: dict = {}
    lat = root / "exp6" / "latency.csv"
    if lat.exists():
        d = pd.read_csv(lat)
        v = d[d.what == "verifier: total per patch (singleton path)"]
        c = d[d.what == "cnn batch 1 (default threads)"]
        if len(v) and len(c):
            out["cnn_speed_ratio"] = {"value": float(v["median"].iloc[0] / c["median"].iloc[0]), "verifier_ms": float(v["median"].iloc[0]),
                                      "cnn_ms": float(c["median"].iloc[0]), "n": int(v.n.iloc[0]), "source": str(lat),
                                      "rows": ["verifier: total per patch (singleton path)", "cnn batch 1 (default threads)"]}
    geo = root / "exp3" / "geometry_f1.csv"
    if geo.exists():
        g = pd.read_csv(geo)

        def f1(m, fam):
            r = g[(g.method == m) & (g.family == fam)]
            return None if not len(r) else {"f1": float(r.f1.iloc[0]), "ci": [float(r.ci_lo.iloc[0]), float(r.ci_hi.iloc[0])], "n": int(r.n.iloc[0])}
        out["manhattan_f1"] = {"verifier": f1("verifier", "manhattan"), "cnn": f1("cnn", "manhattan"), "source": str(geo)}
        out["curvy_f1"] = {"verifier": f1("verifier", "curvy"), "cnn": f1("cnn", "curvy"), "source": str(geo)}
    be = root / "breakeven" / "summary.csv"
    if be.exists():
        b = pd.read_csv(be)
        out["breakeven"] = {"rows": b.astype(str).to_dict("records"), "source": str(be)}
    return out


def renderers(version: str):
    """(results tables renderer, README markers, day-0 renderer, day-0 markers) for one pipeline version."""
    if version == "v3":
        from experiments.v3_run import render_v3
        return (render_v3, (report.V3_START, report.V3_END),
                lambda r: report.render_day0(r, "day0", "v3"), (report.DAY0_V3_START, report.DAY0_V3_END))
    if version == "v2":
        return (report.render_tables, (report.START, report.END),
                lambda r: report.render_day0(r, version="v2"), (report.DAY0_START, report.DAY0_END))
    raise ValueError(f"no evidence tables for pipeline version {version!r}")


def evidence(repo: Path = Path("."), version: str = "v3") -> dict:
    root = results_root(ExpCfg.full(version))
    readme = repo / "README.md"
    tables_fn, tables_markers, day0_fn, day0_markers = renderers(version)
    tables_md, tables_src = _render_with_sources(tables_fn, root)
    day0_md, day0_src = _render_with_sources(day0_fn, root)
    if version == "v3":   # render_v3 reads these directly (not through report._read)
        tables_src += [str(root / f) for f in ("variants.csv", "phase2_targets.csv", "detection_limit.csv", "run_config.json") if (root / f).exists()]
    dec = root / "calib" / "decisions.json"
    return {
        "synthetic": True,
        "pipeline_version": version,
        "root": str(root),
        "readme": {"markdown": tables_md, "sections": _parse(tables_md), "sources": tables_src + ([str(dec)] if dec.exists() else []),
                   "in_sync_with_readme": _readme_block(readme, tables_markers) == tables_md.strip()},
        "day0": {"markdown": day0_md, "sections": _parse(day0_md), "sources": day0_src,
                 "in_sync_with_readme": _readme_block(readme, day0_markers) == day0_md.strip()},
        "losses": _losses(root),
    }
