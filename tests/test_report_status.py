"""README status box: generated from the v3 and Track C CSVs (never typed), placed under the title, idempotent."""
import pandas as pd

from experiments.report import STATUS_END, STATUS_START, _ci, render_status, update_status


def test_status_numbers_come_from_the_csvs():
    md = render_status(app_default="v3")
    g = pd.read_csv("results/v3/exp3/geometry_f1.csv")
    for m in ("verifier", "cnn"):
        r = g[(g.method == m) & (g.family == "via")].iloc[0]
        assert _ci(r.f1, r.ci_lo, r.ci_hi, r.n) in md
    real = pd.read_csv("results/real/rung1/summary.csv")
    r = real[(real.metric == "false alarm: KILLER on clean tiles") & (real.scope == "arm C, all ZMW groups")].iloc[0]
    assert f"{100 * r['mean']:.1f}%" in md and "docs/REAL_DATA_RUNG1.md" in md and "not met" in md
    assert "Synthetic-data prototype" in md and "Where the CNN wins" in md and "runs pipeline **v3** by default" in md
    assert 5 <= md.count("\n> - ") + 1 <= 7


def test_status_default_version_is_the_apps():
    from nanovate.app.engine import DEFAULT_VERSION
    assert f"runs pipeline **{DEFAULT_VERSION}** by default" in render_status()


def test_update_status_goes_under_the_title_and_is_idempotent(tmp_path):
    doc = tmp_path / "README.md"
    doc.write_text("# Title\n\nbody\n")
    update_status(doc, app_default="v3")
    update_status(doc, app_default="v3")
    text = doc.read_text()
    assert text.startswith(f"# Title\n\n{STATUS_START}\n") and text.count(STATUS_START) == 1 and text.endswith("body\n")
    assert text.split(STATUS_END)[0].count("> - ") == render_status(app_default="v3").count("> - ")


def test_readme_status_block_is_in_sync():
    text = open("README.md").read()
    assert text.split(STATUS_START, 1)[1].split(STATUS_END, 1)[0].strip() == render_status().strip()


def test_status_reports_the_latest_real_rung():
    md = render_status(app_default="v3")
    s = pd.read_csv("results/real/rung1d/summary.csv")
    r = s[(s.population == "fresh") & (s.metric == "NEAR_MISS on clean tiles")].iloc[0]
    assert "Latest (Rung 1d)" in md and "docs/REAL_DATA_RUNG1D.md" in md
    assert _ci(r["mean"], r.ci_lo, r.ci_hi, r.n_tiles, pct=True) in md
