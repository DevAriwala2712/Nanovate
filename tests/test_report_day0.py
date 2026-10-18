"""Part 1: the generated day-0 section (README + docs/USE_CASE.md) renders from CSVs only and is idempotent."""
import shutil
from pathlib import Path

from experiments.report import DAY0_END, DAY0_START, render_day0, update_block


def _root_with_scenario(tmp_path):
    root = tmp_path / "v2"
    (root / "day0_scenario").mkdir(parents=True)
    shutil.copy("results/v2/day0_scenario/scenario.csv", root / "day0_scenario" / "scenario.csv")
    return root


def test_render_day0_without_breakeven_says_not_yet_run(tmp_path):
    md = render_day0(_root_with_scenario(tmp_path))
    assert "not yet run" in md and "Killers missed" in md and "synthetic" in md


def test_day0_block_update_is_idempotent(tmp_path):
    root = _root_with_scenario(tmp_path)
    doc = tmp_path / "USE_CASE.md"
    doc.write_text(f"# Use case\n\nprose\n\n{DAY0_START}\nold\n{DAY0_END}\n\ntail\n")
    update_block(doc, render_day0(root), (DAY0_START, DAY0_END))
    update_block(doc, render_day0(root), (DAY0_START, DAY0_END))
    text = doc.read_text()
    assert text.count(DAY0_START) == 1 and text.count(DAY0_END) == 1 and "old" not in text and text.endswith("tail\n")


def test_scenario_rows_name_non_killers_not_harmless():
    """'flagged KILLER' counts near-miss sites too, so the row must say non-killer, not harmless."""
    md = render_day0(_root_with_scenario(tmp_path_dummy()))
    assert "Non-killer sites flagged KILLER (incl. near-misses)" in md and "Harmless sites flagged KILLER" not in md


def tmp_path_dummy():
    import tempfile
    return Path(tempfile.mkdtemp())
