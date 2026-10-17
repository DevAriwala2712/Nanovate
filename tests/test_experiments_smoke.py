"""Every experiment runs end-to-end on a tiny config, and two runs give identical non-timing results.

Tiny config: 1 layout x 1 die, one dose / sigma / grain, a handful of bootstrap resamples. Results go to a
temp dir (NANOVATE_RESULTS_ROOT). The shared Context fits D2D rules and loads m6_cnn/cnn.pt once per module.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from experiments import exp1_nuisance, exp2_dose, exp3_geometry, exp4_misalignment, exp5_nearmiss, exp6_latency, select_config
from experiments.common import ExpCfg
from experiments.harness import Context

TINY = ExpCfg(quick=True, layouts=1, dies=1, n_boot=20, doses=(50,), exp1_sigmas=(1.0,), exp1_grains=(0.0,), exp5_dies=1,
              exp6_calib_runs=2, exp6_patches=6, val_layouts=1, val_dies=1)


@pytest.fixture(scope="module")
def root(tmp_path_factory):
    return tmp_path_factory.mktemp("results")


@pytest.fixture(scope="module")
def ctx(root):
    import os

    os.environ["NANOVATE_RESULTS_ROOT"] = str(root)
    select_config.run(TINY)
    yield Context(TINY)
    os.environ.pop("NANOVATE_RESULTS_ROOT", None)


def test_select_config_writes_frozen_decisions(root, ctx):
    d = json.loads((Path(root) / "v2" / "quick" / "calib" / "decisions.json").read_text())
    assert d["pipeline_cfg"]["field_option"] in ("percentile", "calibrated", "template")
    assert d["near_miss_budget"] == 0.05 and d["quick"] is True and d["version"] == "v2"
    assert d["pipeline_cfg"]["h1_fill"] == "dielectric"
    assert (Path(root) / "v2" / "quick" / "calib" / "candidates.csv").exists()
    assert select_config.load_pipeline_cfg(TINY).tau == d["pipeline_cfg"]["tau"]


@pytest.mark.parametrize("mod,cols,files", [
    (exp1_nuisance, {"ler_sigma", "grain", "method", "metric", "mean", "ci_lo", "ci_hi", "n"}, ["nuisance_stress.csv", "nuisance_stress.png", "config.json"]),
    (exp2_dose, {"dose", "method", "f1", "ci_lo", "ci_hi", "n", "n_killer"}, ["dose_cliff.csv", "dose_cliff.png", "lowest_dose.csv", "config.json"]),
    (exp3_geometry, {"family", "method", "f1", "ci_lo", "ci_hi", "n"}, ["geometry_f1.csv", "geometry_f1.png", "f1_drop.csv", "config.json"]),
    (exp4_misalignment, {"method", "kind", "metric", "mean", "ci_lo", "ci_hi", "n"}, ["misalignment.csv", "misalignment.png", "config.json"]),
    (exp5_nearmiss, {"kind", "rule", "metric", "mean", "ci_lo", "ci_hi", "n"}, ["near_miss.csv", "near_miss.png", "config.json"]),
])
def test_experiment_runs_and_writes_csv_png_config(root, ctx, mod, cols, files):
    tab = mod.run(TINY, ctx)
    assert cols <= set(tab.columns) and len(tab) > 0 and (tab["n"] > 0).all()
    out = Path(root) / "v2" / "quick" / mod.__name__.split(".")[-1].split("_")[0].replace("exp", "exp")
    for f in files:
        assert (out / f).exists(), f
    cfg = json.loads((out / "config.json").read_text())
    assert cfg["data"] == "synthetic" and cfg["exp_cfg"]["quick"] is True and "pipeline_cfg" in cfg


def test_exp6_runs_and_reports_calibration_target(root, ctx):
    tab = exp6_latency.run(TINY, ctx)
    assert (tab["median"] > 0).all()
    cal = tab[tab.what.str.startswith("calibration (3 scribe patches, with")].iloc[0]
    assert cal["median"] > 0 and cal["target_ms"] == 50.0


@pytest.mark.parametrize("mod", [exp1_nuisance, exp3_geometry, exp4_misalignment, exp5_nearmiss])
def test_two_runs_give_identical_non_timing_tables(root, ctx, mod):
    a = mod.run(TINY, ctx)
    b = mod.run(TINY, Context(TINY))   # fresh context: calibrations, D2D rules and the CNN reloaded
    pd.testing.assert_frame_equal(a, b)


def test_readme_tables_come_from_the_csvs_and_update_is_idempotent(root, ctx, tmp_path):
    from experiments import report

    qroot = Path(root) / "v2" / "quick"
    exp2_dose.run(TINY, ctx)   # make the test independent of test order
    text = report.render_tables(qroot)
    assert "synthetic data" in text and "Exp 2" in text and "Pre-registered targets vs measured" in text
    e2 = pd.read_csv(qroot / "exp2" / "dose_cliff.csv")
    v = e2[(e2.method == "verifier") & (e2.dose == 50)].iloc[0]
    assert f"{v.f1:.2f} [{v.ci_lo:.2f}, {v.ci_hi:.2f}] (n={int(v.n)})" in text      # the number is the CSV's
    readme = tmp_path / "README.md"
    readme.write_text("# T\n\nintro\n")
    a = report.update_readme(readme, qroot)
    b = report.update_readme(readme, qroot)
    assert a == b and a.count(report.START) == 1 and a.startswith("# T\n\nintro")
    readme.write_text("before\n" + report.START + "\nOLD\n" + report.END + "\nafter\n")
    c = report.update_readme(readme, qroot)
    assert "OLD" not in c and c.startswith("before\n") and c.endswith("after\n")
