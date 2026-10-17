"""Experiment infrastructure: bootstrap CIs on known distributions, F1, config presets, writer."""
import json

import numpy as np
import pandas as pd
import pytest

from experiments.common import ExpCfg, bootstrap_ci, bootstrap_diff, bootstrap_stat, f1_score, write_results


def test_bootstrap_mean_ci_matches_the_normal_theory_interval():
    x = np.random.default_rng(0).normal(0, 1, 400)
    m, lo, hi = bootstrap_ci(x, n_boot=1000, seed=0)
    assert m == pytest.approx(x.mean()) and lo < m < hi
    se = 1 / np.sqrt(400)
    assert hi - lo == pytest.approx(2 * 1.96 * se, rel=0.2)


def test_bootstrap_coverage_of_a_bernoulli_proportion_is_about_95_percent():
    rng = np.random.default_rng(1)
    p, n, hits = 0.7, 80, 0
    for r in range(150):
        x = (rng.random(n) < p).astype(float)
        _, lo, hi = bootstrap_ci(x, n_boot=300, seed=r)
        hits += lo <= p <= hi
    assert 0.88 <= hits / 150 <= 0.99


def test_bootstrap_is_deterministic_and_handles_degenerate_input():
    x = np.array([1.0, 0.0, 1.0, 1.0])
    assert bootstrap_ci(x, 200, seed=3) == bootstrap_ci(x, 200, seed=3)
    assert bootstrap_ci(np.ones(10), 100, seed=0) == (1.0, 1.0, 1.0)
    m, lo, hi = bootstrap_ci(np.array([]), 100, seed=0)
    assert np.isnan(m) and np.isnan(lo) and np.isnan(hi)


def test_f1_and_its_bootstrap():
    y = np.array([1, 1, 0, 0, 1, 0], bool)
    assert f1_score(y, y) == 1.0
    assert f1_score(np.zeros(6, bool), y) == 0.0
    assert f1_score(np.ones(6, bool), y) == pytest.approx(2 * 3 / (2 * 3 + 3))
    m, lo, hi = bootstrap_stat([y, y], lambda a, b: f1_score(a, b), n_boot=100, seed=0)
    assert (m, lo, hi) == (1.0, 1.0, 1.0)
    pred = np.array([1, 0, 0, 0, 1, 1], bool)
    m, lo, hi = bootstrap_stat([pred, y], lambda a, b: f1_score(a, b), n_boot=500, seed=0)
    assert lo <= m <= hi and hi - lo > 0.1


def test_quick_is_much_smaller_than_full_and_roundtrips():
    q, f = ExpCfg.quick(), ExpCfg.full()
    assert q.quick and not f.quick and q.n_boot == 200 and f.n_boot == 1000
    assert q.layouts < f.layouts and len(q.doses) <= len(f.doses)
    assert ExpCfg.from_dict(q.to_dict()) == q


def test_write_results_saves_csv_and_config(tmp_path):
    df = pd.DataFrame({"a": [1, 2]})
    write_results(tmp_path / "expX", "table", df, {"seed": 0, "x": ExpCfg.quick().to_dict()})
    assert pd.read_csv(tmp_path / "expX" / "table.csv").equals(df)
    cfg = json.loads((tmp_path / "expX" / "config.json").read_text())
    assert cfg["seed"] == 0 and cfg["x"]["quick"] is True


def test_bootstrap_diff_recovers_a_known_difference_of_means():
    rng = np.random.default_rng(2)
    a, b = rng.normal(1.0, 1, 300), rng.normal(0.0, 1, 300)
    d, lo, hi = bootstrap_diff([a], [b], lambda x: x.mean(), n_boot=500, seed=0)
    assert d == pytest.approx(a.mean() - b.mean()) and lo < 1.0 < hi and lo > 0.5
    d0, lo0, hi0 = bootstrap_diff([a], [a.copy()], lambda x: x.mean(), n_boot=500, seed=0)
    assert d0 == 0.0 and lo0 < 0 < hi0
