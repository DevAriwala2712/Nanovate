"""Part 1: day-0 review-queue projection from measured per-kind rates (pure arithmetic + CSV runner)."""
import numpy as np
import pandas as pd
import pytest

from experiments.day0_scenario import project, run

KILL, SAFE = ("bridge", "open", "particle"), ("nuisance", "notch", "filament")


def _rates(flag=None, nm=None):
    flag = flag or {}; nm = nm or {}
    return {"flag": {k: flag.get(k, 0.0) for k in KILL + SAFE}, "nm": {k: nm.get(k, 0.0) for k in KILL + SAFE}}


def test_perfect_detector_catches_every_killer_and_sends_nothing_harmless():
    r = project(_rates(flag={k: 1.0 for k in KILL}), prevalence=0.05)
    assert r["killers_total"] == pytest.approx(50) and r["killers_caught"] == pytest.approx(50)
    assert r["harmless_sent_as_killer"] == 0 and r["killers_missed"] == 0 and r["sites_to_engineer"] == pytest.approx(50)


def test_nuisance_flag_rate_scales_with_the_nuisance_share():
    r = project(_rates(flag={"nuisance": 0.1}), prevalence=0.05)
    assert r["harmless_sent_as_killer"] == pytest.approx(0.1 * 950 * 0.9)


def test_near_miss_alarm_on_a_killer_is_reviewed_not_missed():
    r = project(_rates(nm={"bridge": 1.0}), prevalence=0.06)
    assert r["killers_flagged_near_miss"] == pytest.approx(20) and r["killers_missed"] == pytest.approx(40)


def test_run_skips_a_method_column_that_is_absent(tmp_path, monkeypatch):
    from experiments.common import ExpCfg
    cfg = ExpCfg(quick=True, layouts=1, dies=1, n_boot=20, doses=(50,), exp1_sigmas=(1.0,), exp1_grains=(0.0,), exp5_dies=1,
                 exp6_calib_runs=1, exp6_patches=1, val_layouts=1, val_dies=1)
    monkeypatch.setenv("NANOVATE_RESULTS_ROOT", str(tmp_path))
    kinds = list(KILL + SAFE) * 4
    df = pd.DataFrame(dict(family="via", dose=50.0, kind=kinds,
                           verdict=["KILLER_BRIDGE" if k in KILL else "NUISANCE" for k in kinds],
                           d2d_single=[True] * len(kinds), d2d_double=[False] * len(kinds)))
    p = tmp_path / "samples.csv"; df.to_csv(p, index=False)
    tab = run(cfg, samples_csv=str(p))
    assert set(tab.method) == {"verifier", "d2d_single", "d2d_double"}
    v = tab[(tab.method == "verifier") & (tab.prevalence == 0.05) & (tab.metric == "killers_caught")].iloc[0]
    assert v["mean"] == pytest.approx(50) and (tmp_path / "v2" / "quick" / "day0_scenario" / "scenario.csv").exists()
