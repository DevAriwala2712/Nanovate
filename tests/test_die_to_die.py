"""M6 baseline: die-to-die difference (single and double) with val-only threshold selection."""
import numpy as np
import pytest

from nanovate.baselines.die_to_die import D2DRule, d2d_flag, d2d_score, d2d_stats, fit_d2d_rule
from nanovate.config import TestbedCfg
from nanovate.modules.calibration import calibrate
from nanovate.testbed.dataset import get_layout, make_sample, window
from nanovate.testbed.scribe import make_scribe_patches


@pytest.fixture(scope="module")
def world():
    import dataclasses

    base = TestbedCfg()
    cfg = dataclasses.replace(base, sem=dataclasses.replace(base.sem, dose=100.0))
    cal = calibrate(make_scribe_patches(cfg, seed=0))
    layout = get_layout("manhattan", "val", 1, cfg, 4)
    refs = [make_sample("manhattan", "val", 1, 900 + j, "nuisance", cfg, 4, layout=layout) for j in range(2)]
    return cfg, cal, layout, refs, window(cfg)


# Behavioural fixture only (read off one val layout at dose 100: unsmoothed maps separate best);
# the real rule is fitted on Manhattan val by fit_d2d_rule.
RULE = D2DRule(mode="single", sigma=0.0, theta=5.0, min_area=6)


def test_flags_an_obvious_bridge(world):
    cfg, cal, layout, refs, win = world
    tgt = make_sample("manhattan", "val", 1, 0, "bridge", cfg, 4, layout=layout)
    score = d2d_score(tgt.image, [refs[0].image], layout.cad, cal, mode="single")
    flagged, area = d2d_flag(score, RULE, win)
    assert flagged and area >= RULE.min_area


def test_does_not_flag_a_clean_die_or_a_reacquisition(world):
    cfg, cal, layout, refs, win = world
    clean = make_sample("manhattan", "val", 1, 1, "nuisance", cfg, 4, layout=layout)
    again = make_sample("manhattan", "val", 1, 901, "nuisance", cfg, 4, layout=layout, acq_tag="b")
    for img in (clean.image, again.image):
        score = d2d_score(img, [refs[0].image], layout.cad, cal, mode="single")
        assert d2d_flag(score, RULE, win)[0] is False
    # literally identical image -> zero difference
    zero = d2d_score(refs[0].image, [refs[0].image], layout.cad, cal, mode="single")
    assert np.abs(zero).max() < 1e-9


def test_double_difference_cancels_a_defect_in_one_reference(world):
    cfg, cal, layout, refs, win = world
    bad_ref = make_sample("manhattan", "val", 1, 2, "bridge", cfg, 4, layout=layout)
    clean_target = make_sample("manhattan", "val", 1, 3, "nuisance", cfg, 4, layout=layout)
    single = d2d_score(clean_target.image, [bad_ref.image], layout.cad, cal, mode="single")
    double = d2d_score(clean_target.image, [bad_ref.image, refs[1].image], layout.cad, cal, mode="double")
    assert d2d_flag(single, RULE, win)[0] is True        # the single-reference false alarm
    assert d2d_flag(double, D2DRule("double", 0.0, 5.0, 6), win)[0] is False


def test_threshold_selection_only_on_manhattan_val():
    stats = [np.zeros((3, 4))]
    kw = dict(labels=[True], mode="single", sigmas=(0.0, 1.0, 2.0), thetas=(2.0, 3.0, 4.0, 5.0), areas=(4, 8))
    with pytest.raises(ValueError):
        fit_d2d_rule(stats, split="test", family="manhattan", **kw)
    with pytest.raises(ValueError):
        fit_d2d_rule(stats, split="val", family="curvy", **kw)
    assert isinstance(fit_d2d_rule(stats, split="val", family="manhattan", **kw), D2DRule)


def test_fit_picks_the_separating_rule():
    sig, th = (0.0, 1.0), (2.0, 4.0)
    # item stats[s, t] = max component area; positives have large area at theta=4 only
    pos = np.array([[0, 20], [0, 20]], float)
    neg = np.array([[30, 0], [30, 0]], float)  # negatives only fire at the low theta
    rule = fit_d2d_rule([pos, pos, neg, neg], labels=[True, True, False, False], split="val", family="manhattan",
                        mode="single", sigmas=sig, thetas=th, areas=(10,))
    assert rule.theta == 4.0 and rule.min_area == 10


def test_stats_shape(world):
    cfg, cal, layout, refs, win = world
    score = np.random.default_rng(0).normal(0, 1, (256, 256))
    st = d2d_stats(score, win, sigmas=(0.0, 1.0), thetas=(2.0, 3.0, 4.0))
    assert st.shape == (2, 3) and np.all(st >= 0)
