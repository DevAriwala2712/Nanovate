"""Part 1 (day-0 value case): break-even helpers, Colab training script smoke test, local evaluation."""
import json
import subprocess
import sys

import numpy as np
import pytest

from nanovate.baselines.breakeven import breakeven_subsets, finetune_epochs, pick_threshold_labelled


def test_subsets_sizes_disjoint_nested_and_seeded():
    ft, th = breakeven_subsets(768, 12, seed=0)
    assert len(ft) == 9 and len(th) == 3 and not set(ft) & set(th)
    ft2, th2 = breakeven_subsets(768, 12, seed=0)
    assert np.array_equal(ft, ft2) and np.array_equal(th, th2)
    assert not np.array_equal(breakeven_subsets(768, 12, seed=1)[0], ft)
    bft, bth = breakeven_subsets(768, 24, seed=0)
    assert set(np.r_[ft, th]) <= set(np.r_[bft, bth])          # nested in k for a seed
    assert all(len(a) == 0 for a in breakeven_subsets(768, 0, seed=0))
    with pytest.raises(ValueError):
        breakeven_subsets(10, 12, seed=0)


def test_threshold_fallbacks():
    assert pick_threshold_labelled(np.array([]), np.array([], bool)) == 0.39
    assert pick_threshold_labelled(np.array([0.2, 0.9]), np.array([False, False])) == 0.39
    assert pick_threshold_labelled(np.array([0.2, 0.6, 0.9]), np.array([False, True, True])) == 0.6


def test_finetune_epochs():
    assert finetune_epochs(0) == 0 and finetune_epochs(9) == 300 and finetune_epochs(576) == 17
