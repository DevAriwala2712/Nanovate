"""Label break-even helpers (Part 1 of docs/superpowers/specs/2026-10-18-next-steps-design.md).

How many labelled patches of a new layout family does the Manhattan-trained CNN need to match the
training-free verifier? The CNN is fine-tuned on k labelled patches of the family's *train* split; 25%
of those k (rounded up) choose its probability threshold, so every label a day-0 user would need is
counted in k. Only the baseline is adapted; the verifier and the test split are never touched.
Synthetic data.
"""
from __future__ import annotations

import math

import numpy as np

K_GRID = (0, 12, 24, 48, 96, 192, 384, 768)
SEEDS = (0, 1, 2)
DEFAULT_THRESHOLD = 0.39   # M6 Manhattan-val threshold, used for k = 0 and the fixed-threshold curve
FT_STEPS = 300             # optimiser steps per fine-tune, fixed before any test number
FT_LR = 5e-4


def breakeven_subsets(n_pool: int, k: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """(fine-tune indices, threshold indices) of the first k entries of a seeded pool permutation.

    The permutation depends on ``seed`` only, so subsets are nested in k; the last ceil(k/4) of the
    k picked indices are the threshold set. k = 0 gives two empty arrays.
    """
    if k > n_pool:
        raise ValueError(f"k = {k} exceeds the pool of {n_pool} patches")
    picked = np.random.default_rng((seed, 7231)).permutation(n_pool)[:k]
    n_thr = math.ceil(k / 4)
    return picked[: k - n_thr], picked[k - n_thr:]


def pick_threshold_labelled(probs: np.ndarray, labels: np.ndarray, default: float = DEFAULT_THRESHOLD) -> float:
    """Max-F1 probability threshold on a small labelled set (ties: lowest); ``default`` without positives."""
    y = np.asarray(labels, bool)
    probs = np.asarray(probs, float)
    if y.size == 0 or not y.any():
        return float(default)
    best_t, best_f1 = float(default), -1.0
    for t in np.unique(probs):
        pred = probs >= t
        tp = float((pred & y).sum())
        f1 = 2 * tp / max(2 * tp + float((pred & ~y).sum()) + float((~pred & y).sum()), 1.0)
        if f1 > best_f1 + 1e-12:
            best_t, best_f1 = float(t), f1
    return best_t


def finetune_epochs(n_ft: int, steps: int = FT_STEPS, batch: int = 32) -> int:
    """Epochs giving about ``steps`` optimiser steps on ``n_ft`` patches (0 when there is nothing to fit)."""
    if n_ft <= 0:
        return 0
    return math.ceil(steps / math.ceil(n_ft / batch))
