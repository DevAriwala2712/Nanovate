"""Shared experiment infrastructure: config presets, percentile-bootstrap CIs, F1, result writer.

Every experiment reports mean + 95% percentile bootstrap CI (1,000 resamples; 200 under --quick) with
n beside each number, writes CSV + PNG + config.json under results/expN/, and labels results
"synthetic data" (AGENTS.md §6).
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class ExpCfg:
    """Sample budgets and sweeps for all experiments (saved to every config.json)."""

    quick: bool
    seed: int = 0
    layouts: int = 12                 # test layouts per family
    dies: int = 2                     # dies per layout (each with its own LER / noise / shift / defect)
    n_boot: int = 1000
    doses: tuple = (5, 10, 20, 35, 50, 100, 200)            # Exp 2
    exp1_sigmas: tuple = (0.5, 1.0, 1.5, 2.0, 2.5)          # nm
    exp1_grains: tuple = (0.0, 0.10, 0.20, 0.30)            # conductor grain contrast
    nominal_dose: float = 50.0
    exp4_shift: float = 8.0           # px, shifts ~ U(-8, 8)
    exp4_warp: float = 1.0            # px, smooth warp amplitude
    exp5_dies: int = 6                # dies per layout for the near-miss experiment (small classes need more n)
    exp6_calib_runs: int = 100
    exp6_patches: int = 1000
    val_layouts: int = 8              # Manhattan val layouts for baseline rule fitting / config selection
    val_dies: int = 2

    @classmethod
    def full(cls) -> "ExpCfg":
        return cls(quick=False)

    @classmethod
    def quick(cls) -> "ExpCfg":  # noqa: F811 (classmethod shares the field name on purpose)
        return cls(quick=True, layouts=3, dies=1, n_boot=200, doses=(20, 50, 200), exp1_sigmas=(0.5, 1.5, 2.5),
                   exp1_grains=(0.0, 0.30), exp5_dies=2, exp6_calib_runs=10, exp6_patches=24, val_layouts=3, val_dies=1)

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ExpCfg":
        d = dict(d)
        for k in ("doses", "exp1_sigmas", "exp1_grains"):
            d[k] = tuple(d[k])
        return cls(**d)


def f1_score(pred: np.ndarray, y: np.ndarray) -> float:
    """F1 of boolean predictions against boolean labels (0 when there are no positives at all)."""
    pred, y = np.asarray(pred, bool), np.asarray(y, bool)
    tp = float((pred & y).sum())
    denom = 2 * tp + float((pred & ~y).sum()) + float((~pred & y).sum())
    return 2 * tp / denom if denom > 0 else 0.0


def bootstrap_stat(arrays: list, stat: Callable, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """(statistic, 2.5%, 97.5%) by percentile bootstrap over rows; ``arrays`` are resampled jointly."""
    arrays = [np.asarray(a) for a in arrays]
    n = len(arrays[0])
    if n == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, n, size=(n_boot, n))
    vals = np.array([stat(*[a[i] for a in arrays]) for i in idx])
    return float(stat(*arrays)), float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def bootstrap_diff(a: list, b: list, stat: Callable, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """(stat(a) - stat(b), 2.5%, 97.5%): independent percentile bootstrap of two sample sets (e.g. families)."""
    a, b = [np.asarray(x) for x in a], [np.asarray(x) for x in b]
    ra, rb = np.random.default_rng([seed, 1]), np.random.default_rng([seed, 2])
    ia = ra.integers(0, len(a[0]), size=(n_boot, len(a[0])))
    ib = rb.integers(0, len(b[0]), size=(n_boot, len(b[0])))
    d = np.array([stat(*[x[i] for x in a]) - stat(*[x[j] for x in b]) for i, j in zip(ia, ib)])
    return float(stat(*a) - stat(*b)), float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))


def bootstrap_ci(x: np.ndarray, n_boot: int = 1000, seed: int = 0) -> tuple[float, float, float]:
    """(mean, 2.5%, 97.5%) percentile bootstrap CI of the mean of ``x``."""
    x = np.asarray(x, dtype=float)
    if len(x) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = x[rng.integers(0, len(x), size=(n_boot, len(x)))].mean(axis=1)
    return float(x.mean()), float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def write_results(out_dir, name: str, df: pd.DataFrame, config: dict) -> Path:
    """Write ``name``.csv and config.json into ``out_dir`` (created). Returns the CSV path."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.csv"
    df.to_csv(path, index=False)
    (out / "config.json").write_text(json.dumps(config, indent=2, sort_keys=True, default=str))
    return path
