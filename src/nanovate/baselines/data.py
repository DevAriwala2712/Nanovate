"""Shared data recipes for the baselines (Manhattan only; synthetic)."""
from __future__ import annotations

import dataclasses

import numpy as np

from ..config import TestbedCfg
from ..testbed.dataset import KINDS, make_sample

TRAIN_DOSES = (5, 10, 20, 35, 50, 100, 200)


def with_dose(cfg: TestbedCfg, dose: float) -> TestbedCfg:
    """Copy of ``cfg`` with the SEM dose (e-/px) replaced."""
    return dataclasses.replace(cfg, sem=dataclasses.replace(cfg.sem, dose=float(dose)))


def make_cnn_train_set(
    n_layouts: int, dies: int, seed: int = 0, doses: tuple = TRAIN_DOSES, cfg: TestbedCfg | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Manhattan *train* split: every kind on ``n_layouts`` x ``dies``, a random dose per sample.

    Returns (images e-/px, cads bool, killer labels, doses). Deterministic in ``seed``.
    """
    base = cfg or TestbedCfg()
    rng = np.random.default_rng(seed)
    imgs, cads, ys, ds = [], [], [], []
    for lay in range(n_layouts):
        for die in range(dies):
            for kind in KINDS:
                dose = float(rng.choice(doses))
                s = make_sample("manhattan", "train", lay, die, kind, with_dose(base, dose), seed)
                imgs.append(s.image); cads.append(s.cad); ys.append(s.killer); ds.append(dose)
    return np.stack(imgs), np.stack(cads), np.array(ys), np.array(ds)
