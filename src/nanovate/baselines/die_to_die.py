"""Baseline: die-to-die difference (single) and double-difference, with a val-only threshold fit.

Fairness choices (AGENTS.md §5.6): target and reference dies are aligned to the same Module-1
template with the same Module-3 aligner (alignment is not a confound) and stabilised with the same
Module-1 generalised Anscombe transform, so one threshold in z-units serves every dose. A patch is
flagged if any connected region of |smoothed difference| > theta inside the central window has
at least ``min_area`` px. (sigma, theta, min_area) are selected on Manhattan *val* only
(``fit_d2d_rule`` refuses anything else) and then frozen.

Single: d = (z_t - z_r) / sqrt(2).  Double (two neighbours, as in the inspection-tool literature):
keep a pixel only where d1 and d2 have the same sign, with magnitude min(|d1|, |d2|), so a defect
in one reference cancels. Units: z = unit-variance noise after the GAT; areas in px.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from ..modules.align_sdf import align_to_template
from ..modules.calibration import Calibration
from ..topo_utils import Window


@dataclass(frozen=True)
class D2DRule:
    """Frozen die-to-die decision rule."""

    mode: str          # 'single' | 'double'
    sigma: float       # Gaussian smoothing of the difference map, px
    theta: float       # threshold on the smoothed map, in white-noise z-units
    min_area: int      # minimum connected area, px


def d2d_score(
    target: np.ndarray, refs: list[np.ndarray], cad: np.ndarray, cal: Calibration, mode: str = "single"
) -> np.ndarray:
    """Signed difference map in z-units (CAD frame). ``refs``: 1 (single) or 2 (double) reference dies."""
    template = cal.predict(cad)
    z = lambda im: cal.stabilise(align_to_template(im, template)[0])  # noqa: E731
    zt = z(target)
    d = [(zt - z(r)) / np.sqrt(2.0) for r in refs]
    if mode == "single":
        return d[0]
    if mode == "double":
        if len(d) != 2:
            raise ValueError("double-difference needs exactly two reference dies")
        same = np.sign(d[0]) == np.sign(d[1])
        return np.where(same, np.sign(d[0]) * np.minimum(np.abs(d[0]), np.abs(d[1])), 0.0)
    raise ValueError(f"unknown mode {mode!r}")


def _smoothed_abs(score: np.ndarray, sigma: float) -> np.ndarray:
    """|G_sigma * score| rescaled so white unit-variance noise stays unit variance (z-units)."""
    if sigma <= 0:
        return np.abs(score)
    return np.abs(ndi.gaussian_filter(score, sigma)) * (2.0 * np.sqrt(np.pi) * sigma)


def _max_area(mask: np.ndarray) -> int:
    lab, n = ndi.label(mask)
    return int(np.bincount(lab.ravel())[1:].max()) if n else 0


def d2d_stats(score: np.ndarray, win: Window, sigmas: tuple, thetas: tuple) -> np.ndarray:
    """(len(sigmas), len(thetas)) max connected-region area inside the window, for rule selection."""
    out = np.zeros((len(sigmas), len(thetas)))
    for i, s in enumerate(sigmas):
        a = win.crop(_smoothed_abs(score, s))
        for j, t in enumerate(thetas):
            out[i, j] = _max_area(a > t)
    return out


def d2d_flag(score: np.ndarray, rule: D2DRule, win: Window) -> tuple[bool, int]:
    """(flagged, max region area) of a score map under a frozen rule."""
    area = _max_area(win.crop(_smoothed_abs(score, rule.sigma)) > rule.theta)
    return area >= rule.min_area, area


def fit_d2d_rule(
    stats: list[np.ndarray],
    labels: list[bool],
    split: str,
    family: str,
    mode: str,
    sigmas: tuple,
    thetas: tuple,
    areas: tuple,
) -> D2DRule:
    """Pick (sigma, theta, min_area) maximising killer F1 over ``stats`` (from ``d2d_stats``).

    Refuses anything but Manhattan val (AGENTS.md §5.2/§5.6). Ties keep the first grid point
    (small sigma, low theta, small area), so the choice is deterministic.
    """
    if split != "val" or family != "manhattan":
        raise ValueError("D2D thresholds may only be fitted on Manhattan val")
    S = np.stack(stats)                      # (n, sigmas, thetas)
    y = np.asarray(labels, bool)
    best, best_f1 = None, -1.0
    for i, s in enumerate(sigmas):
        for j, t in enumerate(thetas):
            for a in areas:
                pred = S[:, i, j] >= a
                tp = float((pred & y).sum())
                f1 = 2 * tp / max(2 * tp + float((pred & ~y).sum()) + float((~pred & y).sum()), 1.0)
                if f1 > best_f1 + 1e-12:
                    best, best_f1 = D2DRule(mode, float(s), float(t), int(a)), f1
    assert best is not None
    return best
