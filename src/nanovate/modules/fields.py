"""Conductor probability field construction (AGENTS.md §5.2, PLAN.md §4.1).

Constructs a continuous field P in [0, 1] from an SEM image, representing conductor
probability (1 = conductor, 0 = dielectric).

Candidates:
  (a) Percentile normalisation (reference baseline): clip to [p_lo, p_hi] -> [0, 1].
      Subject to Bug #4 (dim bridges missed, bloom fake bridges).
  (b) Calibrated level mapping: clip((I - mu_diel) / (mu_cond - mu_diel), 0, 1),
      where (mu_diel, mu_cond) are predicted by Module 1 calibration.
      Temporarily uses oracle levels from SEMCfg until M3 calibration lands (flagged in metadata).
  (c) Template-anchored residual field (candidate for M3/M4):
      clip(C_hat + (I - T_hat) / Delta, 0, 1).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from ..config import SEMCfg


def oracle_levels(sem: SEMCfg) -> tuple[float, float]:
    """Expected mean counts (e-/px) for dielectric and conductor under SEMCfg.

    Used as temporary oracle levels until Module 1 calibration lands in M3.
    """
    mu_diel = sem.dose * sem.a0
    mu_cond = sem.dose * (sem.a0 + sem.a1)
    return float(mu_diel), float(mu_cond)


def field_percentile(image: np.ndarray, p_lo: float = 2.0, p_hi: float = 98.0) -> np.ndarray:
    """Field option (a): percentile normalisation clipped to [0, 1]."""
    lo = float(np.percentile(image, p_lo))
    hi = float(np.percentile(image, p_hi))
    denom = hi - lo if abs(hi - lo) > 1e-6 else 1.0
    return np.clip((image.astype(float) - lo) / denom, 0.0, 1.0)


def field_calibrated(image: np.ndarray, mu_diel: float, mu_cond: float) -> np.ndarray:
    """Field option (b): map intensity to conductor probability via predicted levels.

    Handles contrast inversion if mu_cond < mu_diel.
    """
    denom = mu_cond - mu_diel
    if abs(denom) < 1e-6:
        denom = 1.0
    return np.clip((image.astype(float) - mu_diel) / denom, 0.0, 1.0)


def field_template(image: np.ndarray, template: np.ndarray, cad_smooth: np.ndarray, contrast: float) -> np.ndarray:
    """Field option (c): template-anchored residual, clip(C_hat + (I - T_hat) / contrast, 0, 1).

    T_hat already contains the bloom, so bloom cancels and only unexplained intensity moves the
    field; ``contrast`` = mu_cond - mu_diel (negative for inverted contrast). Injects a CAD prior, so
    faint real defects are pulled toward the CAD (judged on recall, PLAN.md §4.1).
    """
    if abs(contrast) < 1e-6:
        contrast = 1.0
    return np.clip(cad_smooth + (np.asarray(image, dtype=float) - template) / contrast, 0.0, 1.0)


def build_field(
    image: np.ndarray,
    option: str = "calibrated",
    *,
    levels: tuple[float, float] | None = None,
    sem_cfg: SEMCfg | None = None,
    p_lo: float = 2.0,
    p_hi: float = 98.0,
    pre_smooth: float = 0.0,
    template: np.ndarray | None = None,
    cad_smooth: np.ndarray | None = None,
    contrast: float | None = None,
) -> tuple[np.ndarray, dict]:
    """Construct conductor probability field P in [0, 1] and return (field, metadata).

    Args:
        image: 2D array of SEM counts (e-/px).
        option: 'calibrated' (option b) or 'percentile' (option a).
        levels: (mu_diel, mu_cond) if available. If None and option='calibrated',
                falls back to oracle_levels(sem_cfg).
        sem_cfg: optional SEMCfg used if levels is None.
        p_lo, p_hi: percentiles for option (a).
        pre_smooth: Gaussian filter standard deviation (px) applied before return.
    """
    meta: dict = {"field_option": option, "pre_smooth": pre_smooth}
    if option == "percentile":
        p = field_percentile(image, p_lo, p_hi)
        meta["levels_source"] = "percentile"
        meta["p_lo"] = p_lo
        meta["p_hi"] = p_hi
    elif option == "calibrated":
        if levels is not None:
            mu_diel, mu_cond = levels
            meta["levels_source"] = "calibrated"
        elif sem_cfg is not None:
            mu_diel, mu_cond = oracle_levels(sem_cfg)
            meta["levels_source"] = "oracle"
        else:
            raise ValueError("option='calibrated' requires either 'levels' or 'sem_cfg'")
        p = field_calibrated(image, mu_diel, mu_cond)
        meta["mu_diel"] = mu_diel
        meta["mu_cond"] = mu_cond
    elif option == "template":
        if template is None or cad_smooth is None or contrast is None:
            raise ValueError("option='template' requires template, cad_smooth and contrast")
        p = field_template(image, template, cad_smooth, contrast)
        meta["levels_source"] = "template"
    else:
        raise ValueError(f"unknown field option {option!r}; expected 'calibrated', 'percentile' or 'template'")

    if pre_smooth > 0:
        p = ndi.gaussian_filter(p, sigma=pre_smooth)
        p = np.clip(p, 0.0, 1.0)

    return p, meta
