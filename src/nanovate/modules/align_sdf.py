"""Module 3: sub-pixel alignment to the calibrated template + CAD SDF tension energy (AGENTS.md §5.4).

Alignment: cross-correlation (``calibration.estimate_shift``, 20x sub-pixel upsampling) of the
patch against the Module-1 predicted template T(C), not the binary CAD (its appearance differs
from SEM); the shift is applied with a cubic ``ndi.shift``. Convention: the returned d satisfies
``ndi.shift(image, d) ~ template``; a testbed image rendered with shift s gives d ~ -s.

SDF: phi from the CAD by Euclidean distance transforms, in px (= nm at 1 nm/px), positive in the
dielectric and negative inside conductors. Energy: E = |S| * exp(gamma * min(|phi|, cap)) with the
residual S in units of the predicted conductor-dielectric contrast. No Langevin step.
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from ..topo_utils import Window
from .calibration import Calibration, estimate_shift


def align_to_template(
    image: np.ndarray, template: np.ndarray, upsample: int = 20, passes: int = 3
) -> tuple[np.ndarray, tuple[float, float]]:
    """Register ``image`` to ``template``. Returns (aligned image, (dy, dx) in px).

    The correlation is circular, so content entering from outside the frame biases the peak in
    proportion to the shift. Re-estimating on the progressively aligned image (``passes``) removes
    that bias; the shifts are accumulated and applied once to the original image.
    """
    img = np.asarray(image, dtype=float)
    total = np.zeros(2)
    cur = img
    for _ in range(passes):
        total += estimate_shift(reference=template, moving=cur, upsample=upsample)
        cur = ndi.shift(img, tuple(total), order=3, mode="nearest")
    return cur, (float(total[0]), float(total[1]))


def cad_sdf(cad: np.ndarray) -> np.ndarray:
    """Signed distance (px = nm) to the CAD boundary: > 0 in dielectric, < 0 in conductor."""
    c = np.asarray(cad, dtype=bool)
    return ndi.distance_transform_edt(~c) - ndi.distance_transform_edt(c)


def tension_energy(residual: np.ndarray, phi: np.ndarray, gamma: float, cap: float) -> np.ndarray:
    """E = |S| * exp(gamma * min(|phi|, cap)); ``cap`` (px) keeps the weight finite on sparse layouts."""
    return np.abs(residual) * np.exp(gamma * np.minimum(np.abs(phi), cap))


def max_tension_energy(
    image: np.ndarray,
    cad: np.ndarray,
    cal: Calibration,
    win: Window,
    gamma: float = 0.2,
    cap: float = 10.0,
    smooth: float = 1.0,
) -> float:
    """Max tension energy over the central window for one (singleton) patch.

    S = G_smooth * (aligned image - T(C)) / |mu_cond - mu_diel| (contrast units). ``gamma`` is fixed
    a priori so the centre of a 14 px feature or gap (|phi| ~ 7) is weighted ~4x an edge pixel;
    it is not tuned on any split.
    """
    template = cal.predict(cad)
    aligned, _ = align_to_template(image, template)
    contrast = abs(cal.mu_cond - cal.mu_diel)
    s = ndi.gaussian_filter(aligned - template, smooth) / contrast if smooth > 0 else (aligned - template) / contrast
    e = tension_energy(s, cad_sdf(cad), gamma, cap)
    return float(win.crop(e).max())
