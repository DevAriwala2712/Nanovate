"""Module 2: same-layout patch stack, rank-2 randomized SVD, per-pixel MAD threshold (AGENTS.md §5.5).

Every die is first aligned to the Module-1 template T(C) (bug #3: a misaligned stack is not
low-rank), then the K aligned dies form the matrix D (K x pixels). L = rank-2 randomized SVD of D
(shared appearance incl. the dominant per-die variation, e.g. charging ramp); the residual R = D - L
is thresholded per pixel at k * 1.4826 * MAD_K(R) (the per-pixel noise floor across dies, which
also tracks the heteroscedastic Poisson noise without a variance-stabilising transform) to give the
sparse part S. The field passed to Module 4 is the die's denoised estimate L_i + S_i (not S alone).
Singletons (no K peers) fall back to median + Gaussian denoise and are flagged ``singleton=True``.

Soft threshold as specified, plus a 1 px residual smoothing before the threshold (a measured
deviation: without it isolated noise pixels flood Module 4 at low dose); ``shrink`` and
``resid_smooth`` are ablation knobs (PROGRESS.md). Units: e-/px. Deterministic given ``seed``.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi

from .align_sdf import align_to_template
from .calibration import Calibration


def randomized_svd(
    M: np.ndarray, rank: int = 2, oversample: int = 2, n_iter: int = 2, seed: int = 0
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Top-``rank`` SVD of M by seeded randomized subspace iteration. Returns (U, s, Vt)."""
    rng = np.random.default_rng(seed)
    n_cols = M.shape[1]
    k = min(rank + oversample, min(M.shape))
    Q = np.linalg.qr(M @ rng.standard_normal((n_cols, k)))[0]
    for _ in range(n_iter):  # power iterations with re-orthonormalisation
        Z = np.linalg.qr(M.T @ Q)[0]
        Q = np.linalg.qr(M @ Z)[0]
    Ub, s, Vt = np.linalg.svd(Q.T @ M, full_matrices=False)
    return (Q @ Ub)[:, :rank], s[:rank], Vt[:rank]


def soft_threshold(R: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """sign(R) * max(|R| - tau, 0), tau broadcast over the first axis."""
    return np.sign(R) * np.maximum(np.abs(R) - tau, 0.0)


def hard_threshold(R: np.ndarray, tau: np.ndarray) -> np.ndarray:
    """R where |R| > tau, else 0 (keeps the amplitude of detected pixels: no shrinkage)."""
    return np.where(np.abs(R) > tau, R, 0.0)


def _decompose(
    aligned: np.ndarray, rank: int, k: float, seed: int, shrink: str, resid_smooth: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """(L, S, sigma, residual energy) of an aligned (K, H, W) stack.

    ``resid_smooth`` > 0 Gaussian-smooths each die's residual (px) before the per-pixel MAD
    threshold: a matched filter that lifts coherent defects above the noise floor while isolated
    noise pixels stay below it. ``shrink``: 'soft' (spec), 'hard', or 'support' (detect on the
    smoothed residual, keep the raw residual amplitude on that support).
    """
    K = aligned.shape[0]
    D = aligned.reshape(K, -1)
    U, s, Vt = randomized_svd(D, rank=rank, seed=seed)
    L = (U * s) @ Vt
    R = D - L
    energy = float(np.linalg.norm(R) / np.linalg.norm(D))
    R_raw = R
    if resid_smooth > 0:
        R = ndi.gaussian_filter(R.reshape(aligned.shape), sigma=(0, resid_smooth, resid_smooth), mode="nearest").reshape(K, -1)
    sigma = 1.4826 * np.median(np.abs(R - np.median(R, axis=0)), axis=0)
    if shrink == "support":  # detect on the (smoothed) residual, keep the raw residual amplitude there
        S = np.where(np.abs(R) > k * sigma, R_raw, 0.0)
    else:
        S = {"soft": soft_threshold, "hard": hard_threshold}[shrink](R, k * sigma)
    return L.reshape(aligned.shape), S.reshape(aligned.shape), sigma.reshape(aligned.shape[1:]), energy


def sparse_part(
    aligned: np.ndarray, rank: int = 2, k: float = 2.0, seed: int = 0, shrink: str = "soft", resid_smooth: float = 0.0
) -> np.ndarray:
    """Sparse part S of an already-aligned (K, H, W) stack."""
    return _decompose(aligned, rank, k, seed, shrink, resid_smooth)[1]


def denoise_singleton(image: np.ndarray, median: int = 3, sigma: float = 1.0) -> np.ndarray:
    """Fallback for a patch without same-layout peers: median filter, then Gaussian (px)."""
    return ndi.gaussian_filter(ndi.median_filter(np.asarray(image, dtype=float), size=median, mode="nearest"), sigma)


@dataclass
class StackResult:
    """Per-stack output. Arrays are (K, H, W) in the CAD frame (dies aligned to T(C))."""

    L: np.ndarray               # low-rank part
    S: np.ndarray               # sparse (thresholded) residual
    field: np.ndarray           # L + S, the per-die denoised estimate passed to Module 4
    aligned: np.ndarray         # aligned input images
    shifts: list                # per-die (dy, dx) applied, px
    sigma: np.ndarray           # per-pixel MAD noise floor (H, W), e-/px
    residual_energy: float      # ||D - L||_F / ||D||_F
    rank: int
    K: int
    seed: int
    singleton: bool             # True: median/Gaussian fallback, no stack statistics


def denoise_stack(
    images: list[np.ndarray],
    cad: np.ndarray,
    cal: Calibration,
    rank: int = 2,
    k: float = 2.0,
    seed: int = 0,
    align: bool = True,
    shrink: str = "soft",
    resid_smooth: float = 1.0,
) -> StackResult:
    """Module 2 on K same-layout dies (images in e-/px, same tile/CAD).

    Args:
        images: K patches of the same layout (each with its own shift, noise, nuisance, defects).
        cad: CAD clip (shared by the stack).
        cal: Module-1 calibration (supplies the template for alignment).
        rank: rank of L (2).
        k: threshold multiplier on the per-pixel MAD sigma (fixed a priori, not tuned per sample).
        seed: randomized-SVD sketch seed.
        align: align each die to T(C) first; ``False`` only to measure the misalignment penalty.
        shrink: 'soft' (spec), 'hard' or 'support' thresholding of the residual.
        resid_smooth: Gaussian sigma (px) applied to the residual before thresholding. Default 1.0
            deviates from the spec's pure per-pixel threshold (0): at 0 the false-killer rate is 1.00 at
            dose 20 (PROGRESS.md, M5).
    """
    K = len(images)
    if K == 1:  # singleton: flagged fallback
        out = denoise_singleton(images[0])[None]
        return StackResult(out, np.zeros_like(out), out, np.asarray(images[0], dtype=float)[None], [(0.0, 0.0)],
                           np.full(out.shape[1:], np.nan), float("nan"), 0, 1, seed, True)
    template = cal.predict(cad)
    if align:
        pairs = [align_to_template(im, template) for im in images]
        aligned = np.stack([p[0] for p in pairs])
        shifts = [p[1] for p in pairs]
    else:
        aligned = np.stack([np.asarray(im, dtype=float) for im in images])
        shifts = [(0.0, 0.0)] * K
    L, S, sigma, energy = _decompose(aligned, rank, k, seed, shrink, resid_smooth)
    return StackResult(L, S, L + S, aligned, shifts, sigma, energy, rank, K, seed, False)
