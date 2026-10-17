"""Module 1: scribe-line ridge calibration + generalised Anscombe transform (AGENTS.md §5.3).

Features of the CAD clip C (1 = conductor):  [1, G_s1*C, G_s2*C, |grad(G_s1*C)|, lap(G_s1*C)].
A closed-form ridge fits the SEM image (counts, e-/px) on the 3 scribe patches. From the
weights w: dielectric level mu_diel = w0, conductor level mu_cond = w0 + w1 + w2 (deep interior
of a feature: gradient and Laplacian vanish). Residuals give the noise model
Var(I) = alpha * mu + sigma2 (alpha = gain, sigma2 = read variance in e-^2), which parameterises
the generalised Anscombe transform A(y) = (2/alpha) sqrt(alpha*y + 3/8*alpha^2 + sigma2).

Units: image and template in e-/px; sigmas in px (= nm at 1 nm/px). The noise-model fit is
robust (MAD) but residuals still contain model error, so alpha/sigma2 are upper-biased.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage as ndi
from skimage.registration import phase_cross_correlation


def features(cad: np.ndarray, s1: float, s2: float) -> np.ndarray:
    """(H, W, 5) feature stack [1, G1*C, G2*C, |grad G1*C|, lap G1*C] of a CAD clip."""
    c = np.asarray(cad, dtype=float)
    g1 = ndi.gaussian_filter(c, s1, mode="nearest")
    g2 = ndi.gaussian_filter(c, s2, mode="nearest")
    gy, gx = np.gradient(g1)
    return np.stack([np.ones_like(c), g1, g2, np.hypot(gx, gy), ndi.laplace(g1, mode="nearest")], axis=-1)


def estimate_shift(reference: np.ndarray, moving: np.ndarray, upsample: int = 20) -> tuple[float, float]:
    """Sub-pixel (dy, dx) such that ``ndi.shift(moving, (dy, dx))`` aligns with ``reference``.

    Phase correlation with ``upsample`` sub-pixel refinement (px). Used here only to register
    scribe patches; Module 3 owns alignment of in-die patches.
    """
    # Plain (un-whitened) cross-correlation of mean-removed images: phase normalisation
    # amplifies near-empty high-frequency bins and fails on smooth or noisy inputs.
    d, _, _ = phase_cross_correlation(
        reference - reference.mean(), moving - moving.mean(), upsample_factor=upsample, normalization=None
    )
    return float(d[0]), float(d[1])


def fit_noise_model(mean: np.ndarray, resid: np.ndarray, nbins: int = 12) -> tuple[float, float]:
    """Fit Var = alpha * mean + sigma2 to residuals binned by predicted mean (robust MAD variance).

    Returns (alpha, sigma2) with alpha >= 1e-6 and sigma2 >= 0 (e-/px and e-^2).
    """
    mean = np.asarray(mean).ravel()
    resid = np.asarray(resid).ravel()
    uniq = np.unique(mean)
    if uniq.size <= 4 * nbins:  # genuinely few, discrete means: one bin per value
        edges = np.append(uniq, uniq[-1] + 1.0)
    else:  # continuous means (plateaus collapse some quantile edges, which is fine)
        edges = np.unique(np.quantile(mean, np.linspace(0, 1, nbins + 1)))
    idx = np.clip(np.digitize(mean, edges[1:-1]), 0, edges.size - 2)
    m, v, w = [], [], []
    for b in range(edges.size - 1):
        sel = idx == b
        if sel.sum() < 50:
            continue
        r = resid[sel]
        m.append(mean[sel].mean())
        v.append((1.4826 * np.median(np.abs(r - np.median(r)))) ** 2)
        w.append(sel.sum())
    m, v, w = np.array(m), np.array(v), np.sqrt(np.array(w, dtype=float))
    A = np.stack([m, np.ones_like(m)], axis=1) * w[:, None]
    (alpha, s2), *_ = np.linalg.lstsq(A, v * w, rcond=None)
    return float(max(alpha, 1e-6)), float(max(s2, 0.0))


def fit_noise_model_diff(image: np.ndarray, template: np.ndarray, flat: np.ndarray) -> tuple[float, float]:
    """Fit Var = alpha*mean + sigma2 from horizontal neighbour differences in flat regions.

    d = (y[:, 1:] - y[:, :-1]) / sqrt(2) has the noise variance of one pixel but cancels smooth
    structure (charging ramp, grain interiors, template error), which a plain residual fit would
    absorb into alpha. ``flat`` marks pixels away from edges; the MAD in ``fit_noise_model`` rejects
    grain-boundary outliers. Noise is applied after the beam blur, so it is white.
    """
    ok = flat[:, 1:] & flat[:, :-1]
    d = (image[:, 1:] - image[:, :-1]) / np.sqrt(2.0)
    mean = 0.5 * (template[:, 1:] + template[:, :-1])
    return fit_noise_model(mean[ok], d[ok])


def gat(y: np.ndarray, alpha: float, sigma2: float) -> np.ndarray:
    """Generalised Anscombe transform of counts ``y`` for Var = alpha*mu + sigma2 (unit variance)."""
    return (2.0 / alpha) * np.sqrt(np.maximum(alpha * np.asarray(y, dtype=float) + 0.375 * alpha**2 + sigma2, 0.0))


def remove_ramp(image: np.ndarray, template: np.ndarray) -> np.ndarray:
    """Subtract a zero-mean 2nd-order polynomial fitted to ``image - template`` (optional step).

    The basis is mean-centred, so the image's mean level is preserved.
    """
    H, W = image.shape
    y, x = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, W), indexing="ij")
    B = np.stack([b.ravel() - b.mean() for b in (x, y, x * x, x * y, y * y)], axis=1)
    coef, *_ = np.linalg.lstsq(B, (image - template).ravel(), rcond=None)
    return image - (B @ coef).reshape(H, W)


@dataclass
class Calibration:
    """Per-layer calibration from scribe patches."""

    weights: np.ndarray                  # (5,) ridge weights on the feature stack
    sigmas: tuple[float, float]          # (s1, s2) feature scales, px
    mu_diel: float                       # predicted dielectric level, e-/px
    mu_cond: float                       # predicted conductor level, e-/px
    alpha: float                         # noise gain, Var = alpha*mu + sigma2
    sigma2: float                        # read-noise variance, e-^2
    timing_ms: float                     # wall time of calibrate(), ms
    shifts: list[tuple[float, float]] = field(default_factory=list)  # scribe registration, px

    @property
    def levels(self) -> tuple[float, float]:
        """(mu_diel, mu_cond) in the order expected by ``fields.field_calibrated``."""
        return self.mu_diel, self.mu_cond

    def predict(self, cad: np.ndarray) -> np.ndarray:
        """Predicted template T(C) in e-/px for a CAD clip."""
        return features(cad, *self.sigmas) @ self.weights

    def stabilise(self, image: np.ndarray) -> np.ndarray:
        """Generalised Anscombe transform of an image in e-/px."""
        return gat(image, self.alpha, self.sigma2)


def _shifted_features(F: np.ndarray, d: tuple[float, float]) -> np.ndarray:
    """Features of the CAD clip shifted by ``d`` px, from its unshifted feature stack ``F``.

    Bilinear shift and Gaussian blur are both convolutions and commute, so shifting the two
    blurred channels equals blurring the bilinearly shifted CAD (up to the ``nearest`` border);
    gradient and Laplacian are recomputed from the shifted G1*C. Avoids re-blurring per pass.
    """
    g1 = ndi.shift(F[..., 1], d, order=1, mode="nearest")
    g2 = ndi.shift(F[..., 2], d, order=1, mode="nearest")
    gy, gx = np.gradient(g1)
    return np.stack([F[..., 0], g1, g2, np.hypot(gx, gy), ndi.laplace(g1, mode="nearest")], axis=-1)


def _fit(Fs: list[np.ndarray], ys: list[np.ndarray], ridge: float) -> np.ndarray:
    # Closed-form ridge on standardised features, computed from 5x5 moment sums instead of the
    # (n, 5) standardised design matrix: same normal equations, ~10x less work per fit.
    G = np.zeros((5, 5)); b = np.zeros(5); n = 0
    for f, v in zip(Fs, ys):
        X = f.reshape(-1, 5)
        G += X.T @ X; b += X.T @ v.ravel(); n += X.shape[0]
    G /= n; b /= n
    mu = G[0, 1:]                                   # column means (column 0 is the constant 1)
    cov = G[1:, 1:] - np.outer(mu, mu)
    sd = np.sqrt(np.maximum(np.diag(cov), 0.0)) + 1e-12
    ZZ = np.eye(5); ZZ[1:, 1:] = cov / np.outer(sd, sd)
    Zy = np.empty(5); Zy[0] = b[0]; Zy[1:] = (b[1:] - mu * b[0]) / sd
    pen = np.diag([0.0, ridge, ridge, ridge, ridge])
    ws = np.linalg.solve(ZZ + pen, Zy)
    w = np.empty(5)
    w[1:] = ws[1:] / sd
    w[0] = ws[0] - float(w[1:] @ mu)
    return w


def noise_from_flat(Fs: list[np.ndarray], ys: list[np.ndarray], w: np.ndarray, grad_max: float = 1e-3) -> tuple[float, float]:
    """Noise model pooled over patches, from flat pixels (|grad(G1*C)| < ``grad_max``)."""
    ok_a, mean_a, d_a = [], [], []
    for F, y in zip(Fs, ys):
        T = F @ w
        flat = F[..., 3] < grad_max
        ok = flat[:, 1:] & flat[:, :-1]
        ok_a.append(ok.ravel())
        mean_a.append((0.5 * (T[:, 1:] + T[:, :-1])).ravel())
        d_a.append(((y[:, 1:] - y[:, :-1]) / np.sqrt(2.0)).ravel())
    ok, mean, d = np.concatenate(ok_a), np.concatenate(mean_a), np.concatenate(d_a)
    return fit_noise_model(mean[ok], d[ok])


def calibrate(
    patches: list[tuple[np.ndarray, np.ndarray]],
    sigmas: tuple[float, float] = (1.5, 4.0),
    ridge: float = 1e-3,
    register: bool = True,
) -> Calibration:
    """Fit Module 1 on scribe patches ``[(cad, image_e_per_px), ...]``.

    Args:
        patches: CAD clip (bool/float) and its SEM image in e-/px, same shape.
        sigmas: feature scales (s1, s2) in px.
        ridge: ridge strength on standardised non-intercept features (unitless).
        register: estimate and remove each scribe patch's sub-pixel shift (two passes).
    """
    t0 = time.perf_counter()
    s1, s2 = sigmas
    Fs = [features(c, s1, s2) for c, _ in patches]
    ys = [np.asarray(im, dtype=float) for _, im in patches]
    shifts = [(0.0, 0.0)] * len(patches)
    w = _fit(Fs, ys, ridge)
    if register:
        for _ in range(2):  # registration against the current template, then refit
            shifts = [estimate_shift(y, F @ w) for F, y in zip(Fs, ys)]
            Fa = [_shifted_features(F, d) for F, d in zip(Fs, shifts)]
            w = _fit(Fa, ys, ridge)
        Fs = Fa
    alpha, sigma2 = noise_from_flat(Fs, ys, w)
    return Calibration(
        weights=w, sigmas=(s1, s2), mu_diel=float(w[0]), mu_cond=float(w[0] + w[1] + w[2]),
        alpha=alpha, sigma2=sigma2, timing_ms=(time.perf_counter() - t0) * 1e3, shifts=shifts,
    )
