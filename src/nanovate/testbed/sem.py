"""SEM forward model (SEM mode only): wafer W -> noisy image in e- counts.

    Y      = a0 + a1 * W * (1 + grain) + a2 * s * tanh(|grad(G_edge * W)| / s)   (yield)
    I_bar  = G_beam(sig_r, sig_c) * Y  shifted by (dy, dx), cropped to the tile, + charging ramp
    I      = Poisson(dose * I_bar) + N(0, read_noise^2)                          (e-/px)

The beam is anisotropic and the bloom saturates with its own edge scale, so the model is
deliberately outside the span of Module 1's 5 linear features (PLAN.md, risk R1).
Shift convention: the returned image satisfies image(x) ~ I_unshifted(x - shift).
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from ..config import SEMCfg


def sem_yield(W: np.ndarray, grain: np.ndarray | None, cfg: SEMCfg) -> np.ndarray:
    """Noise-free secondary-electron yield map (before beam blur), same shape as W."""
    w = W.astype(np.float32)
    g = ndi.gaussian_filter(w, cfg.sigma_edge)
    gy, gx = np.gradient(g)
    bloom = np.hypot(gx, gy)
    bloom = cfg.bloom_sat * np.tanh(bloom / cfg.bloom_sat)
    cond = w if grain is None else w * (1.0 + grain)
    return cfg.a0 + cfg.a1 * cond + cfg.a2 * bloom


def charging_ramp(shape: tuple[int, int], amp: float, rng: np.random.Generator) -> np.ndarray:
    """Random 2nd-order polynomial brightness ramp with max |value| = amp * U(0, 1)."""
    H, Wd = shape
    y, x = np.meshgrid(np.linspace(-1, 1, H), np.linspace(-1, 1, Wd), indexing="ij")
    basis = np.stack([x, y, x * x, x * y, y * y])
    c = rng.normal(0, 1, 5)
    ramp = np.tensordot(c, basis, 1)
    ramp -= ramp.mean()
    m = np.abs(ramp).max()
    return (ramp / m * amp * rng.uniform(0, 1)) if m > 0 else ramp


def warp_field(shape: tuple[int, int], amp: float, rng: np.random.Generator, scale: float = 50.0) -> tuple[np.ndarray, np.ndarray]:
    """Smooth random displacement field (dy, dx) in px with max |displacement| = ``amp`` (0 -> zeros).

    White noise smoothed at ``scale`` px, so the warp varies slowly (a few px of displacement change
    over hundreds of px), then jointly normalised so the largest component magnitude is ``amp``.
    """
    if amp <= 0:
        return np.zeros(shape), np.zeros(shape)
    f = [ndi.gaussian_filter(rng.normal(size=shape), scale, mode="reflect") for _ in range(2)]
    m = max(np.abs(f[0]).max(), np.abs(f[1]).max())
    return f[0] / m * amp, f[1] / m * amp


def expected_image(
    W_canvas: np.ndarray,
    grain_canvas: np.ndarray | None,
    cfg: SEMCfg,
    pad: int,
    tile: int,
    shift: tuple[float, float],
    ramp: np.ndarray | None = None,
    warp: tuple[np.ndarray, np.ndarray] | None = None,
) -> np.ndarray:
    """Noise-free mean image on the tile, in yield units (multiply by dose for e-/px).

    ``warp`` = (dy, dx) displacement fields on the canvas (px): image(x) = y(x + d(x)) after the shift.
    """
    y = sem_yield(W_canvas, grain_canvas, cfg)
    y = ndi.gaussian_filter(y, sigma=cfg.sigma_beam)
    if shift != (0.0, 0.0):
        y = ndi.shift(y, shift, order=3, mode="nearest")
    if warp is not None:
        yy, xx = np.meshgrid(np.arange(y.shape[0]), np.arange(y.shape[1]), indexing="ij")
        y = ndi.map_coordinates(y, [yy + warp[0], xx + warp[1]], order=3, mode="nearest")
    out = y[pad:pad + tile, pad:pad + tile]
    if ramp is not None:
        out = out + ramp
    return np.clip(out, 0.0, None)


def acquire(I_bar: np.ndarray, cfg: SEMCfg, rng: np.random.Generator) -> np.ndarray:
    """Shot + read noise. Returns float32 counts (e-/px)."""
    lam = cfg.dose * I_bar
    img = rng.poisson(lam).astype(np.float32)
    if cfg.read_noise > 0:
        img += rng.normal(0, cfg.read_noise, img.shape).astype(np.float32)
    return img
