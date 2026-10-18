"""Re-render the same physical wafer at s-times finer pixel sampling (v3 Phase 1 diagnostic).

The ground-truth wafer W (and its grain map) is generated exactly as for the 1 nm/px testbed, then each
1 nm pixel is split into s x s sub-pixels (nearest-neighbour upsampling). The physical geometry, the
defect and its topology are therefore *identical* at every sampling; only the pixel pitch changes
(1/s nm/px). The SEM forward model is held fixed in nm: beam blur, bloom smoothing scale and shift are
multiplied by s in px, and the bloom term uses the physical gradient (per nm, not per px). With s = 1
every function here reproduces the testbed's own render bit for bit (pinned by a test).

Optional ``blur_scale`` multiplies the beam and bloom widths (in nm): a supplementary "sharper optics"
arm, not part of the sampling sweep. Units: lengths nm unless stated, images e-/px.
"""
from __future__ import annotations

import dataclasses

import numpy as np
from scipy import ndimage as ndi

from ..config import SEMCfg, TestbedCfg
from ..rng import child_rng
from .dataset import WaferDie, get_layout, make_wafer
from .layouts import Layout, rasterize
from .perturb import apply_ler, grain_field
from .scribe import _features
from .sem import acquire, charging_ramp


def upsample(a: np.ndarray, s: int) -> np.ndarray:
    """Split every pixel into s x s sub-pixels (preserves 4-/8-connectivity of binary masks)."""
    return a if s == 1 else np.repeat(np.repeat(a, s, 0), s, 1)


def scaled_sem(sem: SEMCfg, blur_scale: float = 1.0) -> SEMCfg:
    """SEMCfg with beam and bloom widths multiplied by ``blur_scale`` (nm). Dose stays per px."""
    if blur_scale == 1.0:
        return sem
    return dataclasses.replace(sem, sigma_edge=sem.sigma_edge * blur_scale,
                               sigma_beam=tuple(v * blur_scale for v in sem.sigma_beam))


def sem_yield_px(W: np.ndarray, grain: np.ndarray | None, cfg: SEMCfg, s: int) -> np.ndarray:
    """``sem.sem_yield`` at s sub-pixels per nm: Gaussian widths in px = s * nm, gradient per nm."""
    w = W.astype(np.float32)
    g = ndi.gaussian_filter(w, cfg.sigma_edge * s if s != 1 else cfg.sigma_edge)
    gy, gx = np.gradient(g)
    bloom = np.hypot(gx, gy) if s == 1 else np.hypot(gx, gy) * s
    bloom = cfg.bloom_sat * np.tanh(bloom / cfg.bloom_sat)
    cond = w if grain is None else w * (1.0 + grain)
    return cfg.a0 + cfg.a1 * cond + cfg.a2 * bloom


def expected_image_px(W_canvas, grain_canvas, cfg: SEMCfg, pad: int, tile: int, shift_nm, ramp, s: int) -> np.ndarray:
    """``sem.expected_image`` (no warp) on the s-times finer grid; ``pad``/``tile`` in 1 nm px."""
    y = sem_yield_px(W_canvas, grain_canvas, cfg, s)
    sig = cfg.sigma_beam if s == 1 else tuple(v * s for v in cfg.sigma_beam)
    y = ndi.gaussian_filter(y, sigma=sig)
    if tuple(shift_nm) != (0.0, 0.0):
        y = ndi.shift(y, tuple(v * s for v in shift_nm), order=3, mode="nearest")
    out = y[pad * s:(pad + tile) * s, pad * s:(pad + tile) * s]
    if ramp is not None:
        out = out + ramp
    return np.clip(out, 0.0, None)


def _acquire_expected(wafer_W, wafer_grain, layout_pad, layout_tile, sem: SEMCfg, rng, s: int):
    """Draw shift and ramp from ``rng`` in the testbed's order; return (expected yield, shift nm)."""
    shift = tuple(float(v) for v in rng.uniform(-sem.max_shift, sem.max_shift, 2))
    ramp = charging_ramp((layout_tile * s, layout_tile * s), sem.charge_amp, rng)
    I_bar = expected_image_px(upsample(wafer_W, s), upsample(wafer_grain, s), sem, layout_pad, layout_tile, shift, ramp, s)
    return I_bar, shift


@dataclasses.dataclass
class FineSample:
    """One wafer rendered noise-free at s sub-pixels per nm (tile coords, s * 256 px)."""

    kind: str
    label: str
    killer: bool
    topo: dict
    cad: np.ndarray        # bool (s*tile, s*tile)
    W: np.ndarray          # bool
    mask: np.ndarray       # bool, defect pixels
    expected: np.ndarray   # float32 noise-free counts, e-/px
    shift_nm: tuple[float, float]
    params: dict
    s: int


def make_fine_sample(family: str, split: str, layout_id: int, die: int, kind: str, cfg: TestbedCfg, seed: int,
                     s: int, blur_scale: float = 1.0, layout: Layout | None = None, acq_tag: str = "") -> FineSample:
    """Same keys and random streams as ``dataset.make_sample``; noise-free image at s sub-pixels per nm."""
    from .labels import LABEL_OF_KIND

    layout = layout or get_layout(family, split, layout_id, cfg, seed)
    wafer = make_wafer(layout, kind, cfg, child_rng(seed, "geom", family, split, layout_id, die, kind))
    sem = scaled_sem(cfg.sem, blur_scale)
    I_bar, shift = _acquire_expected(wafer.W_canvas, wafer.grain_canvas, layout.pad, layout.tile, sem,
                                     child_rng(seed, "acq", family, split, layout_id, die, acq_tag), s)
    p, t = layout.pad, layout.tile
    crop = (slice(p, p + t), slice(p, p + t))
    return FineSample(kind, LABEL_OF_KIND[kind], LABEL_OF_KIND[kind] == "KILLER", wafer.topo,
                      upsample(layout.cad, s), upsample(wafer.W_canvas[crop], s), upsample(wafer.mask_canvas[crop], s),
                      (sem.dose * I_bar).astype(np.float32), shift, wafer.params, s)


def make_fine_scribe_patches(cfg: TestbedCfg, seed: int, s: int, blur_scale: float = 1.0, layer: str = "layer0"):
    """``scribe.make_scribe_patches`` at s sub-pixels per nm (noisy, dose per px unchanged)."""
    lc = cfg.layout
    N = lc.tile + 2 * lc.pad
    sem = scaled_sem(cfg.sem, blur_scale)
    rng = child_rng(seed, "scribe", layer)
    out = []
    for i, polys in enumerate(_features(N, rng)):
        cad = rasterize(polys, N)
        g = child_rng(seed, "scribe_geom", layer, i)
        W = apply_ler(polys, N, cfg.nuisance, g)
        grain = grain_field(N, cfg.nuisance, g)
        r = child_rng(seed, "scribe_acq", layer, i)
        I_bar, _ = _acquire_expected(W, grain, lc.pad, lc.tile, sem, r, s)
        img = acquire(I_bar, sem, r)  # noise from the same stream after shift + ramp, as in the testbed
        out.append((upsample(cad[lc.pad:lc.pad + lc.tile, lc.pad:lc.pad + lc.tile], s), img))
    return out


__all__ = ["upsample", "scaled_sem", "make_fine_sample", "make_fine_scribe_patches", "FineSample", "WaferDie"]
