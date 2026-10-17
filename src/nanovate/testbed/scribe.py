"""Scribe-line calibration patches: 3 large, isolated features per layer.

Rendered with the same forward model (same SEMCfg) as in-die tiles. Used only by Module 1.
"""
from __future__ import annotations

import numpy as np

from ..config import TestbedCfg
from ..rng import child_rng
from .dataset import WaferDie, render
from .layouts import Layout, _rect, rasterize
from .perturb import apply_ler, grain_field


def _features(N: int, rng: np.random.Generator) -> list[list[np.ndarray]]:
    c = N / 2
    j = rng.uniform(-6, 6, 2)
    pad_sq = [_rect(c - 60 + j[0], c - 60 + j[1], c + 60 + j[0], c + 60 + j[1])]
    wide_line = [_rect(-20, c - 30 + j[1], N + 20, c + 30 + j[1])]
    cross = [_rect(c - 20 + j[0], c - 90, c + 20 + j[0], c + 90), _rect(c - 90, c - 20 + j[1], c + 90, c + 20 + j[1])]
    return [pad_sq, wide_line, cross]


def make_scribe_patches(cfg: TestbedCfg, seed: int, layer: str = "layer0") -> list[tuple[np.ndarray, np.ndarray]]:
    """Return [(cad_tile, image_tile)] x 3 for one layer (image in e-/px)."""
    lc = cfg.layout
    N = lc.tile + 2 * lc.pad
    rng = child_rng(seed, "scribe", layer)
    out = []
    for i, polys in enumerate(_features(N, rng)):
        cad = rasterize(polys, N)
        lay = Layout("scribe", polys, N, lc.pad, lc.tile, cad)
        g = child_rng(seed, "scribe_geom", layer, i)
        W = apply_ler(polys, N, cfg.nuisance, g)
        wafer = WaferDie("nuisance", W, grain_field(N, cfg.nuisance, g), np.zeros_like(W), {}, {}, 0)
        img, _, _ = render(lay, wafer, cfg.sem, child_rng(seed, "scribe_acq", layer, i))
        out.append((lay.cad.copy(), img))
    return out
