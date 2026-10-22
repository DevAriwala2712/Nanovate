"""Systematic CD offset (Rung 1d): the wafer's conductor is uniformly wider (b > 0) or narrower (b < 0) than CAD.

``grow(mask, b)`` uses an exact Euclidean distance transform at 1 nm/px: b > 0 adds every dielectric pixel within b of
conductor, b < 0 removes every conductor pixel within |b| of dielectric; b = 0 returns the mask unchanged. The same
function builds the CD-corrected CAD in the verifier. ``make_sample_cd`` re-renders the testbed's own wafer with the
offset applied, using the same random streams as ``make_sample``, and relabels from the offset wafer. The existing
generators are not modified (b = 0 reproduces ``make_sample`` bit for bit; tested). Units: nm = px.
"""
from __future__ import annotations

import dataclasses

import numpy as np
from scipy import ndimage as ndi

from ..config import TestbedCfg
from ..rng import child_rng
from .dataset import LABEL_OF_KIND, Sample, WaferDie, get_layout, make_wafer, render, window
from .labels import consistent, topology_label
from .layouts import Layout, rasterize
from .perturb import apply_ler, grain_field
from .scribe import _features


def grow(mask: np.ndarray, b: float) -> np.ndarray:
    """Conductor mask grown by b px (b > 0) or shrunk by |b| px (b < 0)."""
    m = np.asarray(mask, bool)
    if b == 0:
        return m.copy()
    if b > 0:
        return m | (ndi.distance_transform_edt(~m) <= b)
    return m & (ndi.distance_transform_edt(m) > -b)


def make_sample_cd(family: str, split: str, layout_id: int, die: int, kind: str, cfg: TestbedCfg, seed: int, b: float,
                   layout: Layout | None = None) -> tuple[Sample, bool]:
    """(sample rendered from grow(W, b), topology_ok). topology_ok is False when the offset itself changed the topology
    away from ``kind`` (such patches are excluded by the Rung 1d protocol)."""
    layout = layout or get_layout(family, split, layout_id, cfg, seed)
    wafer = make_wafer(layout, kind, cfg, child_rng(seed, "geom", family, split, layout_id, die, kind))
    p, t = layout.pad, layout.tile
    Wb = grow(wafer.W_canvas, b)
    topo = topology_label(Wb[p:p + t, p:p + t], layout.cad, window(cfg)) if b != 0 else wafer.topo
    ok = b == 0 or consistent(kind, topo)
    wb = dataclasses.replace(wafer, W_canvas=Wb, topo=topo)
    img, exp, sh = render(layout, wb, cfg.sem, child_rng(seed, "acq", family, split, layout_id, die, ""))
    s = Sample(family, split, layout_id, die, kind, LABEL_OF_KIND[kind], topo, layout.cad.copy(), Wb[p:p + t, p:p + t].copy(),
               wafer.mask_canvas[p:p + t, p:p + t].copy(), img, exp, sh, cfg.sem.dose,
               wafer.params | {"cd_bias_nm": b, "resample_reasons": wafer.resample_reasons}, wafer.resamples)
    return s, ok


def make_scribe_patches_cd(cfg: TestbedCfg, seed: int, b: float, layer: str = "layer0") -> list[tuple[np.ndarray, np.ndarray]]:
    """``scribe.make_scribe_patches`` with the same CD offset applied to each scribe wafer (same random streams)."""
    lc = cfg.layout
    N = lc.tile + 2 * lc.pad
    rng = child_rng(seed, "scribe", layer)
    out = []
    for i, polys in enumerate(_features(N, rng)):
        cad = rasterize(polys, N)
        lay = Layout("scribe", polys, N, lc.pad, lc.tile, cad)
        g = child_rng(seed, "scribe_geom", layer, i)
        W = grow(apply_ler(polys, N, cfg.nuisance, g), b)
        wafer = WaferDie("nuisance", W, grain_field(N, cfg.nuisance, g), np.zeros_like(W), {}, {}, 0)
        img, _, _ = render(lay, wafer, cfg.sem, child_rng(seed, "scribe_acq", layer, i))
        out.append((lay.cad.copy(), img))
    return out
