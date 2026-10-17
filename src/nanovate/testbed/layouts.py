"""CAD layout families built with gdstk and rasterised with skimage.

Coordinates: (x, y) in nm on a square canvas of ``N = tile + 2*pad`` px at 1 nm/px.
Pixel (row r, col c) has its centre at (x=c, y=r). The tile is canvas[pad:pad+tile].

Families
  A "manhattan": parallel lines at 14 nm half-pitch, with designed line-end cuts and
    straps between neighbouring lines (straps create merges and loops).
  B "curvy": parallel spline wires (shared smooth undulation + per-wire wobble), rotated
    by a random angle, with occasional line-end cuts.
  C "via": square/round pillar arrays with one or two wide rails.
"""
from __future__ import annotations

from dataclasses import dataclass

import gdstk
import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label as sk_label

from ..config import LayoutCfg

FAMILIES = ("manhattan", "curvy", "via")


@dataclass
class Layout:
    """One die layout: polygons (list of (n, 2) xy arrays, nm) and the binary CAD canvas."""

    family: str
    polygons: list[np.ndarray]
    N: int
    pad: int
    tile: int
    cad_canvas: np.ndarray  # bool (N, N)

    @property
    def cad(self) -> np.ndarray:
        """Binary CAD on the tile (tile, tile)."""
        p, t = self.pad, self.tile
        return self.cad_canvas[p:p + t, p:p + t]


def fill_polygon(pts: np.ndarray, N: int) -> np.ndarray:
    """Even-odd scanline fill of one xy polygon on an (N, N) canvas.

    Pixel (r, c) is inside iff its centre (x=c, y=r) is inside the polygon. Equivalent to
    ``skimage.draw.polygon`` up to centres lying exactly on an edge (pinned by a test),
    but O(rows x edges) instead of O(bbox x edges), which matters for long spline wires.
    """
    out = np.zeros((N, N), dtype=bool)
    x0, y0 = pts[:, 0], pts[:, 1]
    x1, y1 = np.roll(x0, -1), np.roll(y0, -1)
    rlo = max(0, int(np.ceil(min(y0.min(), N))))
    rhi = min(N - 1, int(np.floor(y0.max())))
    if rhi < rlo:
        return out
    rows = np.arange(rlo, rhi + 1, dtype=float)[:, None]
    cross = ((y0 <= rows) & (rows < y1)) | ((y1 <= rows) & (rows < y0))
    with np.errstate(divide="ignore", invalid="ignore"):
        xc = x0 + (rows - y0) * (x1 - x0) / (y1 - y0)
    for i, r in enumerate(range(rlo, rhi + 1)):
        xs = np.sort(xc[i][cross[i]])
        for a, b in zip(xs[0::2], xs[1::2]):
            ca = max(0, int(np.ceil(a)))
            cb = min(N, int(np.ceil(b)))
            if cb > ca:
                out[r, ca:cb] = True
    return out


def rasterize(polygons: list[np.ndarray], N: int) -> np.ndarray:
    """Union-rasterise xy polygons onto an (N, N) bool canvas (pixel-centre rule)."""
    out = np.zeros((N, N), dtype=bool)
    for pts in polygons:
        out |= fill_polygon(pts, N)
    return out


def _rect(x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    return np.asarray(gdstk.rectangle((x0, y0), (x1, y1)).points, dtype=float)


def _manhattan(cfg: LayoutCfg, rng: np.random.Generator, N: int) -> list[np.ndarray]:
    w = cfg.man_half_pitch
    pitch = 2 * w
    off = rng.uniform(0, pitch)
    ys = np.arange(off - pitch, N + pitch, pitch)
    lo, hi = -20.0, N + 20.0
    polys: list[np.ndarray] = []
    for y in ys:
        if rng.random() < cfg.man_cut_prob:
            xc = rng.uniform(cfg.pad + cfg.margin, N - cfg.pad - cfg.margin)
            g = rng.uniform(*cfg.man_cut_gap)
            polys.append(_rect(lo, y, xc - g / 2, y + w))
            polys.append(_rect(xc + g / 2, y, hi, y + w))
        else:
            polys.append(_rect(lo, y, hi, y + w))
    # straps join line i and i+1 (one strap merges, two straps also close a loop)
    for y in ys[:-1]:
        if rng.random() < cfg.man_strap_prob:
            n_straps = 1 + int(rng.random() < 0.5)
            xs = rng.uniform(cfg.pad, N - cfg.pad, size=n_straps)
            for xs_i in xs:
                polys.append(_rect(xs_i, y + w / 2, xs_i + w, y + pitch + w / 2))
    if rng.random() < 0.5:  # vertical orientation
        polys = [p[:, ::-1].copy() for p in polys]
    return polys


def _curvy(cfg: LayoutCfg, rng: np.random.Generator, N: int) -> list[np.ndarray]:
    ext = 1.5 * N  # generate over an enlarged extent, then rotate about the canvas centre
    cx = cy = N / 2
    amp = rng.uniform(*cfg.curvy_amp)
    lam = rng.uniform(*cfg.curvy_wavelength)
    phase = rng.uniform(0, 2 * np.pi)
    theta = rng.uniform(0, np.pi)
    xs = np.arange(-ext / 2, ext / 2 + 1, 2.0)
    base = amp * np.sin(2 * np.pi * xs / lam + phase)
    pitch = cfg.curvy_pitch
    offs = np.arange(-ext / 2, ext / 2, pitch) + rng.uniform(0, pitch)
    c, s = np.cos(theta), np.sin(theta)
    polys: list[np.ndarray] = []
    for y0 in offs:
        # small per-wire wobble (smooth, low amplitude so spacing is preserved)
        wob = ndi.gaussian_filter1d(rng.normal(0, 1, xs.size), 20) * 8.0
        wob = np.clip(wob, -2.0, 2.0)
        ys = y0 + base + wob
        width = rng.uniform(*cfg.curvy_width)
        pts = np.stack([xs, ys], 1)
        segments = [pts]
        if rng.random() < 0.3:  # line-end cut somewhere near the middle of the canvas
            k = int(rng.integers(xs.size // 2 - 40, xs.size // 2 + 40))
            gap = int(rng.uniform(9, 12))  # 2 nm per sample -> 18-24 nm gap
            segments = [pts[:k], pts[k + gap:]]
        for seg in segments:
            path = gdstk.FlexPath(seg.tolist(), width, tolerance=0.05)
            for poly in path.to_polygons():
                p = np.asarray(poly.points, dtype=float)
                xr = c * p[:, 0] - s * p[:, 1] + cx
                yr = s * p[:, 0] + c * p[:, 1] + cy
                polys.append(np.stack([xr, yr], 1))
    return polys


def _via(cfg: LayoutCfg, rng: np.random.Generator, N: int) -> list[np.ndarray]:
    pitch = cfg.via_pitch
    size = rng.uniform(*cfg.via_size)
    round_ = rng.random() < cfg.via_round_prob
    ox, oy = rng.uniform(0, pitch, 2)
    rails = []
    for _ in range(1 + int(rng.random() < 0.5)):
        rw = rng.uniform(*cfg.rail_width)
        ry = rng.uniform(cfg.pad + cfg.margin, N - cfg.pad - cfg.margin - rw)
        rails.append((ry, rw))
    polys: list[np.ndarray] = []
    for ry, rw in rails:
        polys.append(_rect(-20, ry, N + 20, ry + rw))
    clear = cfg.min_space + 2
    for yc in np.arange(oy - pitch, N + pitch, pitch):
        for xc in np.arange(ox - pitch, N + pitch, pitch):
            if any(yc + size / 2 + clear > ry and yc - size / 2 - clear < ry + rw for ry, rw in rails):
                continue
            if round_:
                pts = gdstk.ellipse((xc, yc), size / 2, tolerance=0.05).points
            else:
                pts = gdstk.rectangle((xc - size / 2, yc - size / 2), (xc + size / 2, yc + size / 2)).points
            polys.append(np.asarray(pts, dtype=float))
    return polys


_BUILDERS = {"manhattan": _manhattan, "curvy": _curvy, "via": _via}


def min_space_ok(cad: np.ndarray, min_space: float) -> bool:
    """True if no two distinct conductor components are closer than ~``min_space`` px."""
    r = max(1, int(min_space // 2) - 1)
    n0 = sk_label(cad, connectivity=1).max()
    dil = ndi.binary_dilation(cad, structure=ndi.generate_binary_structure(2, 1), iterations=r)
    return sk_label(dil, connectivity=1).max() == n0


def make_layout(family: str, cfg: LayoutCfg, rng: np.random.Generator, max_tries: int = 50) -> Layout:
    """Build one CAD layout of ``family``; regenerates until spacing rules hold."""
    if family not in _BUILDERS:
        raise ValueError(f"unknown family {family!r}; expected one of {FAMILIES}")
    N = cfg.tile + 2 * cfg.pad
    for _ in range(max_tries):
        polys = _BUILDERS[family](cfg, rng, N)
        cad = rasterize(polys, N)
        if min_space_ok(cad, cfg.min_space):
            return Layout(family, polys, N, cfg.pad, cfg.tile, cad)
    raise RuntimeError(f"could not build a DRC-clean {family} layout in {max_tries} tries")
