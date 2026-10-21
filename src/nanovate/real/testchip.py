"""Track C Rung 2: a test-chip layout (GDS) with deliberate defects of known size, plus its exact defect map.

Every block is a small line/space or via-array patch (one defect, or none, at its centre), so one SEM field of view
per block is enough. All lengths are nm and scale with the line width ``w`` (``ChipCfg.w``), so the same design can be
written at a conservative or aggressive feature size. Polygons on the pattern layer are the CONDUCTOR (the material
that stays: metal after lift-off, or resist/oxide for a negative resist); the facility confirms the polarity.

Ground truth is checked, not assumed: each block is rasterised and its Betti numbers are compared with its clean
twin, so the defect map's expected Delta-beta comes from the drawn geometry (same rule as the synthetic testbed).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field

import gdstk
import numpy as np

from ..testbed.layouts import rasterize
from ..modules.topology import classify_verdict
from ..topo_utils import betti_binary

LAYER_PATTERN = 1     # expose: lines, vias, defects, calibration features
LAYER_NAV = 2         # expose (same dose): alignment crosses and block labels, kept >= 0.3 block gap from any block
LAYER_DOC = 63        # documentation only, do NOT expose: block outlines

PRESETS = {
    "ebeam": dict(w=100.0),             # conservative university e-beam default (100 nm lines / 100 nm spaces)
    "ebeam_aggressive": dict(w=50.0),
    "optical": dict(w=1000.0),          # contact / i-line optical lithography
}


@dataclass(frozen=True)
class ChipCfg:
    """Test-chip geometry in nm; defect sizes are fractions of the line width ``w``."""

    w: float = 100.0                                    # line width = space = via size, nm
    block_pitches: int = 10                             # block edge = block_pitches * 2w
    block_gap_frac: float = 0.5                         # gap between blocks, as a fraction of the block edge
    reps: int = 3                                       # replicate sets (independent placements)
    seed: int = 0                                       # shuffles block order inside each replicate
    bridge: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)  # bridge width / w
    open_: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)   # cut length / w
    particle: tuple[float, ...] = (0.25, 0.5, 0.75)     # isolated square side / w (in the space)
    notch: tuple[float, ...] = (0.25, 0.5, 0.75)        # remaining wire width / w (near-miss)
    filament: tuple[float, ...] = (0.1, 0.2, 0.3)       # gap left to the neighbour / w (near-miss)
    via_bridge: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0)
    via_extra: tuple[float, ...] = (0.25, 0.5, 0.75)    # extra conductive square between 4 vias / w
    extra_clean_ls: int = 3                             # extra clean line/space controls per replicate
    extra_clean_via: int = 2

    @property
    def pitch(self) -> float:
        return 2.0 * self.w

    @property
    def block(self) -> float:
        return self.block_pitches * self.pitch

    @property
    def gap(self) -> float:
        return self.block_gap_frac * self.block


@dataclass
class Block:
    block_id: str
    family: str           # 'line_space' | 'via_array' | 'calibration'
    defect: str           # 'clean' | 'bridge' | 'open' | 'particle' | 'notch' | 'filament' | 'via_bridge' | 'missing_via' | 'extra_via' | cal name
    size_frac: float | None
    size_param: str
    polys: list = field(default_factory=list)          # xy arrays, nm, block-local (0..block)
    defect_bbox: tuple | None = None                   # (x0, y0, x1, y1) nm, block-local
    rep: int = -1
    origin: tuple[float, float] = (0.0, 0.0)           # chip coords of the block's lower-left corner, nm


def _r(x0, y0, x1, y1) -> np.ndarray:
    return np.array([[x0, y0], [x1, y0], [x1, y1], [x0, y1]], float)


def _lines(c: ChipCfg) -> list[np.ndarray]:
    """Vertical lines across the block; line k spans x in [k*p + w/2, k*p + 3w/2)."""
    return [_r(k * c.pitch + c.w / 2, 0, k * c.pitch + 1.5 * c.w, c.block) for k in range(c.block_pitches)]


def _vias(c: ChipCfg, skip: tuple[int, int] | None = None) -> list[np.ndarray]:
    out = []
    for i in range(c.block_pitches):
        for j in range(c.block_pitches):
            if (i, j) == skip:
                continue
            x, y = i * c.pitch + c.w / 2, j * c.pitch + c.w / 2
            out.append(_r(x, y, x + c.w, y + c.w))
    return out


def line_space_block(c: ChipCfg, defect: str, f: float | None) -> Block:
    """Line/space block with one defect at the centre (between central lines k0 and k0 + 1)."""
    polys = _lines(c)
    k0 = c.block_pitches // 2 - 1
    xl = k0 * c.pitch + 1.5 * c.w          # right edge of line k0
    xr = xl + c.w                           # left edge of line k0 + 1
    yc = c.block / 2
    w = c.w
    bbox, param = None, ""
    if defect == "bridge":
        b = f * w
        polys.append(_r(xl - 0.01, yc - b / 2, xr + 0.01, yc + b / 2)); bbox = (xl, yc - b / 2, xr, yc + b / 2); param = "bridge width"
    elif defect == "open":
        g = f * w
        x0 = k0 * c.pitch + w / 2
        polys[k0] = _r(x0, 0, x0 + w, yc - g / 2)
        polys.insert(k0 + 1, _r(x0, yc + g / 2, x0 + w, c.block)); bbox = (x0, yc - g / 2, x0 + w, yc + g / 2); param = "cut length"
    elif defect == "particle":
        p = f * w
        xm = (xl + xr) / 2
        polys.append(_r(xm - p / 2, yc - p / 2, xm + p / 2, yc + p / 2)); bbox = (xm - p / 2, yc - p / 2, xm + p / 2, yc + p / 2); param = "particle side"
    elif defect == "notch":
        rem = f * w
        x0 = k0 * c.pitch + w / 2
        polys[k0] = _r(x0, 0, x0 + w, yc - w / 2)
        polys.insert(k0 + 1, _r(x0, yc + w / 2, x0 + w, c.block))
        polys.insert(k0 + 2, _r(x0, yc - w / 2 - 0.01, x0 + rem, yc + w / 2 + 0.01))   # constriction: only rem of w kept
        bbox = (x0 + rem, yc - w / 2, x0 + w, yc + w / 2); param = "remaining width"
    elif defect == "filament":
        q = f * w
        polys.append(_r(xl - 0.01, yc - w / 4, xr - q, yc + w / 4)); bbox = (xl, yc - w / 4, xr - q, yc + w / 4); param = "gap left to neighbour"
    elif defect != "clean":
        raise ValueError(defect)
    return Block("", "line_space", defect, f, param, polys, bbox)


def via_block(c: ChipCfg, defect: str, f: float | None) -> Block:
    i0 = c.block_pitches // 2 - 1
    w = c.w
    bbox, param = None, ""
    skip = (i0, i0) if defect == "missing_via" else None
    polys = _vias(c, skip)
    x0, y0 = i0 * c.pitch + w / 2, i0 * c.pitch + w / 2          # lower-left via of the central 2 x 2 cell
    if defect == "via_bridge":
        b = f * w
        yc = y0 + w / 2
        polys.append(_r(x0 + w - 0.01, yc - b / 2, x0 + c.pitch + 0.01, yc + b / 2)); bbox = (x0 + w, yc - b / 2, x0 + c.pitch, yc + b / 2)
        param = "bridge width"
    elif defect == "missing_via":
        bbox = (x0, y0, x0 + w, y0 + w); param = "via size (absent)"; f = 1.0
    elif defect == "extra_via":
        p = f * w
        xm, ym = x0 + w + w / 2, y0 + w + w / 2                       # centre of the gap between 4 vias
        polys.append(_r(xm - p / 2, ym - p / 2, xm + p / 2, ym + p / 2)); bbox = (xm - p / 2, ym - p / 2, xm + p / 2, ym + p / 2)
        param = "extra square side"
    elif defect != "clean":
        raise ValueError(defect)
    return Block("", "via_array", defect, f, param, polys, bbox)


def calibration_blocks(c: ChipCfg) -> list[Block]:
    """Three large isolated features (Module 1 scribe patches): a square, a wide line, a disc; each about block/2."""
    B, h = c.block, c.block / 4
    disc = np.asarray(gdstk.ellipse((B / 2, B / 2), h, tolerance=c.w / 50).points, float)
    return [Block("", "calibration", "cal_square", None, "", [_r(B / 2 - h, B / 2 - h, B / 2 + h, B / 2 + h)]),
            Block("", "calibration", "cal_line", None, "", [_r(B / 2 - h / 2, B / 8, B / 2 + h / 2, B - B / 8)]),
            Block("", "calibration", "cal_disc", None, "", [disc])]


def block_set(c: ChipCfg) -> list[Block]:
    """One replicate: every defect type at every size, plus clean controls (unshuffled)."""
    out = [line_space_block(c, "clean", None) for _ in range(1 + c.extra_clean_ls)]
    for name, fs in (("bridge", c.bridge), ("open", c.open_), ("particle", c.particle), ("notch", c.notch), ("filament", c.filament)):
        out += [line_space_block(c, name, f) for f in fs]
    out += [via_block(c, "clean", None) for _ in range(1 + c.extra_clean_via)]
    out += [via_block(c, "via_bridge", f) for f in c.via_bridge]
    out += [via_block(c, "missing_via", None)]
    out += [via_block(c, "extra_via", f) for f in c.via_extra]
    return out


def raster(b: Block, c: ChipCfg, px_nm: float | None = None) -> np.ndarray:
    """Binary raster of one block (True = conductor), pixel-centre rule; default px = w / 20."""
    px = px_nm or c.w / 20.0
    N = int(round(c.block / px))
    return rasterize([p / px for p in b.polys], N)


EXPECTED = {   # defect -> (label, exact design Delta-beta vs the clean twin)
    "clean": ("NUISANCE", (0, 0)), "notch": ("NEAR_MISS", (0, 0)), "filament": ("NEAR_MISS", (0, 0)),
    "bridge": ("KILLER", (-1, 0)), "via_bridge": ("KILLER", (-1, 0)), "open": ("KILLER", (1, 0)),
    "particle": ("KILLER", (1, 0)), "extra_via": ("KILLER", (1, 0)),
    "missing_via": ("KILLER", (-1, 0)),   # a lost component: the topology rule names Delta-beta0 < 0 "BRIDGE" (see rule_verdict)
}


def rule_verdict(d_beta0: int, d_beta1: int) -> str:
    """Verdict Module 4's ladder (topology.classify_verdict) gives for an exact Delta-beta; (0, 0) depends on the near-miss gate."""
    if (d_beta0, d_beta1) == (0, 0):
        return "NUISANCE or NEAR_MISS"
    return classify_verdict((d_beta0, d_beta1))


def design_delta_beta(b: Block, c: ChipCfg) -> tuple[int, int]:
    """Exact (d_beta0, d_beta1) of the drawn block vs its clean twin, on the full block raster."""
    clean = line_space_block(c, "clean", None) if b.family == "line_space" else via_block(c, "clean", None)
    b0, b1 = betti_binary(raster(b, c))
    r0, r1 = betti_binary(raster(clean, c))
    return b0 - r0, b1 - r1


def build_chip(c: ChipCfg) -> tuple[gdstk.Library, list[Block]]:
    """GDS library (one top cell 'NANOVATE_TESTCHIP') and the placed blocks (ids, reps and origins filled)."""
    rng = np.random.default_rng(c.seed)
    blocks: list[Block] = []
    cols = 8
    step = c.block + c.gap
    y_cursor = 0.0
    for rep in range(c.reps):
        bs = block_set(c)
        order = rng.permutation(len(bs))
        bs = [bs[i] for i in order]
        rows = int(np.ceil(len(bs) / cols))
        for k, b in enumerate(bs):
            r, col = divmod(k, cols)
            b.rep = rep
            b.origin = (col * step, y_cursor + (rows - 1 - r) * step)
            b.block_id = f"R{rep}-{chr(ord('A') + r)}{col + 1}"
            blocks.append(b)
        y_cursor += rows * step + c.gap
    # calibration row on top
    for k, b in enumerate(calibration_blocks(c)):
        b.rep, b.origin, b.block_id = -1, (k * step, y_cursor), f"CAL{k + 1}"
        blocks.append(b)
    width = cols * step - c.gap
    height = y_cursor + c.block
    lib = gdstk.Library("NANOVATE", unit=1e-9, precision=1e-11)     # database unit 1 nm
    cell = lib.new_cell("NANOVATE_TESTCHIP")
    for b in blocks:
        ox, oy = b.origin
        for p in b.polys:
            cell.add(gdstk.Polygon((p + [ox, oy]).tolist(), layer=LAYER_PATTERN))
        cell.add(gdstk.rectangle((ox, oy), (ox + c.block, oy + c.block), layer=LAYER_DOC))
        th = c.gap * 0.35
        for poly in gdstk.text(b.block_id, th, (ox, oy - c.gap * 0.5 - th / 2), layer=LAYER_NAV):
            cell.add(poly)
    # alignment crosses at the four corners, one block edge outside the array
    arm, lw = c.block / 2, max(c.w * 2, c.block / 10)
    for cx, cy in ((-c.block, -c.block), (width + c.block, -c.block), (-c.block, height + c.block), (width + c.block, height + c.block)):
        cell.add(gdstk.rectangle((cx - arm, cy - lw / 2), (cx + arm, cy + lw / 2), layer=LAYER_NAV))
        cell.add(gdstk.rectangle((cx - lw / 2, cy - arm), (cx + lw / 2, cy + arm), layer=LAYER_NAV))
    return lib, blocks


def defect_rows(blocks: list[Block], c: ChipCfg) -> list[dict]:
    """Defect map rows (positions in um, chip coordinates, origin = lower-left of block R0 row bottom-left)."""
    rows = []
    for b in blocks:
        ox, oy = b.origin
        db = design_delta_beta(b, c) if b.family != "calibration" else (None, None)
        bb = b.defect_bbox
        rows.append(dict(
            block_id=b.block_id, rep=b.rep, family=b.family, defect_type=b.defect,
            label=EXPECTED[b.defect][0] if b.defect in EXPECTED else "CALIBRATION",
            rule_verdict=rule_verdict(*db) if db[0] is not None else "CALIBRATION",
            size_param=b.size_param, size_frac_w=b.size_frac,
            size_nm=None if b.size_frac is None else round(b.size_frac * c.w, 3),
            design_d_beta0=db[0], design_d_beta1=db[1],
            block_x0_um=ox / 1e3, block_y0_um=oy / 1e3, block_x1_um=(ox + c.block) / 1e3, block_y1_um=(oy + c.block) / 1e3,
            defect_x_um=None if bb is None else (ox + (bb[0] + bb[2]) / 2) / 1e3,
            defect_y_um=None if bb is None else (oy + (bb[1] + bb[3]) / 2) / 1e3,
            defect_bbox_um=None if bb is None else "{:.4f} {:.4f} {:.4f} {:.4f}".format(*(np.array([ox, oy, ox, oy]) + bb) / 1e3),
            line_width_nm=c.w, pitch_nm=c.pitch))
    return rows


def cfg_dict(c: ChipCfg) -> dict:
    d = dataclasses.asdict(c)
    d.update(pitch_nm=c.pitch, block_nm=c.block, gap_nm=c.gap, layers=dict(pattern=LAYER_PATTERN, nav=LAYER_NAV, doc_do_not_expose=LAYER_DOC))
    return d
