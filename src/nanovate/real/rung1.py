"""Track C Rung 1: real SEM images of defect-free hole arrays through the FROZEN v3 verifier (real data).

Everything that is chosen here is a fixed rule written into docs/REAL_DATA_PREREG.md before the verifier sees a real
image: which images are used, which regions calibrate, how the FITTED reference is built, how tiles are cut, and the
two pre-declared scale arms. Nothing is selected from verifier output. Lengths: px of the image at hand unless the
name ends in ``_nm``. Intensities: raw 8-bit grey levels (no e-/px; the dose is unknown).

Reference = FITTED, not CAD: lattice basis fitted on calibration images only, hole diameter from the design value in
the file name (``dset``, the FIB set diameter) when one exists, else fitted on calibration regions, and a per-image
translation registered from the image's own hole centres (2 parameters; real images share no coordinate frame).
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from skimage.transform import rescale as sk_rescale

from .lattice import Lattice, detect_holes, render

TILE = 256                  # px, the verifier's tile (frozen)
VIA_PX = 16.0               # px, testbed via size mid-range (LayoutCfg.via_size = 14-18 nm at 1 nm/px), arm B target
G29_CAL_INDEX = (0, 16, 31)  # calibration images: positions in the filename-sorted G29 list (spread over depth settings)
G01_CAL_IMAGE = "z_110_overview_002.tif"
G01_CAL_TILES = ((0, 0), (0, 2), (0, 4))   # tile (row, col) used as Module-1 scribe stand-ins
G01_CAL_ROWS = 2            # tile rows 0..1 of the calibration image = calibration region (lattice, diameter); excluded from eval


@dataclass
class RealImage:
    """One real SEM image (data bar removed) and its provenance."""

    image_id: str
    group: str                 # 'G29' | 'G01' | 'INGAN'
    path: str
    px_nm: float               # nm/px from FEI metadata (or scale bar for INGAN)
    px_source: str
    img: np.ndarray            # float (H, W), grey levels
    design_d_nm: float | None  # FIB set diameter from the file name (dset / d), None if absent or a range
    meta: dict = field(default_factory=dict)
    overlay: np.ndarray | None = None   # bool (H, W): burned-in display overlay (FEI centre cross), pixels left unaltered


def sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def design_diameter(name: str) -> float | None:
    """Set diameter in nm from 'dset_226' / '_d_100_' style names; None for ranges ('220-280') or none."""
    m = re.search(r"d(?:set)?_(\d+)(?:_|\.|$)", name)
    if m is None or re.search(r"d_\d+-\d+", name):
        return None
    return float(m.group(1))


def load_fei(path: str | Path, group: str) -> RealImage:
    """FEI/Thermo TIFF: grey image above the data bar, pixel size and imaging conditions from the embedded metadata."""
    import tifffile
    p = Path(path)
    with tifffile.TiffFile(p) as t:
        a = t.pages[0].asarray()
        m = t.fei_metadata
    ry = int(m["Image"]["ResolutionY"])
    a = a[:ry]
    overlay = None
    if a.ndim == 3:
        # grey SEM data is stored as R = G = B; a coloured display overlay (the yellow centre cross) breaks that.
        # Those pixels are flagged, never inpainted: their grey value is unknown.
        overlay = (a[..., 0] != a[..., 1]) | (a[..., 0] != a[..., 2])
        a = a[..., 0]
    meta = dict(system=m["System"].get("SystemType"), detector=m["Detectors"].get("Name"), mode=m["Detectors"].get("Mode"),
                hv_V=m["Beam"].get("HV"), current_A=m["EBeam"].get("BeamCurrent"), dwell_s=m["Scan"].get("Dwelltime"),
                hfw_m=m["Scan"].get("HorFieldsize"), tilt_rad=m["Stage"].get("StageT"), integrate=m["Image"].get("Integrate"),
                average=m["Image"].get("Average"), dtype=str(a.dtype), clip_lo=float((a == 0).mean()), clip_hi=float((a == 255).mean()),
                sha256=sha256(p), overlay_px=int(overlay.sum()) if overlay is not None else 0)
    return RealImage(p.name, group, str(p), float(m["Scan"]["PixelWidth"]) * 1e9, "FEI TIFF metadata Scan/PixelWidth",
                     a.astype(float), design_diameter(p.name), meta, overlay)


def scale_bar_nm_per_px(a: np.ndarray, bar_nm: float) -> float:
    """Pixel size from a white scale bar in the bottom-right corner (figure JPGs): bar_nm / longest white run."""
    H, W = a.shape
    sub = a[int(H * 0.85):, int(W * 0.75):] > 245
    r = sub.sum(1).argmax()
    cols = np.nonzero(sub[r])[0]
    runs = np.split(cols, np.nonzero(np.diff(cols) != 1)[0] + 1)
    return bar_nm / max(len(x) for x in runs)


def load_figure_jpg(path: str | Path, bar_nm: float = 200.0, keep_rows: int = 512) -> RealImage:
    """Spot-check figure panel (JPEG, no metadata): grey = mean of RGB, rows below ``keep_rows`` (scale bar) dropped."""
    from PIL import Image
    p = Path(path)
    a = np.asarray(Image.open(p).convert("L")).astype(float)
    px = scale_bar_nm_per_px(a, bar_nm)
    meta = dict(system="unknown (figure panel)", detector="unknown", dtype="uint8 JPEG", sha256=sha256(p),
                note="published figure panel, JPEG-compressed, possibly resampled by the authors")
    return RealImage(p.name, "INGAN", str(p), float(px), f"scale bar ({bar_nm:g} nm) measured in px", a[:keep_rows], None, meta)


# --------------------------------------------------------------------------- selection (pre-registered rule)

def select_zmw(root: str | Path) -> tuple[list[str], list[str], list[tuple[str, str]]]:
    """(G29 paths, G01 paths, [(excluded path, reason)]) under data/real/zmw/extracted."""
    import tifffile
    g29, g01, ex = [], [], []
    for p in sorted(Path(root).rglob("*.tif")):
        with tifffile.TiffFile(p) as t:
            m = t.fei_metadata
        tilt = abs(float(m["Stage"]["StageT"]))
        px = float(m["Scan"]["PixelWidth"]) * 1e9
        if tilt > 1e-3:
            ex.append((str(p), f"tilted {np.degrees(tilt):.0f} deg (cross-section)"))
        elif p.parent.name == "Gla_29_Pd_1" and p.name.startswith("2022_03_15"):
            g29.append(str(p))
        elif p.parent.name == "Gla_0127_14" and ("overview" in p.name or "220-280" in p.name) and px <= 30.0:
            g01.append(str(p))
        elif "overview" in p.name:
            ex.append((str(p), f"overview at {px:.1f} nm/px (> 30 nm/px: holes ~3 px)"))
        else:
            ex.append((str(p), "single well or two wells per field (no array)"))
    return g29, g01, ex


# --------------------------------------------------------------------------- reference

def _short_vectors(c: np.ndarray) -> np.ndarray:
    D = (c[None] - c[:, None]).reshape(-1, 2)
    n = np.hypot(*D.T)
    return D[n > 0]


def fit_basis_nm(centres_nm: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Lattice basis (nm, row/col) fitted jointly on several images' hole centres (each image its own origin)."""
    V = np.concatenate([_short_vectors(c) for c in centres_nm if len(c) > 1])
    n = np.hypot(*V.T)
    v1 = V[np.argmin(n)]
    near = V[(np.abs(np.hypot(*V.T) - np.hypot(*v1)) < 0.15 * np.hypot(*v1))]
    near = np.where(((near @ v1) < 0)[:, None], -near, near)
    a1 = near[np.abs(np.cross(near, v1)) / (np.hypot(*near.T) * np.hypot(*v1)) < 0.3].mean(0)
    cos = np.abs(V @ a1) / (np.hypot(*V.T) * np.hypot(*a1))
    W = V[cos < 0.87]                    # > 30 deg from a1
    w1 = W[np.argmin(np.hypot(*W.T))]
    near2 = W[np.abs(np.hypot(*W.T) - np.hypot(*w1)) < 0.15 * np.hypot(*w1)]
    near2 = near2[np.abs(np.cross(near2, w1)) / (np.hypot(*near2.T) * np.hypot(*w1)) < 0.3]
    near2 = np.where(((near2 @ w1) < 0)[:, None], -near2, near2)
    a2 = near2.mean(0)
    # joint least-squares refinement: c = o_k + i a1 + j a2
    A = np.stack([a1, a2], 1)
    rows, rhs, k0 = [], [], 0
    K = len(centres_nm)
    for k, c in enumerate(centres_nm):
        ij = np.rint(np.linalg.solve(A, (c - c[0]).T).T)
        for (i, j), cc in zip(ij, c):
            for ax in (0, 1):
                r = np.zeros(4 + 2 * K)
                r[ax * 2:ax * 2 + 2] = (i, j)   # a1[ax], a2[ax] -> columns (a1r, a2r, a1c, a2c)
                r[4 + 2 * k + ax] = 1.0
                rows.append(r); rhs.append(cc[ax])
    x, *_ = np.linalg.lstsq(np.array(rows), np.array(rhs), rcond=None)
    return np.array([x[0], x[2]]), np.array([x[1], x[3]])


def register_origin(centres: np.ndarray, a1: np.ndarray, a2: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Lattice origin (px) from an image's own hole centres, modulo the lattice; and each hole's residual (px).

    Robust to missing or extra holes: the origin is the circular median of the wrapped positions.
    """
    A = np.stack([a1, a2], 1)
    q = np.linalg.solve(A, np.asarray(centres, float).T).T          # lattice coordinates
    ang = 2 * np.pi * q
    mean = np.angle(np.exp(1j * ang).mean(0)) / (2 * np.pi)          # circular mean, per axis
    fr = q - mean
    fr -= np.rint(fr)
    o_q = mean + np.median(fr, axis=0)
    o = A @ o_q
    res_q = q - o_q
    res_q -= np.rint(res_q)
    res = np.hypot(*(A @ res_q.T))
    return o, res


@dataclass(frozen=True)
class Reference:
    lattice: Lattice           # px of the image
    d_source: str              # 'design (file name)' | 'fitted (calibration regions)'
    hole_resid_px: tuple       # per detected hole: distance to the nearest reference centre, px
    n_holes_detected: int


def build_reference(im: RealImage, a1_nm: np.ndarray, a2_nm: np.ndarray, d_fit_nm: float | None) -> Reference:
    """FITTED reference for one image: fixed basis (nm), design or calibration diameter, registered origin."""
    rp = detect_holes(im.img)
    c = np.array([r.centroid for r in rp])
    a1, a2 = a1_nm / im.px_nm, a2_nm / im.px_nm
    o, res = register_origin(c, a1, a2)
    if im.design_d_nm is not None:
        d_nm, src = im.design_d_nm, "design (file name)"
    else:
        if d_fit_nm is None:
            raise ValueError(f"{im.image_id}: no design diameter and no fitted diameter")
        d_nm, src = d_fit_nm, "fitted (calibration regions)"
    lat = Lattice(tuple(map(float, a1)), tuple(map(float, a2)), tuple(map(float, o)), float(d_nm / im.px_nm))
    return Reference(lat, src, tuple(map(float, res)), len(c))


def fitted_diameter_nm(ims: list[RealImage], regions: list[tuple[int, int]] | None = None) -> float:
    """Median equivalent diameter (nm) of holes detected inside the given tiles (or whole images)."""
    ds = []
    for im in ims:
        rp = detect_holes(im.img)
        for r in rp:
            if regions is not None:
                rr, cc = r.centroid
                if (int(rr // TILE), int(cc // TILE)) not in regions:
                    continue
            ds.append(r.equivalent_diameter_area * im.px_nm)
    return float(np.median(ds))


# --------------------------------------------------------------------------- tiles

def tile_grid(shape: tuple[int, int]) -> list[tuple[int, int]]:
    """Non-overlapping TILE x TILE tiles from the top-left corner, (row, col) indices."""
    return [(r, c) for r in range(shape[0] // TILE) for c in range(shape[1] // TILE)]


def crop(a: np.ndarray, rc: tuple[int, int]) -> np.ndarray:
    r, c = rc
    return a[r * TILE:(r + 1) * TILE, c * TILE:(c + 1) * TILE]


def holes_inside(cad: np.ndarray) -> int:
    """Number of reference holes fully inside a tile (dielectric components not touching the border)."""
    from skimage.measure import label
    lab = label(~cad, connectivity=2)
    border = set(np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]])))
    return int(len(set(np.unique(lab)) - border - {0}))


def touches_overlay(overlay: np.ndarray | None, rc: tuple[int, int]) -> bool:
    return overlay is not None and bool(crop(overlay, rc).any())


def best_cal_tile(cad: np.ndarray, overlay: np.ndarray | None = None) -> tuple[int, int]:
    """Pre-registered scribe stand-in rule: the overlay-free tile with the most fully contained reference holes (ties: first)."""
    g = [rc for rc in tile_grid(cad.shape) if not touches_overlay(overlay, rc)]
    return max(g, key=lambda rc: (holes_inside(crop(cad, rc)), -g.index(rc)))


def rescale_image(im: RealImage, s: float) -> RealImage:
    """Arm B: resample by ``s`` (order 1, anti-aliased when downsampling); pixel size becomes px_nm / s."""
    a = sk_rescale(im.img, s, order=1, anti_aliasing=s < 1, preserve_range=True)
    ov = None if im.overlay is None else sk_rescale(im.overlay.astype(float), s, order=0, anti_aliasing=False) > 0
    return RealImage(im.image_id, im.group, im.path, im.px_nm / s, im.px_source + f"; rescaled x{s:.4f}", a,
                     im.design_d_nm, dict(im.meta, rescale=s), ov)
