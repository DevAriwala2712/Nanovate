"""Track C Rung 3 helpers for the Puschner et al. (IEEE S&P 2023) 90 nm dataset (doi:10.17617/3.396Q7I, CC-BY-4.0).

Backside SEM tiles of a real 90 nm CMOS chip plus its ABSTRACTED, trojanized GDSII: only standard-cell bounding boxes
with anonymised type ids ("FILL_" = filler); 10 cells were swapped for different same-size cells (positions withheld).
There is no polygon CAD, so the verifier runs in reference-cell mode: a cell instance is compared with the median of
other instances of the same type and orientation (Phase 4's reference-die idea at cell level).
Units: GDS in um; image coordinates in px of the stitched mosaic (MIST global positions).
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

EXTRACTED = Path("data/real/puschner/extracted/90nm")
POS_RE = re.compile(r"file: (\S+); corr: (-?[\d,.]+); position: \((-?\d+), (-?\d+)\); grid: \((\d+), (\d+)\);")


def parse_positions(path: str | Path) -> pd.DataFrame:
    """MIST global-positions file -> DataFrame(file, corr, x, y, col, row); x/y = tile top-left in mosaic px."""
    rows = []
    for line in Path(path).read_text().splitlines():
        m = POS_RE.search(line)
        if m:
            f, c, x, y, col, row = m.groups()
            rows.append(dict(file=f, corr=float(c.replace(",", ".")), x=int(x), y=int(y), col=int(col), row=int(row)))
    return pd.DataFrame(rows)


def gds_instances(path: str | Path) -> pd.DataFrame:
    """One row per placed cell: type, filler flag, orientation code, placed bounding box (um)."""
    import gdstk
    lib = gdstk.read_gds(str(path))
    top = lib.top_level()[0]
    boxes = {c.name: np.array(c.bounding_box()) for c in lib.cells if c.bounding_box() is not None}
    rows = []
    for r in top.references:
        name = r.cell.name
        b = boxes[name]                                   # [[x0, y0], [x1, y1]] in cell coordinates
        corners = np.array([[b[0, 0], b[0, 1]], [b[1, 0], b[0, 1]], [b[1, 0], b[1, 1]], [b[0, 0], b[1, 1]]])
        if r.x_reflection:
            corners[:, 1] = -corners[:, 1]
        th = r.rotation or 0.0
        R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
        p = corners @ R.T + np.asarray(r.origin)
        rows.append(dict(cell_type=name, filler="FILL_" in name, rot=int(round(np.degrees(th))) % 360, mirror=bool(r.x_reflection),
                         x0=p[:, 0].min(), y0=p[:, 1].min(), x1=p[:, 0].max(), y1=p[:, 1].max()))
    df = pd.DataFrame(rows)
    df["orient"] = df.rot.astype(str) + np.where(df.mirror, "m", "")
    return df


@dataclass(frozen=True)
class Affine:
    """mosaic_px = A @ gds_um + t."""

    A: tuple
    t: tuple

    def apply(self, xy: np.ndarray) -> np.ndarray:
        return np.asarray(xy, float) @ np.asarray(self.A, float).T + np.asarray(self.t, float)


# --------------------------------------------------------------------------- registration (GDS um -> mosaic px)

def read_tile(zf, name: str) -> np.ndarray:
    """One tile from the (unextracted) 90 nm zip as a float grey image."""
    import io

    import tifffile
    member = next(m for m in zf.namelist() if m.endswith(name))
    with zf.open(member) as f:
        a = tifffile.imread(io.BytesIO(f.read()))
    return (a if a.ndim == 2 else a[..., 0]).astype(np.float32)


def low_res_mosaic(zip_path: str | Path, pos: pd.DataFrame, factor: int, tile_px: int) -> np.ndarray:
    """Block-mean downsampled mosaic (later tiles overwrite the 10% overlaps). Shape ~ (max y + tile)/factor."""
    import zipfile
    H = int((pos.y.max() + tile_px) // factor) + 1
    W = int((pos.x.max() + tile_px) // factor) + 1
    M = np.zeros((H, W), np.float32)
    with zipfile.ZipFile(zip_path) as zf:
        for r in pos.itertuples():
            a = read_tile(zf, r.file)
            h, w = (a.shape[0] // factor) * factor, (a.shape[1] // factor) * factor
            s = a[:h, :w].reshape(h // factor, factor, w // factor, factor).mean((1, 3))
            y0, x0 = r.y // factor, r.x // factor
            M[y0:y0 + s.shape[0], x0:x0 + s.shape[1]] = s[: H - y0, : W - x0]
    return M


def density_map(inst: pd.DataFrame, um_per_px: float, shape: tuple[int, int], origin_um: tuple[float, float]) -> np.ndarray:
    """Functional (non-filler) cell coverage on a low-res grid: 1 inside functional cells, 0 elsewhere (GDS y up)."""
    D = np.zeros(shape, np.float32)
    f = inst[~inst.filler]
    x0 = ((f.x0.values - origin_um[0]) / um_per_px).astype(int); x1 = np.ceil((f.x1.values - origin_um[0]) / um_per_px).astype(int)
    y0 = ((f.y0.values - origin_um[1]) / um_per_px).astype(int); y1 = np.ceil((f.y1.values - origin_um[1]) / um_per_px).astype(int)
    for a, b, c, d in zip(y0, y1, x0, x1):
        D[max(a, 0):min(b, shape[0]), max(c, 0):min(d, shape[1])] = 1.0
    return D
