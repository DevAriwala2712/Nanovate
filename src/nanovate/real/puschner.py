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
