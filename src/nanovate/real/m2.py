"""Real-wiring false-alarm test (Track C): metal-2 SEM images of a commercial 128 nm IC with labelled tracks and vias.

Data: doi:10.17617/3.HY5SYN (CC-BY-4.0), 4096 x 3536 px, 14.65 nm/px (stated by the authors), BSE at 15 keV.
The reference is the authors' LABEL raster (tracks = SVG polygon paths, vias = circles; both conductor), which is a
validated extraction of the real layout, not the designer's CAD. Labels can miss tracks that span several images,
so ``unlabelled_conductor`` flags bright structures with no label nearby (a label-completeness QC that never looks
at the verifier). Units: px of the image unless the name says nm.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.draw import disk, polygon
from skimage.measure import label

PX_NM = 14.65
ROOT = Path("data/real/metal/extracted/dataset")


def ids() -> list[str]:
    """Image ids with both an M2 image and an M2 label file, sorted."""
    sems = {p.stem[3:] for p in (ROOT / "sems" / "m2").glob("sem*.png")}
    labs = {p.stem[5:] for p in (ROOT / "labels" / "m2").glob("label*.svg")}
    return sorted(sems & labs)


def load_image(i: str) -> np.ndarray:
    a = np.asarray(Image.open(ROOT / "sems" / "m2" / f"sem{i}.png"))
    return (a if a.ndim == 2 else a[..., 0]).astype(float)


def rasterize_svg(text: str, shape: tuple[int, int]) -> np.ndarray:
    """Conductor raster from the label SVG: filled track paths plus via circles."""
    H, W = shape
    out = np.zeros((H, W), bool)
    for d in re.findall(r'<path d="([^"]+)"', text):
        pts = np.array([[float(a), float(b)] for a, b in re.findall(r"(-?[\d.]+),(-?[\d.]+)", d)])
        if len(pts) >= 3:
            rr, cc = polygon(pts[:, 1], pts[:, 0], (H, W))
            out[rr, cc] = True
    for cx, cy, r in re.findall(r'<circle cx="(-?[\d.]+)" cy="(-?[\d.]+)" r="([\d.]+)"', text):
        rr, cc = disk((float(cy), float(cx)), float(r), shape=(H, W))
        out[rr, cc] = True
    return out


def load_label(i: str, shape: tuple[int, int]) -> np.ndarray:
    return rasterize_svg((ROOT / "labels" / "m2" / f"label{i}.svg").read_text(), shape)


def unlabelled_conductor(img: np.ndarray, lab: np.ndarray, smooth: float = 2.0, dilate: int = 5, min_area: int = 200) -> np.ndarray:
    """Label-completeness QC (never looks at the verifier): pixels brighter than the midpoint between the median smoothed
    intensity inside the labels and the median outside the labels dilated by ``dilate`` px, lying outside that dilated
    label, in connected components of >= ``min_area`` px (an unlabelled track or large debris). Returns their bool mask."""
    sm = ndi.gaussian_filter(img, smooth)
    near = ndi.binary_dilation(lab, iterations=dilate)
    thr = 0.5 * (np.median(sm[lab]) + np.median(sm[~near]))
    outside = (sm > thr) & ~near
    lb = label(outside, connectivity=2)
    if lb.max() == 0:
        return np.zeros_like(lab)
    area = np.bincount(lb.ravel())
    keep = area >= min_area
    keep[0] = False
    return keep[lb]
