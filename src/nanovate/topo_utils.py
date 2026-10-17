"""Shared topology conventions (used by testbed labels AND the Module-4 verifier).

Connectivity convention (pinned by tests):
  * conductor is 4-connected (a diagonal corner touch is NOT an electrical contact);
  * dielectric is 8-connected (dual connectivity), so holes (beta_1) are 8-connected
    dielectric regions that do not touch the window border.
GUDHI's cubical complex built from ``vertices=`` reproduces exactly this convention
for the sub-level sets of (1 - P); ``top_dimensional_cells=`` would give 8-connected
conductor and is therefore not used.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.measure import label as sk_label


@dataclass(frozen=True)
class Window:
    """Central comparison window: rows/cols [margin, size - margin) in px."""

    size: int = 256
    margin: int = 16

    @property
    def sl(self) -> tuple[slice, slice]:
        return (slice(self.margin, self.size - self.margin),) * 2

    def crop(self, a: np.ndarray) -> np.ndarray:
        return a[self.sl]


def betti_binary(mask: np.ndarray) -> tuple[int, int]:
    """Exact (beta_0, beta_1) of a binary conductor mask under the shared convention."""
    m = mask.astype(bool)
    b0 = int(sk_label(m, connectivity=1).max())
    diel = sk_label(~m, connectivity=2)
    n = int(diel.max())
    if n == 0:
        return b0, 0
    border = np.unique(np.concatenate([diel[0], diel[-1], diel[:, 0], diel[:, -1]]))
    b1 = n - int(np.count_nonzero(border))
    return b0, b1


def relevance_region(cad: np.ndarray, win: Window, tol: int = 4) -> np.ndarray:
    r"""Region in which features are counted, built from CAD only (no labels needed).

    R* = (W_c  U  (dilate(CAD_in, tol) n grow(W_c, tol)))  \  dilate(CAD_out, tol)
    where W_c is the central window, CAD_in / CAD_out are CAD components that do / do not
    intersect W_c. A feature therefore cannot enter or leave the counting region through
    an edge displacement smaller than ``tol`` px (LER, small residual misalignment), and
    wire fragments created at the outer tile edge (< margin - tol from it) are never counted.
    """
    c = cad.astype(bool)
    inner = np.zeros_like(c)
    inner[win.sl] = True
    lab = sk_label(c, connectivity=1)
    ids_in = np.unique(lab[inner & c])
    ids_in = ids_in[ids_in > 0]
    cad_in = np.isin(lab, ids_in)
    cad_out = c & ~cad_in
    grown = np.zeros_like(c)
    m = max(0, win.margin - tol)
    grown[m:c.shape[0] - m, m:c.shape[1] - m] = True
    disk = ndi.generate_binary_structure(2, 1)
    d_in = ndi.binary_dilation(cad_in, disk, iterations=tol) & grown
    d_out = ndi.binary_dilation(cad_out, disk, iterations=tol)

    # Outside CAD holes (enclosed holes not intersecting inner) must not enter reg via tol-roughness
    diel = sk_label(~c, connectivity=2)
    border = np.unique(np.concatenate([diel[0], diel[-1], diel[:, 0], diel[:, -1]]))
    all_holes = np.setdiff1d(np.unique(diel[~c]), border)
    holes_in_ids = np.unique(diel[inner & ~c])
    holes_in_ids = np.setdiff1d(holes_in_ids[holes_in_ids > 0], border)
    holes_out_ids = np.setdiff1d(all_holes[all_holes > 0], holes_in_ids)
    holes_out = np.isin(diel, holes_out_ids)
    d_out_holes = ndi.binary_dilation(holes_out, disk, iterations=tol)

    return (inner | d_in) & ~d_out & ~d_out_holes


def betti_local(mask: np.ndarray, win: Window, region: np.ndarray | None = None) -> tuple[int, int]:
    r"""Margin-aware Betti numbers of a tile-sized binary mask (AGENTS.md §7.5).

    Topology is computed on the whole tile, but only features reaching ``region``
    (default: the central window; use ``relevance_region(cad)`` to compare against CAD) count:
      beta_0 = # 4-connected conductor components intersecting the region;
      beta_1 = # 8-connected dielectric components not touching the tile border
               (enclosed holes) that intersect the region.
    A plain crop would fragment wires running parallel to the crop edge (LER makes the
    edge weave in and out of the window), producing spurious components.
    """
    m = mask.astype(bool)
    inner = np.zeros_like(m)
    inner[win.sl] = True
    reg = inner if region is None else region
    lab = sk_label(m, connectivity=1)
    ids = np.unique(lab[reg & m])
    b0 = int(np.count_nonzero(ids))
    diel = sk_label(~m, connectivity=2)
    border = np.unique(np.concatenate([diel[0], diel[-1], diel[:, 0], diel[:, -1]]))
    holes_in = np.setdiff1d(np.unique(diel[reg & ~m]), border)
    holes_in = holes_in[holes_in > 0]
    return b0, int(holes_in.size)


def delta_betti(w: np.ndarray, cad: np.ndarray, win: Window) -> tuple[int, int]:
    """(d_beta0, d_beta1) of tile-sized wafer ``w`` vs ``cad``; same CAD-derived region."""
    reg = relevance_region(cad, win)
    bw = betti_local(w, win, reg)
    bc = betti_local(cad, win, reg)
    return bw[0] - bc[0], bw[1] - bc[1]


def topo_class(d: tuple[int, int]) -> str:
    """Map a Betti difference to a topology class.

    'none'   : d_beta == (0, 0)
    'bridge' : merges only (d0 <= 0, d1 >= 0, not both zero)
    'open'   : splits only (d0 >= 0, d1 <= 0, not both zero)
    'mixed'  : anything else
    """
    d0, d1 = d
    if d0 == 0 and d1 == 0:
        return "none"
    if d0 <= 0 and d1 >= 0:
        return "bridge"
    if d0 >= 0 and d1 <= 0:
        return "open"
    return "mixed"
