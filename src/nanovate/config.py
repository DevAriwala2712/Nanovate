"""Dataclass configs. Every run saves its full config next to its results.

Units: lengths in nm (= px, at 1 nm/px), dose in e-/px, intensities in e- counts.
Defaults for the simulator were fixed in M1, before any experiment was run.
"""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class LayoutCfg:
    """Geometry of the three layout families (nm)."""

    tile: int = 256                 # tile edge, px (1 nm/px)
    pad: int = 24                   # extra context rendered around the tile (for shifts), px
    margin: int = 16                # central-window margin used for all topology comparisons, px
    # A: Manhattan lines
    man_half_pitch: float = 14.0
    man_cut_prob: float = 0.35      # prob. a line carries a designed line-end cut
    man_cut_gap: tuple[float, float] = (16.0, 24.0)
    man_strap_prob: float = 0.30    # prob. an adjacent-line pair is joined by a strap (makes loops)
    # B: curvy spline wires
    curvy_width: tuple[float, float] = (14.0, 18.0)
    curvy_pitch: float = 36.0
    curvy_amp: tuple[float, float] = (4.0, 12.0)
    curvy_wavelength: tuple[float, float] = (120.0, 220.0)
    # C: via / pillar arrays with rails
    via_size: tuple[float, float] = (14.0, 18.0)
    via_pitch: float = 36.0
    via_round_prob: float = 0.5
    rail_width: tuple[float, float] = (18.0, 24.0)
    min_space: float = 10.0         # layouts whose min conductor spacing < this are regenerated


@dataclass
class NuisanceCfg:
    """Harmless perturbations of CAD -> wafer W."""

    ler_sigma: float = 1.0          # nm, OU std along edge normals
    ler_xi: float = 20.0            # nm, OU correlation length
    ler_step: float = 0.5           # nm, contour resampling step
    ler_aa: float = 1.0             # nm, anti-alias low-pass of the OU along arc (raster is 1 nm/px);
                                    # the filtered process is rescaled to keep std = ler_sigma
    min_speck: int = 3              # px; LER-only conductor islands / pinholes <= this are removed
    grain_contrast: float = 0.15    # fraction, grain offsets ~ U(-a, a) on conductor yield
    grain_size: float = 15.0        # nm, mean Voronoi cell spacing


@dataclass
class DefectCfg:
    """Defect injection parameters (nm)."""

    bridge_radius: tuple[float, float] = (3.0, 10.0)
    bridge_max_gap: float = 25.0
    open_length: tuple[float, float] = (3.0, 8.0)        # void extent along the trace
    particle_radius: tuple[float, float] = (3.0, 6.0)
    notch_remaining: tuple[int, int] = (2, 4)            # px of wire left after a notch
    notch_length: tuple[float, float] = (4.0, 10.0)
    filament_radius: tuple[float, float] = (2.0, 4.0)
    filament_gap: tuple[int, int] = (1, 3)               # nm short of the neighbour
    max_resample: int = 50


@dataclass
class SEMCfg:
    """SEM forward model. Deliberately NOT in the Module-1 feature span (PLAN R1):
    anisotropic beam, saturating bloom with its own edge scale, multiplicative grain."""

    dose: float = 50.0              # e-/px
    a0: float = 0.25                # dielectric SE yield
    a1: float = 0.45                # conductor - dielectric yield
    a2: float = 0.9                 # edge-bloom gain
    bloom_sat: float = 0.35         # bloom saturation: b -> s*tanh(b/s)
    sigma_edge: float = 1.3         # nm, edge-bloom smoothing scale
    sigma_beam: tuple[float, float] = (1.6, 1.2)   # nm (row, col): scan-direction anisotropy
    charge_amp: float = 0.06        # max |polynomial charging ramp| (yield units)
    read_noise: float = 1.0         # e-, Gaussian read noise std
    max_shift: float = 3.0          # px, random acquisition shift per patch (uniform +-)


@dataclass
class TestbedCfg:
    __test__ = False  # not a pytest class
    layout: LayoutCfg = field(default_factory=LayoutCfg)
    nuisance: NuisanceCfg = field(default_factory=NuisanceCfg)
    defect: DefectCfg = field(default_factory=DefectCfg)
    sem: SEMCfg = field(default_factory=SEMCfg)
    K: int = 32                     # dies per layout stack


def to_dict(cfg: Any) -> dict:
    return cfg if isinstance(cfg, dict) else dataclasses.asdict(cfg)


def save_config(cfg: Any, path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(to_dict(cfg), indent=2, sort_keys=True))
