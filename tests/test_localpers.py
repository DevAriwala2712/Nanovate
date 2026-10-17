"""Region-local union-find persistence: agrees with GUDHI globally and with betti_local on masks."""
import dataclasses

import gudhi
import numpy as np
import pytest
from scipy import ndimage as ndi

from nanovate.config import TestbedCfg
from nanovate.modules.localpers import local_persistence
from nanovate.testbed.dataset import get_layout, make_sample, window
from nanovate.topo_utils import betti_local, relevance_region


def _gudhi_pairs(P: np.ndarray):
    cc = gudhi.CubicalComplex(vertices=np.clip(1.0 - P, 0, 1))
    cc.compute_persistence()
    i0 = np.asarray(cc.persistence_intervals_in_dimension(0), dtype=float).reshape(-1, 2)
    i1 = np.asarray(cc.persistence_intervals_in_dimension(1), dtype=float).reshape(-1, 2)
    return i0[np.isfinite(i0[:, 1])], i1


def _sorted(a: np.ndarray) -> np.ndarray:
    return a[np.lexsort((a[:, 1], a[:, 0]))] if len(a) else a


@pytest.mark.parametrize("seed", range(6))
@pytest.mark.parametrize("quantise", [False, True])
def test_matches_gudhi_intervals(seed, quantise):
    rng = np.random.default_rng(seed)
    P = ndi.gaussian_filter(rng.random((28, 28)), 1.2)
    P = (P - P.min()) / (P.max() - P.min())
    if quantise:  # many exact ties, like a binary CAD
        P = np.round(P * 4) / 4
    lp = local_persistence(P, np.ones(P.shape, bool))
    g0, g1 = _gudhi_pairs(P)
    assert np.allclose(_sorted(lp.h0), _sorted(g0), atol=1e-12), "H0 pairs differ from GUDHI"
    assert np.allclose(_sorted(lp.h1), _sorted(g1), atol=1e-12), "H1 pairs differ from GUDHI"


def test_corner_touch_is_not_a_bridge():
    cad = np.zeros((16, 16), bool)
    cad[3:6, 3:6] = True
    cad[6:9, 6:9] = True
    assert local_persistence(cad.astype(float), np.ones_like(cad)).betti(0.45) == (2, 0)


def test_enclosed_hole_counted_and_border_dielectric_is_not():
    cad = np.zeros((20, 20), bool)
    cad[4:16, 4:16] = True
    cad[8:12, 8:12] = False
    assert local_persistence(cad.astype(float), np.ones_like(cad)).betti(0.45) == (1, 1)


def test_feature_outside_region_is_not_counted():
    m = np.zeros((32, 32), bool)
    m[4:8, 4:28] = True    # inside region
    m[24:28, 4:28] = True  # outside region
    reg = np.zeros_like(m)
    reg[:16] = True
    assert local_persistence(m.astype(float), reg).betti(0.45) == (1, 0)
    assert local_persistence(m.astype(float), np.ones_like(m)).betti(0.45) == (2, 0)


@pytest.mark.parametrize("kind", ["nuisance", "bridge", "open", "particle", "notch", "filament"])
def test_binary_counts_equal_betti_local_on_testbed(kind):
    cfg = TestbedCfg()
    cfg = dataclasses.replace(cfg, nuisance=dataclasses.replace(cfg.nuisance, ler_sigma=1.5))
    win = window(cfg)
    for lay in range(3):
        layout = get_layout("manhattan", "val", lay, cfg, 5)
        s = make_sample("manhattan", "val", lay, 0, kind, cfg, 5, layout=layout)
        reg = relevance_region(s.cad, win)
        for mask in (s.cad, s.W):
            assert local_persistence(mask.astype(float), reg).betti(0.45) == betti_local(mask, win, reg), (
                f"{kind} layout {lay}"
            )
