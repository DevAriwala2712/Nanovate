"""B2 fast Betti path: exact on binary fields (where single-level Betti = lifetime-counted bars), GUDHI convention."""
import numpy as np
import pytest
from scipy import ndimage as ndi

from nanovate.modules.fast_betti import betti_at_level, fast_delta_beta
from nanovate.modules.topology import betti_from_intervals, compute_persistence_intervals
from nanovate.topo_utils import betti_binary


@pytest.mark.parametrize("seed", range(20))
def test_matches_gudhi_and_betti_binary_on_random_binary_fields(seed):
    rng = np.random.default_rng(seed)
    m = ndi.binary_opening(rng.random((64, 64)) < 0.55, iterations=1) if seed % 2 else rng.random((40, 40)) < 0.5
    assert betti_at_level(m.astype(float), 0.5) == betti_binary(m)
    d0, d1 = compute_persistence_intervals(m.astype(float))
    assert betti_at_level(m.astype(float), 0.5) == betti_from_intervals(d0, d1, 0.35)


def test_conventions_diagonal_conductor_does_not_connect_diagonal_dielectric_does():
    m = np.zeros((6, 6), bool)
    m[2, 2] = m[3, 3] = True                     # diagonal conductor pixels: two components
    assert betti_at_level(m.astype(float), 0.5) == (2, 0)
    ring = np.ones((7, 7), bool)
    ring[0], ring[-1], ring[:, 0], ring[:, -1] = False, False, False, False
    ring[2, 2] = ring[3, 3] = False              # diagonal dielectric pixels inside: one hole
    assert betti_at_level(ring.astype(float), 0.5) == (1, 1)


def test_fast_delta_beta_bridge_and_open_on_binary_field():
    cad = np.zeros((64, 64), bool)
    cad[:, 10:20] = cad[:, 30:40] = True
    reg = np.ones_like(cad)
    br = cad.copy(); br[30:34, 20:30] = True
    op = cad.copy(); op[30:34, 10:20] = False
    assert fast_delta_beta(cad.astype(float), cad, reg, 0.5) == (0, 0)
    assert fast_delta_beta(br.astype(float), cad, reg, 0.5) == (-1, 0)
    assert fast_delta_beta(op.astype(float), cad, reg, 0.5) == (1, 0)


def test_single_level_differs_from_lifetime_count_on_a_dim_component():
    """Documents why B2 cannot be exact in general: one level cannot see every bar that lives >= tau."""
    f = np.zeros((32, 32))
    f[5:10, 5:10] = 1.0
    f[20:25, 20:25] = 0.6                        # born at F = 0.4, merges at F = 1: lifetime 0.6 >= tau, counted
    d0, d1 = compute_persistence_intervals(f)
    assert betti_from_intervals(d0, d1, 0.35)[0] == 2
    assert betti_at_level(f, 0.7)[0] == 1        # a level above the blob misses it
    assert betti_at_level(f, 0.5)[0] == 2        # a lower level sees it (but would see other things too)
