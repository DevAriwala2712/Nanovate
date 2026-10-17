"""M2 topology tests: persistent homology, delta-Betti verdicts, Bug #4 regression, corner touches."""
import numpy as np
import pytest

from nanovate.config import TestbedCfg
from nanovate.modules.fields import build_field, field_calibrated, field_percentile, oracle_levels
from nanovate.modules.topology import (
    KILLER_BRIDGE,
    KILLER_OPEN_PARTICLE,
    NEAR_MISS,
    NUISANCE,
    betti_from_intervals,
    classify_verdict,
    compute_bottleneck_distances,
    compute_persistence_intervals,
    verify_topology,
)
from nanovate.testbed.dataset import make_sample, window
from nanovate.topo_utils import Window


def test_classify_verdict_ladder():
    """Verify AGENTS.md §5.2 verdict hierarchy."""
    # Bridge takes highest precedence
    assert classify_verdict((-1, 0)) == KILLER_BRIDGE
    assert classify_verdict((0, 1)) == KILLER_BRIDGE
    assert classify_verdict((-1, 1)) == KILLER_BRIDGE

    # Open / particle
    assert classify_verdict((1, 0)) == KILLER_OPEN_PARTICLE
    assert classify_verdict((0, -1)) == KILLER_OPEN_PARTICLE
    assert classify_verdict((1, -1)) == KILLER_OPEN_PARTICLE

    # Near-miss (requires d_beta == 0 and both thresholds met)
    assert classify_verdict((0, 0), d_bottleneck=0.25, sdf_energy=0.5, d_star=0.20, e_star=0.40) == NEAR_MISS
    assert classify_verdict((0, 0), d_bottleneck=0.15, sdf_energy=0.5, d_star=0.20, e_star=0.40) == NUISANCE
    assert classify_verdict((0, 0), d_bottleneck=0.25, sdf_energy=0.3, d_star=0.20, e_star=0.40) == NUISANCE

    # Default nuisance
    assert classify_verdict((0, 0)) == NUISANCE


def test_corner_touch_is_not_a_bridge():
    """Conductor 4-connectivity: a 1-pixel diagonal corner touch does not bridge."""
    cad = np.zeros((16, 16), bool)
    cad[3:6, 3:6] = True
    cad[6:9, 6:9] = True  # touching only at corner (6, 6)

    # In 4-connectivity, this is 2 separate components
    diag0, diag1 = compute_persistence_intervals(cad.astype(float))
    b0, b1 = betti_from_intervals(diag0, diag1, tau=0.45)
    assert b0 == 2
    assert b1 == 0


def test_hole_detection_in_conductor():
    """Dielectric 8-connectivity: an enclosed 1-cycle is captured in H1."""
    cad = np.zeros((20, 20), bool)
    cad[4:16, 4:16] = True
    cad[8:12, 8:12] = False  # hole inside

    diag0, diag1 = compute_persistence_intervals(cad.astype(float))
    b0, b1 = betti_from_intervals(diag0, diag1, tau=0.45)
    assert b0 == 1
    assert b1 == 1


def test_bug4_dim_bridge_regression():
    """Bug #4: fixed tau=0.35 misses a dim bridge; tau=0.45 on the calibrated field catches it.

    A bridge of conductor level c merges the wires at filtration 1 - c, so the two wires
    stay separate features iff 1 - c >= tau, i.e. a bridge is detected iff c > 1 - tau.
    For c = 0.6: lifetime 0.4 >= 0.35 (missed) but < 0.45 (detected). A c = 0.5 bridge would
    need tau > 0.5; the tau/fake-bridge trade-off is set on Manhattan val, not here.
    """
    # Two parallel wires at full contrast
    cad = np.zeros((32, 32), bool)
    cad[8:12, :] = True
    cad[20:24, :] = True

    # Image with a dim bridge (0.6 conductor contrast)
    img_field = cad.astype(float).copy()
    img_field[12:20, 14:18] = 0.6

    # 1. Buggy reference behavior: fixed tau=0.35 on percentile-like field
    diag0_bug, diag1_bug = compute_persistence_intervals(img_field)
    b0_bug, b1_bug = betti_from_intervals(diag0_bug, diag1_bug, tau=0.35)
    b0_cad, b1_cad = betti_from_intervals(*compute_persistence_intervals(cad.astype(float)), tau=0.35)
    d_beta_bug = (b0_bug - b0_cad, b1_bug - b1_cad)
    # The bug misses the bridge (reports (0, 0) -> NUISANCE)
    assert d_beta_bug == (0, 0)

    # 2. Correct calibrated rule: tau=0.45 detects the merge
    b0_fix, b1_fix = betti_from_intervals(diag0_bug, diag1_bug, tau=0.45)
    d_beta_fix = (b0_fix - b0_cad, b1_fix - b1_cad)
    assert d_beta_fix == (-1, 0)
    verdict, reason = verify_topology(
        img_field,
        cad,
        tau=0.45,
        field_option="calibrated",
        levels=(0.0, 1.0),
    )
    assert verdict == KILLER_BRIDGE
    assert reason["delta_betti"] == (-1, 0)


def test_wire_crossing_window_edge_gives_no_verdict_flip():
    """A wire crossing the margin boundary at an angle does not flip verdict."""
    win = Window(size=64, margin=8)
    cad = np.zeros((64, 64), bool)
    # Angled wire running from row 4 to row 60
    for r in range(4, 60):
        c = int(20 + 0.3 * r)
        cad[r, c:c + 4] = True

    verdict, reason = verify_topology(
        cad.astype(float),
        cad,
        win=win,
        tau=0.45,
        field_option="calibrated",
        levels=(0.0, 1.0),
    )
    assert verdict == NUISANCE
    assert reason["delta_betti"] == (0, 0)


@pytest.mark.parametrize("kind,expected_verdict", [
    ("nuisance", NUISANCE),
    ("bridge", KILLER_BRIDGE),
    ("open", KILLER_OPEN_PARTICLE),
    ("particle", KILLER_OPEN_PARTICLE),
])
def test_noiseless_defect_verdicts_on_manhattan(kind, expected_verdict):
    """Verify verdict on noise-free expected image for each defect class."""
    cfg = TestbedCfg()
    win = window(cfg)
    s = make_sample("manhattan", "val", 0, 0, kind, cfg, seed=42, shift=(0.0, 0.0))

    # Evaluate on the noise-free expected SEM image (in e-/px counts)
    verdict, reason = verify_topology(
        s.expected,
        s.cad,
        win=win,
        tau=0.45,
        field_option="calibrated",
        sem_cfg=cfg.sem,
    )
    assert verdict == expected_verdict, f"kind={kind}: got {verdict}, reason={reason}"
    assert reason["field_meta"]["levels_source"] == "oracle"
    assert reason["singleton"] is True
    assert reason["near_miss_enabled"] is False


def test_ler_only_sample_gives_nuisance():
    """Clean wafer perturbed only by LER and grain gives NUISANCE verdict."""
    cfg = TestbedCfg()
    win = window(cfg)
    for lay_id in range(3):
        s = make_sample("manhattan", "val", lay_id, 0, "nuisance", cfg, seed=10 + lay_id, shift=(0.0, 0.0))
        verdict, reason = verify_topology(
            s.expected,
            s.cad,
            win=win,
            tau=0.45,
            field_option="calibrated",
            sem_cfg=cfg.sem,
        )
        assert verdict == NUISANCE, f"layout {lay_id} gave {verdict}: {reason}"
        assert reason["delta_betti"] == (0, 0)


def test_bottleneck_distance_calculation():
    """Verify bottleneck distance is computed and sensitive to perturbation."""
    cad = np.zeros((32, 32), bool)
    cad[10:14, :] = True
    cad[20:24, :] = True

    # Slightly perturbed image
    img = cad.astype(float).copy()
    img[10:14, 10:15] = 0.7  # shallow notch

    diag0_c, diag1_c = compute_persistence_intervals(cad.astype(float))
    diag0_i, diag1_i = compute_persistence_intervals(img)
    d0, d1 = compute_bottleneck_distances(diag0_i, diag1_i, diag0_c, diag1_c)
    assert d0 > 0.0
    assert d1 == 0.0


def test_near_miss_branch():
    """Verify that NEAR_MISS fires when delta_beta == 0 and both bottleneck & SDF meet threshold."""
    cad = np.zeros((32, 32), bool)
    cad[10:14, :] = True

    # Image with small perturbation (delta_beta == 0)
    img = cad.astype(float).copy()
    img[10:14, 10:12] = 0.6

    # With high thresholds -> NUISANCE
    v1, r1 = verify_topology(
        img, cad, tau=0.45, field_option="calibrated", levels=(0.0, 1.0),
        d_star=0.5, e_star=0.5, sdf_energy=0.2,
    )
    assert v1 == NUISANCE

    # With met thresholds -> NEAR_MISS
    v2, r2 = verify_topology(
        img, cad, tau=0.45, field_option="calibrated", levels=(0.0, 1.0),
        d_star=0.1, e_star=0.1, sdf_energy=0.2,
    )
    assert v2 == NEAR_MISS
    assert r2["near_miss_enabled"] is True
    assert r2["max_sdf_energy"] == 0.2
