"""M5 / Module 2: aligned same-layout stack, rank-2 randomized SVD, per-pixel MAD soft threshold."""
import numpy as np
import pytest
from scipy import ndimage as ndi

from nanovate.config import TestbedCfg
from nanovate.modules.calibration import calibrate
from nanovate.modules.lowrank import (
    StackResult,
    denoise_singleton,
    denoise_stack,
    hard_threshold,
    randomized_svd,
    soft_threshold,
)
from nanovate.testbed.dataset import get_layout, make_sample
from nanovate.testbed.scribe import make_scribe_patches

K = 32


def test_randomized_svd_matches_full_svd_on_rank2_part():
    rng = np.random.default_rng(0)
    A = rng.normal(size=(32, 2)) @ rng.normal(size=(2, 4000)) * 5 + rng.normal(size=(32, 4000)) * 0.1
    U, s, Vt = randomized_svd(A, rank=2, oversample=2, n_iter=2, seed=0)
    Uf, sf, Vtf = np.linalg.svd(A, full_matrices=False)
    L_r = (U * s) @ Vt
    L_f = (Uf[:, :2] * sf[:2]) @ Vtf[:2]
    assert np.linalg.norm(L_r - L_f) / np.linalg.norm(L_f) < 1e-3
    assert np.allclose(s, sf[:2], rtol=1e-3)


def test_randomized_svd_is_seeded():
    A = np.random.default_rng(1).normal(size=(32, 3000))
    a = randomized_svd(A, seed=7)
    b = randomized_svd(A, seed=7)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))


@pytest.fixture(scope="module")
def world():
    cfg = TestbedCfg()
    cal = calibrate(make_scribe_patches(cfg, seed=0))
    layout = get_layout("manhattan", "val", 2, cfg, 11)
    kinds = ["nuisance"] * K
    kinds[3] = "bridge"
    samples = [make_sample("manhattan", "val", 2, k, kind, cfg, 11, layout=layout) for k, kind in enumerate(kinds)]
    return cfg, cal, layout, samples


def test_aligned_stack_residual_far_below_unaligned(world):
    """±8 px shifts (Exp-4 range), noise-free images so the metric sees structure, not shot noise.
    Measured: unaligned 0.211, aligned 0.110, oracle (true shifts undone) 0.111."""
    cfg, cal, layout, _ = world
    rng = np.random.default_rng(0)
    shifts = [tuple(rng.uniform(-8, 8, 2)) for _ in range(K)]
    imgs = [make_sample("manhattan", "val", 2, k, "nuisance", cfg, 11, layout=layout, shift=sh).expected
            for k, sh in enumerate(shifts)]
    al = denoise_stack(imgs, layout.cad, cal, align=True, seed=0)
    un = denoise_stack(imgs, layout.cad, cal, align=False, seed=0)
    oracle = denoise_stack([ndi.shift(i, (-sh[0], -sh[1]), order=3, mode="nearest") for i, sh in zip(imgs, shifts)],
                           layout.cad, cal, align=False, seed=0)
    assert isinstance(al, StackResult)
    assert al.residual_energy < 0.6 * un.residual_energy, (al.residual_energy, un.residual_energy)
    assert al.residual_energy < 1.05 * oracle.residual_energy


def test_injected_defect_is_in_its_own_die_only(world):
    cfg, cal, layout, samples = world
    res = denoise_stack([s.image for s in samples], layout.cad, cal, seed=0)
    m = samples[3].mask & ~samples[3].cad  # bridge pixels added to dielectric (CAD frame)
    assert m.sum() > 10
    energy = np.abs(res.S[:, m]).mean(axis=1)
    others = np.delete(energy, 3)
    assert energy[3] > 3 * np.median(others), (energy[3], np.median(others))
    assert energy[3] > others.max()


def test_same_seed_identical_and_flags(world):
    cfg, cal, layout, samples = world
    imgs = [s.image for s in samples[:12]]
    a = denoise_stack(imgs, layout.cad, cal, seed=5)
    b = denoise_stack(imgs, layout.cad, cal, seed=5)
    assert np.array_equal(a.L, b.L) and np.array_equal(a.S, b.S)
    assert a.singleton is False and a.rank == 2 and a.K == 12


def test_stack_field_is_less_noisy_than_raw(world):
    cfg, cal, layout, samples = world
    res = denoise_stack([s.image for s in samples], layout.cad, cal, seed=0)
    k = 10
    # compare against the die's own noise-free image (same shift => same frame as the *aligned* field only
    # after alignment, so compare noise via a high-pass: neighbour-difference std)
    raw_hp = np.std(np.diff(samples[k].image, axis=1))
    fld_hp = np.std(np.diff(res.field[k], axis=1))
    assert fld_hp < 0.5 * raw_hp


def test_singleton_path_is_flagged_and_denoises(world):
    cfg, cal, layout, samples = world
    img = samples[0].image
    out = denoise_singleton(img)
    assert out.shape == img.shape
    assert np.std(np.diff(out, axis=1)) < 0.6 * np.std(np.diff(img, axis=1))
    res = denoise_stack([img], layout.cad, cal)
    assert res.singleton is True and res.K == 1
    assert res.field.shape == (1,) + img.shape


def test_soft_shrinks_hard_keeps_amplitude():
    R = np.array([[5.0, -5.0, 1.0, -1.0]])
    tau = np.array([[2.0, 2.0, 2.0, 2.0]])
    assert np.allclose(soft_threshold(R, tau), [[3.0, -3.0, 0.0, 0.0]])
    assert np.allclose(hard_threshold(R, tau), [[5.0, -5.0, 0.0, 0.0]])


def _flat_stack_with_blob(amp_sigma, seed=0):
    """K flat noisy images; die 0 carries a 5x5 blob of amplitude amp_sigma (in noise sigmas)."""
    rng = np.random.default_rng(seed)
    D = 50.0 + rng.normal(0, 1.0, (K, 64, 64))
    D[0, 30:35, 30:35] += amp_sigma
    return D


def test_residual_smoothing_is_a_matched_filter_for_coherent_defects():
    """A 5x5 blob at 1.5 sigma is invisible per pixel at k=3 but found after sigma=1 smoothing."""
    from nanovate.modules import lowrank

    D = _flat_stack_with_blob(1.5)
    f = lowrank.sparse_part(D, rank=2, k=3.0, seed=0, shrink="hard", resid_smooth=0.0)
    g = lowrank.sparse_part(D, rank=2, k=3.0, seed=0, shrink="hard", resid_smooth=1.0)
    blob = (slice(30, 35), slice(30, 35))
    hit_plain = (np.abs(f[0][blob]) > 0).mean()
    hit_smooth = (np.abs(g[0][blob]) > 0).mean()
    assert hit_plain < 0.25 and hit_smooth > 0.5 and hit_smooth > 3 * hit_plain, (hit_plain, hit_smooth)
    # and it does not flood the other dies with speckle
    assert (np.abs(g[1:]) > 0).mean() < 0.01


def test_support_mode_detects_with_smoothing_but_keeps_raw_amplitude():
    """'support': the smoothed residual decides *where*, the unsmoothed residual gives *how much*."""
    from nanovate.modules import lowrank

    D = _flat_stack_with_blob(3.0)
    S_sup = lowrank.sparse_part(D, k=3.0, seed=0, shrink="support", resid_smooth=1.0)
    S_soft = lowrank.sparse_part(D, k=3.0, seed=0, shrink="soft", resid_smooth=1.0)
    blob = (slice(31, 34), slice(31, 34))
    assert S_sup[0][blob].mean() > 0.8 * 3.0           # amplitude preserved (blob is +3 sigma)
    assert S_sup[0][blob].mean() > 1.3 * S_soft[0][blob].mean()
    assert (np.abs(S_sup[1:]) > 0).mean() < 0.01        # no speckle flood in the other dies
