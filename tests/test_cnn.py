"""M6 baseline: small ResNet-style CNN on [image, CAD], Manhattan-trained."""
import numpy as np
import pytest
import torch

from nanovate.baselines.cnn import (
    CnnCfg,
    ResNetSmall,
    load_cnn,
    pick_threshold,
    predict_proba,
    prepare,
    save_cnn,
    train_cnn,
)


def _toy(n=8, seed=0, size=256):
    rng = np.random.default_rng(seed)
    imgs = rng.poisson(30, (n, size, size)).astype(np.float32)
    cads = rng.random((n, size, size)) > 0.5
    labels = np.array([i % 2 for i in range(n)], bool)
    return imgs, cads, labels


def test_forward_shape():
    net = ResNetSmall()
    out = net(torch.zeros(4, 2, 256, 256))
    assert out.shape == (4,)
    assert sum(p.numel() for p in net.parameters()) < 500_000  # "small"


def test_prepare_standardises_image_channel_and_keeps_cad():
    imgs, cads, _ = _toy(2)
    x = prepare(imgs, cads)
    assert x.shape == (2, 2, 256, 256) and x.dtype == torch.float32
    assert abs(float(x[0, 0].mean())) < 1e-4 and abs(float(x[0, 0].std()) - 1) < 1e-3
    assert torch.equal(x[:, 1] > 0.5, torch.from_numpy(cads))


def test_two_seeded_runs_have_identical_first_batch_loss_and_weights():
    imgs, cads, labels = _toy(8)
    cfg = CnnCfg(epochs=2, batch=4, seed=3)
    m1, h1 = train_cnn(imgs, cads, labels, cfg)
    m2, h2 = train_cnn(imgs, cads, labels, cfg)
    assert h1["first_batch_loss"] == h2["first_batch_loss"]
    assert all(torch.equal(a, b) for a, b in zip(m1.state_dict().values(), m2.state_dict().values()))
    m3, h3 = train_cnn(imgs, cads, labels, CnnCfg(epochs=2, batch=4, seed=4))
    assert h3["first_batch_loss"] != h1["first_batch_loss"]


def test_overfits_a_tiny_batch():
    imgs, cads, labels = _toy(8, seed=1)
    model, hist = train_cnn(imgs, cads, labels, CnnCfg(epochs=40, batch=8, seed=0, lr=3e-3, augment=False))
    p = predict_proba(model, imgs, cads)
    assert hist["loss"][-1] < 0.1, hist["loss"][-5:]
    assert ((p > 0.5) == labels).all()


def test_threshold_only_on_manhattan_val():
    p = np.array([0.1, 0.2, 0.8, 0.9])
    y = np.array([False, False, True, True])
    with pytest.raises(ValueError):
        pick_threshold(p, y, split="test", family="manhattan")
    with pytest.raises(ValueError):
        pick_threshold(p, y, split="val", family="via")
    t = pick_threshold(p, y, split="val", family="manhattan")
    assert 0.2 < t <= 0.8


def test_save_load_roundtrip_gives_identical_predictions(tmp_path):
    imgs, cads, labels = _toy(4)
    cfg = CnnCfg(epochs=1, batch=4, seed=1, width=8)
    model, hist = train_cnn(imgs, cads, labels, cfg)
    path = tmp_path / "cnn.pt"
    save_cnn(model, cfg, hist, path)
    loaded, cfg2, hist2 = load_cnn(path)
    assert cfg2 == cfg and hist2["loss"] == hist["loss"]
    assert np.array_equal(predict_proba(model, imgs, cads), predict_proba(loaded, imgs, cads))


def test_device_option_defaults_to_cpu_and_unknown_device_fails():
    assert CnnCfg().device == "cpu"
    imgs, cads, labels = _toy(2, size=64)
    with pytest.raises((RuntimeError, AssertionError, ValueError)):
        train_cnn(imgs, cads, labels, CnnCfg(epochs=1, batch=2, device="not-a-device"))


def test_train_set_family_argument():
    """The CNN train-set recipe takes a layout family; the default stays Manhattan."""
    from nanovate.baselines.data import make_cnn_train_set
    imgs, cads, ys, ds = make_cnn_train_set(1, 1, seed=0, family="via")
    assert imgs.shape == (6, 256, 256) and ys.sum() == 3
    ref = make_cnn_train_set(1, 1, seed=0)
    assert not np.array_equal(ref[1], cads)


def test_warm_start_does_not_mutate_init_and_starts_from_it():
    """Fine-tuning starts from a deep copy of ``init``; the shared base model is never changed."""
    imgs, cads, ys = _toy(4)
    base, _ = train_cnn(imgs, cads, ys, CnnCfg(epochs=1))
    before = {k: v.clone() for k, v in base.state_dict().items()}
    ft, _ = train_cnn(imgs, cads, ys, CnnCfg(epochs=1, lr=1e-9), init=base)
    assert all(torch.equal(before[k], v) for k, v in base.state_dict().items())
    assert torch.allclose(ft.stem[0].weight, base.stem[0].weight, atol=1e-6)
    scratch, _ = train_cnn(imgs, cads, ys, CnnCfg(epochs=1, lr=1e-9, seed=5))
    assert not torch.allclose(scratch.stem[0].weight, base.stem[0].weight, atol=1e-6)
