"""Baseline: small ResNet-style killer classifier on [image, CAD] (AGENTS.md §5.6, PLAN.md Q5).

Trained on Manhattan *train* only (the caller passes only those patches), same 256 x 256 patch size
as the verifier, CPU, seeded. The decision threshold is chosen on Manhattan val only
(``pick_threshold`` refuses anything else). Input: channel 0 = the SEM image standardised per patch
(so one network serves all doses), channel 1 = the CAD raster (0/1). Output: killer logit.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F


@dataclass(frozen=True)
class CnnCfg:
    epochs: int = 10
    batch: int = 32
    lr: float = 2e-3
    seed: int = 0
    width: int = 16
    augment: bool = True   # random flips / transpose, applied jointly to image and CAD


class _Block(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.c1 = nn.Conv2d(cin, cout, 3, stride, 1, bias=False)
        self.b1 = nn.BatchNorm2d(cout)
        self.c2 = nn.Conv2d(cout, cout, 3, 1, 1, bias=False)
        self.b2 = nn.BatchNorm2d(cout)
        self.skip = None if (cin == cout and stride == 1) else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = F.relu(self.b1(self.c1(x)))
        y = self.b2(self.c2(y))
        return F.relu(y + (x if self.skip is None else self.skip(x)))


class ResNetSmall(nn.Module):
    """Stem (stride 2) + 4 residual blocks (strides 1, 2, 2, 2) + avg|max pooling + linear."""

    def __init__(self, width: int = 16, in_ch: int = 2):
        super().__init__()
        w = width
        self.stem = nn.Sequential(nn.Conv2d(in_ch, w, 3, 2, 1, bias=False), nn.BatchNorm2d(w), nn.ReLU())
        self.blocks = nn.Sequential(_Block(w, w, 1), _Block(w, 2 * w, 2), _Block(2 * w, 4 * w, 2), _Block(4 * w, 4 * w, 2))
        self.head = nn.Linear(8 * w, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.blocks(self.stem(x))
        return self.head(torch.cat([h.mean((2, 3)), h.amax((2, 3))], dim=1)).squeeze(1)


def prepare(images: np.ndarray, cads: np.ndarray) -> torch.Tensor:
    """(N, 2, H, W) float32: per-patch standardised image, CAD 0/1."""
    img = torch.from_numpy(np.asarray(images, dtype=np.float32))
    mu = img.mean((1, 2), keepdim=True)
    sd = img.std((1, 2), keepdim=True, unbiased=False) + 1e-6
    return torch.stack([(img - mu) / sd, torch.from_numpy(np.asarray(cads, dtype=np.float32))], dim=1)


def _augment(x: torch.Tensor, rng: np.random.Generator) -> torch.Tensor:
    out = []
    for xi in x:
        if rng.random() < 0.5:
            xi = xi.flip(-1)
        if rng.random() < 0.5:
            xi = xi.flip(-2)
        if rng.random() < 0.5:
            xi = xi.transpose(-1, -2)
        out.append(xi)
    return torch.stack(out)


def train_cnn(images: np.ndarray, cads: np.ndarray, labels: np.ndarray, cfg: CnnCfg = CnnCfg()) -> tuple[ResNetSmall, dict]:
    """Train on the given (Manhattan train) patches. Returns (model, history). Deterministic in cfg.seed."""
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    model = ResNetSmall(cfg.width)
    opt = torch.optim.Adam(model.parameters(), lr=cfg.lr)
    y = torch.from_numpy(np.asarray(labels, dtype=np.float32))
    pos_w = torch.tensor(float((y == 0).sum()) / max(float((y == 1).sum()), 1.0))
    n = len(y)
    hist: dict = {"loss": [], "first_batch_loss": None}
    model.train()
    for _ in range(cfg.epochs):
        order = rng.permutation(n)
        total = 0.0
        for s in range(0, n, cfg.batch):
            idx = order[s:s + cfg.batch]
            xb = prepare(images[idx], cads[idx])
            if cfg.augment:
                xb = _augment(xb, rng)
            loss = F.binary_cross_entropy_with_logits(model(xb), y[idx], pos_weight=pos_w)
            if hist["first_batch_loss"] is None:
                hist["first_batch_loss"] = float(loss.detach())
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += float(loss.detach()) * len(idx)
        hist["loss"].append(total / n)
    return model, hist


@torch.no_grad()
def predict_proba(model: ResNetSmall, images: np.ndarray, cads: np.ndarray, batch: int = 64) -> np.ndarray:
    """Killer probability per patch (eval mode)."""
    model.eval()
    out = [torch.sigmoid(model(prepare(images[s:s + batch], cads[s:s + batch]))).numpy() for s in range(0, len(images), batch)]
    return np.concatenate(out)


def pick_threshold(probs: np.ndarray, labels: np.ndarray, split: str, family: str) -> float:
    """Probability threshold maximising killer F1; Manhattan val only (ties: lowest threshold)."""
    if split != "val" or family != "manhattan":
        raise ValueError("the CNN threshold may only be chosen on Manhattan val")
    y = np.asarray(labels, bool)
    best_t, best_f1 = 0.5, -1.0
    for t in np.unique(probs):
        pred = probs >= t
        tp = float((pred & y).sum())
        f1 = 2 * tp / max(2 * tp + float((pred & ~y).sum()) + float((~pred & y).sum()), 1.0)
        if f1 > best_f1 + 1e-12:
            best_t, best_f1 = float(t), f1
    return best_t
