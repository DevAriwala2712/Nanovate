"""Train the Manhattan-only CNN baseline (run on Colab/GPU or anywhere; synthetic data).

Generates the Manhattan *train* split (every defect kind, a random dose per sample from 5-200 e-/px),
trains ``ResNetSmall`` and writes weights + config + loss history. Evaluation (val threshold, F1,
latency) happens afterwards with:  python scripts/baselines_report.py --cnn-weights <out>/cnn.pt
Usage: python scripts/train_cnn.py [--layouts 24] [--dies 4] [--epochs 40] [--device auto] [--out results/m6_cnn]
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import time
from pathlib import Path

import numpy as np
import torch

from nanovate.baselines.cnn import CnnCfg, predict_proba, save_cnn, train_cnn
from nanovate.baselines.data import TRAIN_DOSES, make_cnn_train_set
from nanovate.config import TestbedCfg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--layouts", type=int, default=24)
    ap.add_argument("--dies", type=int, default=4)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--lr", type=float, default=2e-3)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="auto", help="auto = cuda if available else cpu")
    ap.add_argument("--out", default="results/m6_cnn")
    a = ap.parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    imgs, cads, ys, doses = make_cnn_train_set(a.layouts, a.dies, seed=a.seed)
    print(f"train set: n={len(ys)} killers={int(ys.sum())} doses={TRAIN_DOSES}  ({time.time() - t0:.0f}s)", flush=True)
    cfg = CnnCfg(epochs=a.epochs, batch=a.batch, lr=a.lr, seed=a.seed, device=device)
    print(f"training on {device}: {a.epochs} epochs x {int(np.ceil(len(ys) / a.batch))} steps", flush=True)
    t0 = time.time()
    model, hist = train_cnn(imgs, cads, ys, cfg)
    train_s = time.time() - t0
    p = predict_proba(model, imgs, cads)
    acc = float(((p > 0.5) == ys).mean())
    print(f"done in {train_s:.0f}s; loss every 5 epochs: {[round(x, 3) for x in hist['loss'][::5]]}, last {hist['loss'][-1]:.3f}")
    print(f"TRAIN accuracy at 0.5 (in-sample, not a result): {acc:.2f}")
    for lo, hi in ((0, 20), (20, 100), (100, 1e9)):
        m = (doses >= lo) & (doses < hi)
        print(f"  dose [{lo},{hi}): n={int(m.sum())} train acc {float(((p[m] > 0.5) == ys[m]).mean()):.2f}")
    save_cnn(model, cfg, hist, out / "cnn.pt")
    (out / "train_summary.json").write_text(json.dumps({
        "cfg": dataclasses.asdict(cfg), "testbed": dataclasses.asdict(TestbedCfg()), "layouts": a.layouts, "dies": a.dies,
        "n": int(len(ys)), "train_seconds": train_s, "device": device, "loss": hist["loss"], "train_acc_in_sample": acc,
        "torch": torch.__version__}, indent=2))
    print("saved", out / "cnn.pt")


if __name__ == "__main__":
    main()
