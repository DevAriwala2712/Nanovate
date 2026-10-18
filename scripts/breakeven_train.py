"""Fine-tune the Manhattan-trained CNN on k labelled patches of a new family (Colab/GPU; synthetic data).

Part 1 of docs/superpowers/specs/2026-10-18-next-steps-design.md: the label break-even experiment.
For each family, a pool of the family's *train* split (all six kinds, a random dose per patch) is
generated once; for every (k, seed) the base CNN (m6_cnn/cnn.pt) is fine-tuned on 75% of the first k
pool patches of a seeded permutation, the other 25% being kept for the threshold (chosen locally by
experiments/breakeven.py). The verifier and the test split are never touched here.
Writes <out>/manifest.json (rewritten after every run, so a partial run is still readable) and one
weights file per run with k > 0. Evaluate locally with: python -m experiments.breakeven --weights <out>
Usage: python scripts/breakeven_train.py [--families via,curvy] [--k 0,12,...] [--seeds 0,1,2] [--device auto] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch

from nanovate.baselines.breakeven import FT_LR, FT_STEPS, K_GRID, SEEDS, breakeven_subsets, finetune_epochs
from nanovate.baselines.cnn import CnnCfg, load_cnn, save_cnn, train_cnn
from nanovate.baselines.data import make_cnn_train_set

POOL_SEED = 0


def _ints(s: str) -> list[int]:
    return [int(x) for x in s.split(",") if x]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--families", default="via,curvy")
    ap.add_argument("--k", default=",".join(map(str, K_GRID)))
    ap.add_argument("--seeds", default=",".join(map(str, SEEDS)))
    ap.add_argument("--pool-layouts", type=int, default=32)
    ap.add_argument("--pool-dies", type=int, default=4)
    ap.add_argument("--steps", type=int, default=FT_STEPS)
    ap.add_argument("--base", default="m6_cnn/cnn.pt")
    ap.add_argument("--device", default="auto", help="auto = cuda if available else cpu")
    ap.add_argument("--out", default="results/v2/breakeven_weights")
    a = ap.parse_args()
    device = ("cuda" if torch.cuda.is_available() else "cpu") if a.device == "auto" else a.device
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    base, base_cfg, _ = load_cnn(a.base)
    manifest = {"base_weights": a.base, "pool": {"layouts": a.pool_layouts, "dies": a.pool_dies, "seed": POOL_SEED},
                "steps": a.steps, "lr": FT_LR, "torch": torch.__version__, "device": device, "data": "synthetic", "runs": []}

    def write():
        (out / "manifest.json").write_text(json.dumps(manifest, indent=1))

    for fam in a.families.split(","):
        t0 = time.time()
        imgs, cads, ys, _ = make_cnn_train_set(a.pool_layouts, a.pool_dies, seed=POOL_SEED, family=fam)
        print(f"{fam}: pool n={len(ys)} killers={int(ys.sum())} ({time.time() - t0:.0f}s)", flush=True)
        for k in _ints(a.k):
            for seed in _ints(a.seeds):
                ft_idx, thr_idx = breakeven_subsets(len(ys), k, seed)
                run = {"family": fam, "k": k, "seed": seed, "weights": None, "ft_idx": ft_idx.tolist(),
                       "thr_idx": thr_idx.tolist(), "epochs": 0, "lr": FT_LR, "steps": a.steps, "final_loss": None}
                if len(ft_idx):
                    cfg = CnnCfg(epochs=finetune_epochs(len(ft_idx), a.steps), batch=32, lr=FT_LR, seed=seed,
                                 width=base_cfg.width, augment=True, device=device)
                    t0 = time.time()
                    model, hist = train_cnn(imgs[ft_idx], cads[ft_idx], ys[ft_idx], cfg, init=base)
                    name = f"{fam}_k{k}_s{seed}.pt"
                    save_cnn(model, cfg, hist, out / name)
                    run.update(weights=name, epochs=cfg.epochs, final_loss=hist["loss"][-1])
                    print(f"  {fam} k={k} seed={seed}: {cfg.epochs} epochs, loss {hist['loss'][0]:.3f} -> "
                          f"{hist['loss'][-1]:.3f} ({time.time() - t0:.0f}s)", flush=True)
                manifest["runs"].append(run)
                write()
    write()
    print("wrote", out / "manifest.json")


if __name__ == "__main__":
    main()
