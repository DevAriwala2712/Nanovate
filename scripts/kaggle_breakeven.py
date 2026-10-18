"""Run the Part 1 label break-even fine-tuning as a private Kaggle GPU kernel (synthetic data).

Writes a kernel folder (kernel-metadata.json + a notebook pinned to one pushed commit) and, unless --dry-run,
pushes it with the Kaggle CLI. The notebook clones the repo at that commit, installs it, and runs
scripts/breakeven_train.py --device auto --out /kaggle/working/breakeven. Afterwards:
  kaggle kernels status <user>/nanovate-breakeven
  kaggle kernels output <user>/nanovate-breakeven -p results/v2/breakeven_weights
then evaluate locally: python scripts/reproduce.py --skip-select --only day0 --breakeven-weights results/v2/breakeven_weights/breakeven
Kaggle needs a phone-verified account for GPU and internet. notebooks/breakeven_colab.ipynb is the fallback.
Usage: python scripts/kaggle_breakeven.py --user <kaggle-username> [--commit <sha>] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path

SLUG = "nanovate-breakeven"
REPO = "https://github.com/DevAriwala2712/Nanovate.git"
KAGGLE = str(Path.home() / ".local/bin/kaggle")


def _cell(src: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": src.strip("\n").splitlines(True)}


def write_kernel(out: Path, user: str, commit: str) -> Path:
    """Write the kernel folder for ``user`` pinned to ``commit``; returns the folder."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    nb = {"cells": [
        {"cell_type": "markdown", "metadata": {}, "source": [
            "# Nanovate label break-even (synthetic data)\n",
            f"Fine-tunes the Manhattan CNN on k labelled via / curvy patches; pinned to commit `{commit}`.\n"]},
        _cell("!nvidia-smi | head -12"),
        _cell(f"""
import os, subprocess
if not os.path.exists('/kaggle/working/Nanovate'):
    subprocess.run(['git', 'clone', '-q', '{REPO}', '/kaggle/working/Nanovate'], check=True)
os.chdir('/kaggle/working/Nanovate')
!git fetch -q origin && git checkout -q {commit} && git rev-parse HEAD
!pip install -q -e .
"""),
        _cell("""
%cd /kaggle/working/Nanovate
!python scripts/breakeven_train.py --device auto --out /kaggle/working/breakeven
!git rev-parse HEAD > /kaggle/working/breakeven/commit.txt
!rm -rf /kaggle/working/Nanovate
"""),
    ], "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"}, "language_info": {"name": "python"}},
        "nbformat": 4, "nbformat_minor": 4}
    (out / "breakeven.ipynb").write_text(json.dumps(nb, indent=1))
    meta = {"id": f"{user}/{SLUG}", "title": SLUG, "code_file": "breakeven.ipynb", "language": "python", "kernel_type": "notebook",
            "is_private": True, "enable_gpu": True, "enable_internet": True, "dataset_sources": [], "competition_sources": [],
            "kernel_sources": []}
    (out / "kernel-metadata.json").write_text(json.dumps(meta, indent=1))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--user", required=True)
    ap.add_argument("--commit", default="")
    ap.add_argument("--out", default="results/v2/kaggle_kernel")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    commit = a.commit or subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=True).stdout.strip()
    d = write_kernel(Path(a.out), a.user, commit)
    print("kernel folder:", d, "commit:", commit)
    if not a.dry_run:
        subprocess.run([KAGGLE, "kernels", "push", "-p", str(d)], check=True)


if __name__ == "__main__":
    main()
