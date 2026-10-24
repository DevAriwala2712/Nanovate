"""Download the Track C Rung 3 datasets into data/real/ (git-ignored). Real data; nothing is committed.

Both are CC-BY-4.0, open, hosted on Edmond (Max Planck); every file is checked against the MD5 Edmond publishes.
  puschner:  Puschner, Moos, Becker, Kison, Moradi, Paar, "Red Team vs. Blue Team: A Real-World Hardware Trojan
             Detection Case Study Across Four Modern CMOS Technology Generations", IEEE S&P 2023;
             data doi:10.17617/3.396Q7I. ONLY the 90 nm tiles, the (abstracted, trojanized) GDSII files and stitching.
  metal:     "A Real-World Metal-Layer SEM Image Dataset with Partial Labels" (paper: "Towards Unsupervised SEM Image
             Segmentation for IC Layout Extraction"); data doi:10.17617/3.HY5SYN.
Downloads resume (curl -C -). Archives are NOT extracted here; experiments extract only the members they need.
Usage: python scripts/real/download_rung3.py [--out data/real] [--only puschner|metal]
"""
from __future__ import annotations

import argparse
import hashlib
import subprocess
from pathlib import Path

FILES = [  # (dataset, Edmond datafile id, file name, bytes, md5)
    ("puschner", 199839, "stitching.zip", 210796, "f85a54493b2c225d8c6eb952f24a6a81"),
    ("puschner", 199838, "modified_gds.zip", 16061127, "cf6ba95b43da27e1f1b950403db74e7d"),
    ("metal", 223856, "dataset.zip", 6396523974, "f3e3137d58b2171ae0c99468af2e490b"),
    ("puschner", 199836, "90nm_tiles.zip", 27522743231, "4b64e99d9c2528ca3faf2cf2f7bbd98a"),
]


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 24), b""):
            h.update(b)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/real")
    ap.add_argument("--only", choices=["puschner", "metal"])
    a = ap.parse_args()
    for ds, fid, name, size, want in FILES:
        if a.only and ds != a.only:
            continue
        d = Path(a.out) / ds / "raw"
        d.mkdir(parents=True, exist_ok=True)
        p = d / name
        if not (p.exists() and p.stat().st_size == size):
            url = f"https://edmond.mpg.de/api/access/datafile/{fid}"
            print("downloading", name, f"{size / 1e9:.2f} GB", flush=True)
            subprocess.run(["curl", "-sS", "-L", "--fail", "--retry", "5", "-C", "-", "-o", str(p), url], check=True)
        got = md5(p)
        if got != want:
            raise RuntimeError(f"md5 mismatch for {p}: {got} != {want}")
        print("ok", p, got, flush=True)


if __name__ == "__main__":
    main()
