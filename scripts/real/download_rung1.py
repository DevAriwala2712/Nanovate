"""Download the Track C Rung 1 real SEM images into data/real/ (git-ignored). Real data; nothing is committed.

Sources (both CC-BY-4.0, attribution in data/real/MANIFEST.json and docs/REAL_DATA_RUNG1.md):
  zmw:    Yang, Klughammer, Barth, Tanenbaum, Dekker, "Zero-mode waveguide nanowells for single-molecule detection
          in living cells", ACS Nano 2023; data doi:10.5281/zenodo.8060099 (two SEM zips only).
  ingan:  Athanasiou et al., "InGaN Nanohole Arrays Coated by Lead Halide Perovskite Nanocrystals ...";
          data doi:10.5281/zenodo.3747155 (two top-view figure JPGs, spot-check only).
Every file is checked against the md5 Zenodo publishes. Zips are extracted with member-name checks.
Usage: python scripts/real/download_rung1.py [--out data/real]
"""
from __future__ import annotations

import argparse
import hashlib
import urllib.parse
import urllib.request
import zipfile
from pathlib import Path

FILES = [
    ("zmw", "8060099", "Gla_0127_14.zip", "683bf3e8201b0a57354e96eb9642a9ed"),
    ("zmw", "8060099", "Gla_29_Pd_1.zip", "7881fcd2b0fc84e093a49b3f1bfd3444"),
    ("ingan", "3747155", "Figure 2(b) SEM Uncoated Nanoholes top view.jpg", "d21bfd9fa5e9040c1190685ddadff3e5"),
    ("ingan", "3747155", "Figure 2(d) SEM Hybrid Nanoholes top view.jpg", "31f5fcc8b0f1104fc7295e20a8347023"),
]


def md5(p: Path) -> str:
    h = hashlib.md5()
    with open(p, "rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()


def safe_extract(zp: Path, dest: Path) -> None:
    with zipfile.ZipFile(zp) as z:
        for m in z.infolist():
            target = (dest / m.filename).resolve()
            if not str(target).startswith(str(dest.resolve())):
                raise RuntimeError(f"unsafe member {m.filename!r} in {zp}")
        z.extractall(dest)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/real")
    out = Path(ap.parse_args().out)
    for src, rec, key, want in FILES:
        d = out / src / "raw"
        d.mkdir(parents=True, exist_ok=True)
        p = d / key
        if not p.exists() or md5(p) != want:
            url = f"https://zenodo.org/api/records/{rec}/files/{urllib.parse.quote(key)}/content"
            print("downloading", url)
            urllib.request.urlretrieve(url, p)
        got = md5(p)
        if got != want:
            raise RuntimeError(f"md5 mismatch for {p}: {got} != {want}")
        print("ok", p, got)
        if key.endswith(".zip"):
            ex = out / src / "extracted" / key[:-4]
            if not ex.exists():
                ex.mkdir(parents=True)
                safe_extract(p, ex)


if __name__ == "__main__":
    main()
