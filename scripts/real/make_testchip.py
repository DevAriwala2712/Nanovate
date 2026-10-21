"""Track C Rung 2: write the test-chip GDS, the exact defect map and a preview (design only; nothing is sent anywhere).

Outputs (results/real/testchip/<preset>/): nanovate_testchip.gds, defect_map.csv, config.json, preview.png, closeups.png.
Every defect's expected Delta-beta is computed from the drawn geometry; the script fails if any block's topology does
not match its intended class. Usage: python scripts/real/make_testchip.py [--preset ebeam|ebeam_aggressive|optical]
"""
from __future__ import annotations

import argparse
import datetime
import dataclasses
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import pandas as pd  # noqa: E402

from nanovate.real.testchip import EXPECTED, PRESETS, ChipCfg, build_chip, cfg_dict, defect_rows, raster  # noqa: E402

GDS_TIMESTAMP = datetime.datetime(2026, 10, 21)


def preview(lib, out: Path, c: ChipCfg, df: pd.DataFrame) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Polygon as P
    cell = lib.top_level()[0]
    fig, ax = plt.subplots(figsize=(10, 12))
    for p in cell.polygons:
        if p.layer in (1, 2):
            ax.add_patch(P(p.points / 1e3, closed=True, fc="k" if p.layer == 1 else "tab:blue", ec="none"))
    for _, r in df[df.defect_x_um.notna()].iterrows():
        ax.plot(r.defect_x_um, r.defect_y_um, "o", mfc="none", mec="r", ms=6, mew=0.8)
    ax.autoscale(); ax.set_aspect("equal"); ax.set_xlabel("x (um)"); ax.set_ylabel("y (um)")
    ax.set_title(f"Nanovate test chip (DESIGN), line width {c.w:g} nm; red circles = deliberate defects; blue = labels/marks", fontsize=9)
    fig.savefig(out, dpi=150, bbox_inches="tight"); plt.close(fig)


def closeups(blocks, out: Path, c: ChipCfg) -> None:
    """Centre crop of the largest-size block of each defect type in replicate 0 (design raster, px = w/20)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    pick = {}
    for b in blocks:
        if b.rep == 0 and (b.defect not in pick or (b.size_frac or 0) > (pick[b.defect].size_frac or 0)):
            pick[b.defect] = b
    fig, ax = plt.subplots(2, 5, figsize=(14, 6))
    for a, (k, b) in zip(ax.ravel(), sorted(pick.items(), key=lambda kv: (kv[1].family, kv[0]))):
        r = raster(b, c)
        n = r.shape[0]; h = n // 4
        a.imshow(r[n // 2 - h:n // 2 + h, n // 2 - h:n // 2 + h], cmap="gray", origin="lower")
        a.set_title(f"{b.family} / {k}" + (f"\n{b.size_param} {b.size_frac * c.w:g} nm" if b.size_frac else ""), fontsize=8); a.axis("off")
    for a in ax.ravel()[len(pick):]:
        a.axis("off")
    fig.suptitle(f"DESIGN close-ups (centre half of each block), w = {c.w:g} nm; white = conductor", fontsize=9)
    fig.savefig(out, dpi=130, bbox_inches="tight"); plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preset", default="ebeam", choices=sorted(PRESETS))
    a = ap.parse_args()
    c = dataclasses.replace(ChipCfg(), **PRESETS[a.preset])
    out = Path("results/real/testchip") / a.preset
    out.mkdir(parents=True, exist_ok=True)
    lib, blocks = build_chip(c)
    df = pd.DataFrame(defect_rows(blocks, c))
    d = df[df.family != "calibration"]
    bad = d[[EXPECTED[t][1] != (b0, b1) for t, b0, b1 in zip(d.defect_type, d.design_d_beta0, d.design_d_beta1)]]
    if len(bad):
        sys.exit(f"topology check failed:\n{bad[['block_id', 'defect_type', 'size_frac_w', 'design_d_beta0', 'design_d_beta1']]}")
    lib.write_gds(out / "nanovate_testchip.gds", timestamp=GDS_TIMESTAMP)   # fixed: same config -> identical bytes
    df.to_csv(out / "defect_map.csv", index=False)
    (out / "config.json").write_text(json.dumps(dict(preset=a.preset, data="DESIGN (no fabrication yet)", **cfg_dict(c)), indent=2))
    preview(lib, out / "preview.png", c, df)
    closeups(blocks, out / "closeups.png", c)
    print(df.groupby(["family", "defect_type"]).size().to_string())
    print("blocks", len(df), "chip extent um:", round(df.block_x1_um.max() - df.block_x0_um.min(), 2), "x", round(df.block_y1_um.max() - df.block_y0_um.min(), 2))


if __name__ == "__main__":
    main()
