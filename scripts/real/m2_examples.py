"""M2 illustrations (after the one run; change nothing): the first 5 primary KILLER tiles (by image id, tile), as
pre-registered, plus one POST-HOC NEAR_MISS tile containing a via. Real data, doi:10.17617/3.HY5SYN, CC-BY-4.0.
Usage: PYTHONPATH=.:src python scripts/real/m2_examples.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from experiments.real_m2 import arm_arrays, crop

OUT = "results/real/m2"
t = pd.read_csv(f"{OUT}/tiles.csv", dtype={"image_id": str})
t = t[(t.skipped.fillna("") == "") & (t.arm == "native") & (t.qc_flag.astype(str) == "False")]
k = t[t.verdict.str.startswith("KILLER")].sort_values(["image_id", "tile"]).head(5)
nm = t[t.verdict == "NEAR_MISS"].sort_values(["image_id", "tile"]).head(1)
for tag, rows in (("killer", k), ("posthoc_nearmiss", nm)):
    for j, r in enumerate(rows.itertuples()):
        img, lab, _ = arm_arrays(r.image_id, 1.0)
        rc = tuple(int(v) for v in r.tile.split(","))
        fig, ax = plt.subplots(1, 2, figsize=(8, 4.3))
        ax[0].imshow(crop(img, rc), cmap="gray"); ax[0].contour(crop(lab, rc), [0.5], colors="lime", linewidths=0.6)
        ax[0].set_title("real M2 tile + label (lime)", fontsize=7)
        ax[1].imshow(crop(lab, rc), cmap="gray"); ax[1].set_title("label raster (reference)", fontsize=7)
        for a in ax:
            a.axis("off")
        fig.suptitle(f"REAL DATA (doi:10.17617/3.HY5SYN, CC-BY-4.0) sem{r.image_id} tile {r.tile}: {r.verdict}\n"
                     f"dbeta=({r.d_beta0:.0f},{r.d_beta1:.0f}) bottleneck {r.bottleneck:.3f} SDF energy {r.max_sdf_energy:.2f} peak-z {r.peak_z:.1f}"
                     + ("  [POST-HOC illustration]" if tag != "killer" else ""), fontsize=7)
        fig.savefig(f"{OUT}/example_{tag}_{j}_sem{r.image_id}_{r.tile.replace(',', '-')}.png", dpi=110, bbox_inches="tight")
        plt.close(fig)
print(k[["image_id", "tile", "verdict", "d_beta0", "d_beta1", "bottleneck", "max_sdf_energy"]].to_string())
