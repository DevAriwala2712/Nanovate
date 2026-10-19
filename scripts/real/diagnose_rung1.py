"""POST-HOC diagnostic for Track C Rung 1 (real data; run after the one pre-registered run). Changes nothing.

For chosen failing tiles, re-derives the field the frozen verifier saw (Module 1 template, Module 3 alignment, v2
template-anchored field) and plots image / template / field / field > 0.5 against the FITTED reference. Tile choice
(written here, after seeing tiles.csv): the first clean arm-C tile per (group, verdict) in tiles.csv order.
Usage: PYTHONPATH=. python scripts/real/diagnose_rung1.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from experiments.real_rung1 import OUT, build, frozen_cfg
from nanovate.modules.align_sdf import align_to_template
from nanovate.modules.calibration import calibrate, features
from nanovate.modules.fields import build_field
from nanovate.real.lattice import render
from nanovate.real.rung1 import crop


def main() -> None:
    t = pd.read_csv(OUT / "tiles.csv")
    c = t[(t.arm == "C") & t.clean & ~t.overlay]
    pick = c.groupby(["group", "verdict"], sort=False).head(1)
    pc, _ = frozen_cfg()
    groups = {g["name"]: g for g in build("C")[0]}
    rows = []
    for _, r in pick.iterrows():
        g = groups[r.group]
        ims, refs = g["images"], g["refs"]
        cad_of = {k: render(refs[k].lattice, im.img.shape) for k, im in ims.items()}
        cal = calibrate([(crop(cad_of[k], rc).astype(float), crop(ims[k].img, rc)) for k, rc in g["cal_tiles"]])
        rc = tuple(int(v) for v in r.tile.split(","))
        img, cad = crop(ims[r.image_id].img, rc), crop(cad_of[r.image_id], rc)
        T = cal.predict(cad)
        Ia, sh = align_to_template(img, T)
        f, _ = build_field(Ia, option="template", template=T, levels=cal.levels, pre_smooth=pc.pre_smooth,
                           cad_smooth=features(cad, *cal.sigmas)[..., 1], contrast=cal.mu_cond - cal.mu_diel)
        resid = Ia - T
        flat = cad & (np.hypot(*np.gradient(features(cad, *cal.sigmas)[..., 1])) < 1e-3)
        rows.append(dict(group=r.group, image_id=r.image_id, tile=r.tile, verdict=r.verdict, contrast=cal.mu_cond - cal.mu_diel,
                         film_resid_sd=float(resid[flat].std()), film_resid_sd_over_contrast=float(resid[flat].std() / abs(cal.mu_cond - cal.mu_diel)),
                         field_film_p01=float(np.percentile(f[flat], 1)), frac_film_field_below_0_65=float((f[flat] < 1 - pc.tau).mean()),
                         m3_shift=[float(sh[0]), float(sh[1])]))
        fig, ax = plt.subplots(1, 4, figsize=(13, 3.6))
        for a, (im_, ttl) in zip(ax, [(Ia, "aligned real image"), (T, "Module-1 template T(C)"), (f, "field (frozen v2/v3)"),
                                      (f > 0.5, "field > 0.5 (conductor)")]):
            a.imshow(im_, cmap="gray"); a.contour(cad, [0.5], colors="c", linewidths=0.5); a.set_title(ttl, fontsize=8); a.axis("off")
        fig.suptitle(f"POST-HOC DIAGNOSTIC, REAL DATA (doi:10.5281/zenodo.8060099, CC-BY-4.0) {r.group} {r.image_id} tile {r.tile}: "
                     f"{r.verdict}, dbeta=({r.d_beta0},{r.d_beta1}); film residual sd / contrast = {rows[-1]['film_resid_sd_over_contrast']:.2f}",
                     fontsize=8)
        fig.savefig(OUT / f"diag_{r.group}_{r.verdict.replace('/', '-')}.png", dpi=110, bbox_inches="tight"); plt.close(fig)
    d = pd.DataFrame(rows)
    d.to_csv(OUT / "diagnostic.csv", index=False)
    print(d.to_string())


if __name__ == "__main__":
    main()
