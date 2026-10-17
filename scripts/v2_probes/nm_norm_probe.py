"""v2 design probe (Manhattan val only, synthetic data); see scripts/v2_probes/README.md."""
"""Val-only probe: near-miss statistics vs LER sigma (Manhattan val, dose 50). Scratch."""
import sys, numpy as np, pandas as pd
from scipy import ndimage as ndi
from experiments.common import ExpCfg
from experiments.harness import nuisance_cfg
from experiments.select_config import with_dose, load_pipeline_cfg
from nanovate.config import TestbedCfg
from nanovate.modules.calibration import calibrate
from nanovate.modules.align_sdf import cad_sdf, tension_energy, align_to_template
from nanovate.pipeline import verify_patch
from nanovate.testbed.dataset import get_layout, make_sample, window
from nanovate.testbed.scribe import make_scribe_patches
cfg = ExpCfg.full(); pc = load_pipeline_cfg(cfg); base = TestbedCfg(); win = window(base)
def emap(img, cad, cal):
    T = cal.predict(cad); al, _ = align_to_template(img, T)
    s = ndi.gaussian_filter(al - T, 1.0) / abs(cal.mu_cond - cal.mu_diel)
    return s, cad_sdf(cad)
rows = []
for sig in (0.5, 1.0, 1.5, 2.0, 2.5):
  for gr in (0.15, 0.30):
    tc = with_dose(nuisance_cfg(base, sig, gr), 50)
    sp = make_scribe_patches(tc, seed=0); cal = calibrate(sp)
    sq = []
    for c, im in sp:
        s, phi = emap(im, c, cal)
        inner = (slice(20, -20), slice(20, -20))
        e = tension_energy(s, phi, 0.2, 10.0)[inner]; band = (np.abs(phi) <= 3)[inner]
        sq.append(dict(scr_emax=e.max(), scr_band_q99=np.quantile(np.abs(s[inner][band]), 0.99), scr_band_rms=np.sqrt((s[inner][band]**2).mean())))
    scr = pd.DataFrame(sq).max().to_dict()
    for lay in range(8):
      L = get_layout("manhattan", "val", lay, tc, 0)
      for die in range(2):
        for kind in ("nuisance", "notch", "filament"):
          smp = make_sample("manhattan", "val", lay, die, kind, tc, 0, layout=L)
          r = verify_patch(smp.image, smp.cad, cal, win, pc)
          s, phi = emap(smp.image, smp.cad, cal)
          sc, pc_ = win.crop(s), win.crop(phi)
          band = np.abs(pc_) <= 3
          e = tension_energy(sc, pc_, 0.2, 10.0)
          rows.append(dict(sigma=sig, grain=gr, layout=lay, die=die, kind=kind, verdict=r.verdict, d_b=tuple(r.reason["delta_betti"]),
                           bott=r.reason["bottleneck_max"], energy=r.reason["max_sdf_energy"],
                           e_q99=np.quantile(e, 0.99), e_med=np.median(e), band_rms=np.sqrt((sc[band]**2).mean()),
                           band_q99=np.quantile(np.abs(sc[band]), 0.99), **scr))
    print(sig, gr, flush=True)
pd.DataFrame(rows).to_csv(f"{sys.argv[1]}/nm_probe.csv", index=False)
