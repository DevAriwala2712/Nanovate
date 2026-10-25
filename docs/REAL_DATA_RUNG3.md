# Track C Rung 3: reference-cell mode on a real 90 nm chip (Puschner et al.), run once

**REAL DATA.**
- Data: doi:10.17617/3.396Q7I, CC-BY-4.0.
- Pre-registration: `docs/REAL_DATA_RUNG3_PREREG.md` (b7c18c1).
- Results: `results/real/rung3/`.
- Run: started 16:44 and finished 19:28 (2.7 h, 8 workers).

**Disclosures:**
- The first attempt crashed in pass B on float slice indices (fixed in 1e7ba06, no setting changed).
- A second attempt was stopped by the user (power) before writing anything.
- No verdict was seen before this complete run.

## Results vs pre-registered targets

Instance-level; 95% CI from a bootstrap over home tiles.

| Metric | Result | Target | Met? |
|---|---|---|---|
| **KILLER rate**, all verified cells (n = 429,232 in 1,082 tiles) | **28.0% [27.7, 28.4]** | ≤ 5% | **NO** |
| — functional cells (210,941) | 12.2% [11.7, 12.6] | — | — |
| — filler cells (218,291) | 43.4% [42.7, 44.1] | — | — |
| NEAR_MISS (no killer window) | 1.1% [1.0, 1.2] | ≤ 5% | yes |
| Coverage (verified / all 453,850 cells) | 94.6% | ≥ 80% | yes |
| Topology change (Δβ ≠ 0 in any window) | 43.1% | reported | — |

- **References:** all 825 references built were calibration-reliable.
- **Candidate list:** `candidates_top200.csv`. With a 28% false-killer rate it carries little information. **Recall is not measurable** (the swap positions are withheld).

## What broke (post-hoc diagnosis; nothing was changed)

- **The killers are almost all extra components.**
  - Fillers: every one of the 94,751 killer windows is KILLER_OPEN/PARTICLE.
  - Functional cells: 25,886 of 25,922.
  - Δβ₀ is +1 to +5, most often **+2**.
- **It's concentrated in specific filler types:** FILL_EC 54%, FILL_EG 56%, FILL_EI 27%; TI/TO/TU are below 2%.
- **Leading hypothesis (untested): a splice-seam artefact of reference-cell mode.** When the instance is slightly misregistered against the median, structures crossing the cell-box edge are cut, leaving stray fragments on both sides (the +2). The frozen rule counts those fragments as particles. A second candidate is that wide filler types contain features at varying positions, which the median blurs away.

## What this tells us

1. **The reference-cell mode as built is not usable on this chip** (28% false killers), even though it covers 95% of cells, every reference calibrated reliably, and the near-miss rate stayed low.
2. **The failure is in the method's construction (splicing and cell-box geometry), not in the topology core.** On the M2 test, the same core gave 0.31% KILLER against validated labels.
3. **No detection claim is possible:** a 28% false-alarm rate, and no ground-truth positions.

## Next (not done; synthetic first, then one pre-registered repeat)

- Replace the hard splice with a **margin-blended or eroded seam**, or count only components wholly inside the box. Test on synthetic misregistered cells.
- Check the filler-type hypothesis by viewing FILL_EC / FILL_EG medians next to their instances.
- Getting the swap positions from the authors (your call) is the only way to measure recall.
