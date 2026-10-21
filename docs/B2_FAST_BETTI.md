# B2 fast Betti path: acceptance test (synthetic data, Manhattan val only)

**Result: fails acceptance. The kill condition applies: GUDHI stays in the hot path.** The code is kept, off by default.
Nothing in the pipeline calls `src/nanovate/modules/fast_betti.py`.

## What was tested

NEXT_PATHS Track B2 proposed computing Δβ without persistent homology:
- β₀ from one connected-components pass;
- β₁ = β₀ − χ, with χ from local 2 x 2 pattern counts (`skimage.measure.euler_number`).

The fast path uses the same convention as GUDHI's vertex cubical complex (conductor 4-connected, dielectric 8-connected) and the same splices as Module 4's v2 path. That is, H0 on the field spliced with the CAD outside the relevance region, and H1 on the field filled with dielectric outside it.

The pre-registered acceptance test (written in `docs/NEXT_PATHS.md` before this code): **identical Δβ to GUDHI on every validation patch**. The kill condition: any verdict differs → fix or drop.

**Protocol.** `python -m experiments.b2_fast_betti` → `results/b2/` (`rows.csv`, `summary.csv`, `config.json`).
- **Patches:** Manhattan val layouts 0–23 x 2 dies x 6 kinds x {noisy, noise-free} = **576 patches**, at dose 50 e⁻/px, LER 1 nm, grain 15%.
- **Pipeline:** the frozen v2/v3 field (`results/v3/calib/decisions.json`), τ = 0.35. The GUDHI Δβ was cross-checked against `verify_patch`'s own reason on all 576 patches, and all were equal.
- **Fast-path level:** fixed a priori at 0.5. Every level from 0.05 to 0.95 was also scanned, as an in-sample upper bound, not a selection. No test data was generated.

## Results

| Level | Identical Δβ | Same verdict class | Passes |
|---|---|---|---|
| **0.50 (pre-registered)** | **382 / 576 (66.3%)** | 424 / 576 (73.6%) | **no** |
| Best of the 19 levels (also 0.50) | 66.3% | 73.6% | no |
| 0.35 / 0.65 (τ and 1 − τ) | 55.6% / 40.3% | 62.7% / 56.8% | no |

**Speed:** median Δβ computation **2.4 ms** (fast) vs **64.2 ms** (GUDHI, two cubical persistences) per patch, about 27x. Same CPU, 576 patches.

**Where it differs (level 0.5, 194 mismatches):**
- Every kind is affected: notch 46, open 40, filament 38, bridge 29, particle 25, nuisance 16.
- Both noisy (125 of 288) and noise-free (69 of 288) patches.
- Fast minus GUDHI: fewer components in 100 patches, more in 30; more holes in 80, fewer in 11. Up to 4 off.

## Why: a design error in the B2 premise, not an implementation bug

Module 4 counts **persistence bars with lifetime ≥ τ, over all levels** (`topology.betti_from_intervals`). It does not compute the Betti numbers of one superlevel set.
- **Where the two coincide:** on a binary field (`tests/test_fast_betti.py`: exact against GUDHI and `betti_binary` on 20 random binary fields).
- **Where they differ:** on a continuous field. A component born at a level of 0.6 that lives longer than τ is counted by Module 4 but missed by a 0.7 threshold. A dim bridge whose saddle sits below the threshold is the reverse case. Pinned by `test_single_level_differs_from_lifetime_count_on_a_dim_component`.

The NEXT_PATHS sentence "at a fixed threshold, Δβ₀ and Δβ₁ do not need a persistence computation" is true, but Module 4 does not use a fixed threshold.

Separately, the v3 rule needs the *located* bars (merge saddles per seed pair, cut margins). So even an exact Betti shortcut could only serve the v2 Δβ rule or a pre-screen.

## What would still be exact (not done; needs a decision)

- **Lifetime-counted β₀ without GUDHI.** An elder-rule union-find over the sorted field values is exact for H0 (and H1 by duality on the dual grid). Pure-Python loops over 65k pixels would be slower than GUDHI's C++. A compiled version needs a new dependency (e.g. numba), which AGENTS.md says to ask about first.
- **Cheaper exact wins elsewhere.** In the v3 profile (161 ms per patch on 12 val patches): GUDHI persistence 73 ms, merge-level reconstructions 28 ms, alignment 15 ms, bottleneck 10 ms. The reconstruction and bottleneck steps can be sped up without changing any verdict, and each change can be checked verdict by verdict against the frozen output.
