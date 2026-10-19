# Nanovate: Next Paths (post-v2 roadmap, revision 2)

*Status as of v2 (synthetic data only). Every target below is a hypothesis to be measured and reported as met or missed. Every new idea comes with a pre-registered **kill condition**: if the data says it doesn't earn its place, it gets scrapped, including ideas we already built.*

**Revision 2 adds:** Track B (algorithm agility, compute, acquisition robustness), each built as *industry-style baseline first, then our own solution*; a Module 2 keep-or-scrap test; a tighter near-miss false-flag measurement; and a consolidated scrap list.

---

## Where we stand

| Area | v2 measured | Realistic ceiling | v3.0 plan doc claim |
|---|---|---|---|
| Zero-shot killer F1 (via / curvy) | 0.88 / 0.84 | 0.90–0.93 | 0.96 |
| Manhattan killer F1 | 0.81 (CNN: 0.94) | ~0.90 | 0.97 |
| Noise-free ceiling | 0.84 | >0.92 needed for the above | not considered |
| Near-miss recall | 90% | ~95% | 97.4% |
| Nuisance suppression (LER 1.5 nm, grain 30%) | 88%, bridge recall 79% | ~95% / ~95% | 98.6% |
| Low-dose F1 (5 e⁻/px) | 0.67 | ~0.75 | 0.92 |
| Calibration | 43 ms | ~15 ms | 10 ms |
| Time per patch | 99 ms (CPU; 76 ms of it is Module 4) | 5–10 ms GPU, or <5 ms CPU with Track B2 | ~0.03 ms |
| Real SEM validation | none | one public benchmark | claimed |

**Key facts that drive the plan**

1. The verifier tops out at **0.84 F1 even on noise-free images**, so denoising will not raise the ceiling. The bottleneck is resolution and/or the single global τ.
2. The CNN beats us **in-domain**, at **low dose**, and on **speed**. We win on **unseen via layouts with zero labels** and **miss fewer killers** (2 vs 12 of 50).
3. We send **more** sites to review (179 vs 111). The pitch is "fewer escapes", not "less review".
4. The method needs **CAD aligned to every image**, and public SEM datasets almost never include CAD.
5. The **stack path (Module 2) has not beaten per-patch smoothing** for Module 4 (better particles, worse opens). It has to earn its place.
6. The near-miss false-flag rate is unusable as a claim: harmless sites flagged NEAR_MISS = 36, CI [0, 107].

---

## Method for every new component: replicate, then beat, then maybe scrap

1. **Replicate:** implement an *industry-style* baseline from public descriptions of how commercial tools do it (`existing_semiconductor_inspection_systems_bottlenecks.md`). Label it "industry-style", never as the vendor's actual system.
2. **Beat:** build our own solution, measured on the same data against that baseline.
3. **Kill condition:** written down *before* the experiment. If met, the idea is dropped from code paths and from the pitch, and the result is still reported.

---

## Phase 0: BIOS hack (done)

Demo app, frozen v2, honest deck. Remaining items outside the repo: GitHub "About" tagline (still "revolutionary"), a tagged v2 release.

---

## Track A: core accuracy

### Phase 1: Diagnose the ceiling (about 2–4 days, validation data only)

**Question:** is the 0.84 noise-free ceiling caused by resolution or by our decision logic?

- [ ] **Oracle-τ test:** for each noise-free miss, does *any* per-sample τ give the right verdict? Report the recoverable fraction per defect type.
- [ ] **Sampling sweep:** same physical layouts and defects at 1×, 2×, 3× finer sampling, blur and defect sizes fixed in nm. Report F1 and per-type recall.
- [ ] **Miss log:** type, size or gap in nm, Δβ, field value at the defect, distance from τ.
- [ ] **Module 2 keep-or-scrap:** on the same misses and on Exp 1, compare stack path vs singleton path verdict by verdict.
  - *Kill condition:* stacking changes fewer than 2% of verdicts, or does not improve killer F1 by at least its CI half-width → drop the stack path from the default pipeline and from the pitch (keep the code, off by default).

**Decision point**

- Most misses recoverable by oracle τ, F1 flat with sampling → **logic-limited** → Phase 2.
- Misses unrecoverable at any τ, F1 rises with sampling → **resolution-limited** → report a physical detection limit in nm, skip most of Phase 2.

### Phase 2: Fix the core decision rule (about 1–3 weeks)

- [ ] **Per-gap / per-wire local decisions:** predict the expected field value at each CAD gap and wire neck from T(C); call merged or cut relative to that, not one global τ.
- [ ] **Feature-level persistence:** use birth/death of the specific merging or splitting feature.
- [ ] **Grain fix:** a dark grain on a narrow bridge must not cancel the bridge (Exp 1, 79% bridge recall).
- [ ] **Near-miss false-flag measurement:** raise n for harmless near-miss-like sites until the CI on "harmless flagged NEAR_MISS" is at most ±10 percentage points. Report it next to near-miss recall.
- [ ] Select on Manhattan val only; freeze; evaluate once on a **fresh test seed**.

**Targets:** noise-free F1 ≥ 0.92; Exp 1 suppression ≥ 90% *and* bridge recall ≥ 90%; Manhattan F1 ≥ 0.88; harmless NEAR_MISS rate ≤ 5% with CI width ≤ 10 points.

*Kill condition:* the local rule fails to beat global τ on val by at least the CI half-width → keep global τ, report the negative result.

---

## Track B: the three integration factors

### B1. Algorithm agility: a new layer without hand labels

**Industry-style baseline (replicate first).** Commercial ADC fine-tunes a per-layer CNN from an earlier checkpoint, with engineers labelling the images the model is least sure about (uncertainty-driven active learning).

- [ ] Implement: start from the Manhattan CNN, fine-tune on *k* hand-labelled images of the new layer, chosen by uncertainty sampling (not at random).
- [ ] Re-measure the label break-even against the verifier with this stronger baseline. Expect it to be lower than the current 192. Report it honestly.

**Our solutions (two candidates, run both).**

1. **Pseudo-label hybrid (from revision 1):** the verifier labels the new layer, the CNN fine-tunes on its high-confidence verdicts, topology stays on as an explainable veto.
2. **Counter-intuitive: simulate the new layer before the first wafer.** On day 0 we already have the new layer's CAD and a forward model calibrated from 3 real scribe images (Module 1). Use the testbed to inject defects into *that layer's own CAD*, render them through *that layer's calibrated model*, and train the CNN on the result. No hand labels, no pseudo-labels. The simulator becomes the product; the verifier becomes the independent checker.

- [ ] Compare on via and curvy: industry-style active learning at k = 0, 10, 20, 50, 192 vs hybrid vs calibrated-simulation training vs both combined.
- [ ] Report the failure mode: does the CNN inherit the verifier's misses (hybrid) or the simulator's blind spots (simulation)?

**Targets:** matching the industry-style baseline's F1 at k = 50 with ≤ 10 hand labels.

*Kill conditions:* the hybrid never beats the verifier alone → drop it. Calibrated-simulation training does worse than the plain Manhattan CNN on the new layer → the sim-to-layer gap is too large; drop it and record why.

### B2. Compute: fit an on-tool compute budget

**Industry-style baseline (replicate first).** Commercial tools run a cheap pre-filter on every pixel (die-to-die threshold on FPGA), then an expensive model only on the candidates it passes, batched on GPU.

- [ ] Implement the cascade: die-to-die pre-filter at a high-recall threshold → verifier only on candidates. Measure **throughput per tile** (patches/s for the whole cascade), not just per-patch latency, and the recall lost at the pre-filter.

**Our solution: take full persistent homology out of the hot path.**

At a fixed threshold, Δβ₀ and Δβ₁ do not need a persistence computation:

- β₀ = connected components (one union-find or labelling pass, O(n)).
- β₁ from the Euler characteristic, β₁ = β₀ − χ in 2D, and χ is a sum of local 2×2 pixel-pattern counts (Gray's method): fully local, GPU- or even FPGA-friendly.
- Full persistence (bottleneck distance) is then needed only for **near-miss candidates**: Δβ = 0 and SDF energy above e*.

- [ ] Implement the fast Betti path; **acceptance test:** identical Δβ to GUDHI on every validation patch (connectivity convention must match GUDHI's cubical complex).
- [ ] Route only near-miss candidates to GUDHI.
- [ ] Combine with the B2 cascade. Report latency and throughput, CPU and (if available) GPU.

**Targets:** Module 4 under 5 ms per patch on CPU; whole pipeline under 25 ms per patch; zero verdict changes vs v2 on val.

*Kill condition:* any verdict differs from GUDHI → fix or drop. If the cascade's pre-filter loses more than 2 percentage points of killer recall, keep the verifier on all patches and say so.

Out of scope here: thermal design and on-tool hardware. Our job is to fit the tool's existing compute budget.

### B3. Acquisition robustness: vibration and drift show up as image artefacts

We cannot isolate a tool from vibration, but we can be robust to what it does to images.

**Add to the simulator first (Exp 4b):**

- [ ] **Scan-line jitter:** each row (or band of rows) shifted horizontally by a random sub-pixel to few-pixel offset.
- [ ] **Slow drift within a frame:** a gradual shift from top to bottom.
- [ ] **Multi-frame acquisition:** N fast frames, each with its own jitter and drift and 1/N of the dose.
- [ ] Measure verdict stability and killer F1 vs jitter amplitude, for v2 as is.

**Industry-style baseline (replicate).** SEMs acquire several fast frames and register and average them (drift-corrected frame integration); row jitter is corrected by aligning each row to its neighbours or a reference image.

- [ ] Implement drift-corrected frame integration and row-to-reference cross-correlation.

**Our solutions.**

1. **CAD-guided per-row registration:** align each scan row to the noise-free calibrated template T(C) instead of to a noisy reference. The template is the cleanest reference available.
2. **Counter-intuitive: don't fully correct; flag tears.** Topology ignores smooth warps, but a row offset can *tear* a vertical wire and create a fake open. Detect Δβ changes that sit exactly on a row boundary and coincide with a measured row offset; mark them ACQUISITION_ARTEFACT instead of KILLER.

- [ ] Compare: no correction / industry-style / CAD-guided rows / tear flagging / CAD-guided + tear flagging.

**Target:** ≥ 95% verdict stability and ≤ 2-point F1 loss at a jitter amplitude chosen on val before the test (pre-register the amplitude).

*Kill condition:* the industry-style correction alone meets the target → use it and drop our variants. Tear flagging hides real opens (any real open re-labelled ACQUISITION_ARTEFACT on val) → drop it.

---

## Phase 4: Make real data possible (parallel)

- [ ] **Reference-die mode:** clean die (or median of many dies) as the structural reference instead of CAD.
- [ ] Validate reference-die vs CAD mode on synthetic data first (F1 cost).
- [ ] Find a public SEM defect dataset with die-to-die references or repeated patterns; check licence, pixel size, label quality.
- [ ] Ask the university SEM facility about imaging test structures.

*Kill condition:* reference-die mode loses more than 10 F1 points vs CAD on synthetic → real-data work needs CAD access; pause and look for a data partner instead.

---

## Scrap list (decisions owed to the data)

| Candidate to scrap | Decided by | Kill condition |
|---|---|---|
| Stack path (Module 2) as default | Phase 1 | < 2% of verdicts changed, or no significant F1 gain |
| Global τ | Phase 2 | local rule beats it on val |
| Full persistent homology in the hot path | B2 | fast Betti path matches GUDHI exactly |
| Pseudo-label hybrid | B1 | never beats verifier alone |
| Calibrated-simulation training | B1 | worse than Manhattan CNN on new layer |
| Our jitter fixes | B3 | industry-style correction already meets the target |
| Low-dose (5 e⁻/px) as a selling point | already decided | information isn't in the image; drop from pitch |
| Optical-discovery claims | already decided | defects unresolved; out of scope |

---

## Suggested order

1. Phase 1 (+ Module 2 test) and B3 simulator additions: both validation-only, both cheap.
2. B2 fast Betti path: pure speed, zero verdict change, unblocks everything else.
3. Phase 2 core rule.
4. B1 agility comparison.
5. B3 corrections.
6. Phase 4 alongside, as data access allows.

---

## Out of scope (long-term R&D only)

3D X-ray ptychography; fab-wide causal GNN; 25-bin few-shot ADC (needs real labels); optical-discovery nuisance filtering; thermal and mechanical tool design.

---

## Rules carried over

- Thresholds chosen on validation only; each test seed touched once.
- Every number with n and a 95% CI.
- Targets and kill conditions written before the experiment; reported as met or missed.
- Previous versions stay in the README unchanged.
- "Industry-style" baselines are labelled as our implementation of publicly described approaches, never as the vendor's system.
