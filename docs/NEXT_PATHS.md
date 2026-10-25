# Nanovate: Next Paths (post-v2 roadmap, revision 2; status updated for v3 and Track C Rung 1)

## Done / In progress / Next (as of 2026-10-20)

**Done**
- v1 (pre-registered), v2 (post-hoc fixes, val only) and **v3** (per-gap merge rule + located events, frozen on val, Experiments 1–6 run once on a fresh test seed; tag `v3.0`). Synthetic data.
- Day-0 review-queue projection for v2 and v3 (`results/v2/day0_scenario/`, `results/v3/day0/`); label break-even (v2 only).
- Demo app with a v2 | v3 pipeline switch (v3 default), version badges, Evidence tab per version, and val-split demo presets.
- **Track C Rung 1 run once** on real SEM images of defect-free hole arrays: **78.5% false KILLER** on clean tiles (target ≤ 5%, **not met**; `docs/REAL_DATA_RUNG1.md`).

**In progress**
- Fixing the Rung 1 failure: Module 1 calibration transfer (contrast sign inverted on G01; the template explains about 15–25% of the real image's variation), edge-halo appearance, and alignment on periodic patterns. Any change is chosen on synthetic val, then Rung 1 is repeated once.

**Next**
- Rung 1 repeat with the disclosed fix; Rung 2 lab request (own test chip, true CAD); B2 fast Betti path; B1 agility comparison; push tags `v2.0` / `v3.0`.

---

*Status below as of v2 (synthetic data only), with v3 and Rung 1 added where measured. Every target below is a hypothesis to be measured and reported as met or missed. Every new idea comes with a pre-registered **kill condition**: if the data says it doesn't earn its place, it gets scrapped, including ideas we already built.*

**Revision 2 adds:** Track B (algorithm agility, compute, acquisition robustness), each built as *industry-style baseline first, then our own solution*; a Module 2 keep-or-scrap test; a tighter near-miss false-flag measurement; and a consolidated scrap list.

---

## Where we stand

*Numbers copied from the generated README tables (v2: `results/v2/`; v3: `results/v3/`, fresh test seed; Rung 1: `results/real/rung1/`). The plan-doc column is an unmeasured claim (see `docs/README.md`).*

| Area | v2 measured | v3 measured (fresh test seed) | Realistic ceiling | v3.0 plan doc claim (not measured) |
|---|---|---|---|---|
| Zero-shot killer F1 (via / curvy) | 0.88 / 0.84 | 0.84 / 0.90 (CNN 0.75 / 0.83) | 0.90–0.93 | 0.96 |
| Manhattan killer F1 | 0.81 (CNN: 0.94) | 0.87 (CNN: 0.92) | ~0.90 | 0.97 |
| Noise-free ceiling | 0.84 | 0.89 (target 0.92, not met) | >0.92 needed for the above | not considered |
| Near-miss recall | 90% | 85% | ~95% | 97.4% |
| Nuisance suppression (LER 1.5 nm, grain 30%) | 88%, bridge recall 79% | 75%, bridge recall 96% (worst cell at LER ≤ 1.5 nm: 67% / 88%) | ~95% / ~95% | 98.6% |
| Low-dose F1 (5 e⁻/px) | 0.67 | 0.67 | ~0.75 | 0.92 |
| Calibration | 43 ms | 43 ms | ~15 ms | 10 ms |
| Time per patch | 99 ms (CPU; 76 ms of it is Module 4) | 130 ms (CPU; 107 ms of it is Module 4) | 5–10 ms GPU, or <5 ms CPU with Track B2 | ~0.03 ms |
| Killers missed per 50 (via review-queue projection) | 2 (CNN 12) | 3 [1, 6] (CNN 9 [6, 13]) | — | — |
| Real SEM validation | none | Rung 1: 78.5% false KILLER on clean real tiles (target ≤ 5%, not met) | one public benchmark | claimed |

**Key facts that drive the plan**

1. v2 topped out at **0.84 F1 even on noise-free images**; v3's per-gap rule raised that to **0.89**, still short of 0.92. Part of the rest is an optical limit (cuts shorter than 4 nm, `docs/V3_PHASE1_DIAGNOSIS.md`).
2. The CNN beats us **in-domain**, at **low dose**, and on **speed**. We win on **unseen via and curvy layouts with zero labels** and **miss fewer killers** (v2: 2 vs 12 of 50; v3: 3 vs 9 of 50, projection).
3. We send **more** sites to review (v2: 179 vs 111; v3: 353 vs 160). The pitch is "fewer escapes", not "less review".
4. The method needs **CAD aligned to every image**, and public SEM datasets almost never include CAD.
5. The **stack path (Module 2) has not beaten per-patch smoothing** for Module 4 (better particles, worse opens). It has to earn its place.
6. The near-miss false-flag rate is unusable as a claim: harmless sites flagged NEAR_MISS = 36, CI [0, 107] (v2); 214, CI [71, 356] (v3), worse.
7. **Real data (Rung 1):** the frozen v3 verifier calls 78.5% of clean real tiles KILLER. The first bottleneck is Module 1 calibration transfer, not the topology rule (`docs/REAL_DATA_RUNG1.md` §4).

---

## Method for every new component: replicate, then beat, then maybe scrap

1. **Replicate:** implement an *industry-style* baseline from public descriptions of how commercial tools do it (`existing_semiconductor_inspection_systems_bottlenecks.md`). Label it "industry-style", never as the vendor's actual system.
2. **Beat:** build our own solution, measured on the same data against that baseline.
3. **Kill condition:** written down *before* the experiment. If met, the idea is dropped from code paths and from the pitch, and the result is still reported.

---

## Phase 0: BIOS hack (done)

Demo app (now v2 | v3 switch, **v3 by default**, version badge on every verdict and report), frozen v2 and v3 configs, honest deck reporting v3, README status box with the Rung 1 result. GitHub "About" tagline updated (2026-10-20). Annotated tags `v2.0` (commit 58cf3b6, `results/v2/`) and `v3.0` (commit 8ef81d8, `results/v3/`) exist locally; pushing them is the one remaining item.

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

- [x] Implement the fast Betti path; **acceptance test:** identical Δβ to GUDHI on every validation patch (connectivity convention must match GUDHI's cubical complex). *2026-10-21: **failed, kill condition applied** (`docs/B2_FAST_BETTI.md`): best single level identical on 382/576 val patches (66%), because Module 4 counts bars with lifetime ≥ τ over all levels, not one superlevel set; 27x faster (2.4 vs 64 ms). Code kept, off by default.*
- [ ] ~~Route only near-miss candidates to GUDHI.~~ Moot: the fast path cannot replace GUDHI for any patch. Exact options left: elder-rule union-find (needs a compiled dependency, ask first) or exact speed-ups of the reconstruction / bottleneck steps.
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

## Track C: real-world data

*Added 2026-10-19. Everything above is measured on the synthetic testbed only. Track C asks whether any of it survives contact with a real microscope. Rungs run easiest first; each one is pre-registered in a `docs/REAL_*.md` file before the first real image is scored. Settings stay frozen on real data: any change is chosen on synthetic validation and the real run is repeated once, with the change disclosed.*

### Rung 1: defect-free real patterned structures (false alarms, calibration, alignment)

*Status: run once (2026-10-19) on FIB-milled hole arrays; false KILLER 78.5% [66.4, 87.6] of 158 clean tiles, target ≤ 5% **not met** (`docs/REAL_DATA_RUNG1.md`). Next: a calibration guard / edge model / periodic alignment change chosen on synthetic val, then one repeat.*

*Rung 1b (2026-10-21, `docs/REAL_DATA_RUNG1B.md`): calibration guard + template clamp + bounded alignment, chosen on synthetic val (0 val verdicts changed), one repeat. Held-out fresh film tiles: KILLER 0/68, coverage 100% (met, film only). Same Rung 1 tiles: KILLER 13.3% [5.7, 23.6], coverage 73% (**not met**); G29 70.7% → 18.1% KILLER but NEAR_MISS 25% → 57%; G01 now UNVERIFIED (guard). Next: the near-miss gate at real rims; Rung 2 for recall.*

*Rung 1c (2026-10-22, `docs/REAL_DATA_RUNG1C.md`): near-miss gate re-selected with synthetic edge-halo cells (moved < 10%). One repeat: fresh film NEAR_MISS 8.8% → 4.4% (all three targets met, film only); orig G29 NEAR_MISS 57% → 48% (**not met**). The halo surrogate does not reproduce real rims; the remaining flags point to a systematic CD offset (real holes ~40 nm wider than design). Next: a CD-offset term in the testbed, or stop and wait for Rung 2.*

*Rung 1d (2026-10-23, `docs/REAL_DATA_RUNG1D.md`): testbed CD offset added; CD-corrected SDF energy **rejected on val** (pooled near-miss recall 37.5% vs 55.0%; its bias estimate reads wide halos as wider conductor). Adopted gate shifted slightly lower; real NEAR_MISS rose (fresh 4.4% → 7.4%, **not met**; G29 48% → 56%). Post-hoc: the estimator finds −24 nm on G29, matching the measured offset. Recommendation: stop tuning on these images; Rung 2 next; only remaining synthetic candidate is a joint bloom-width + CD estimate.*

- [x] Public, licence-clear real SEM images of simple periodic structures (line/space gratings, contact/via arrays, calibration specimens) with a stated pixel size. Download script + manifest only; no images committed.
- [x] Reference layout: no CAD exists, so fit a parametric layout (pitch, line width, angle, offset) on calibration regions only and render it. Labelled everywhere as a **FITTED reference, not CAD**: a fit can absorb a real defect.
- [x] Module 1 stand-in scribe patches: 3 named regions from calibration images.
- [x] Measure: wrong KILLER rate on clean structures (target ≤ 5%), calibration error, alignment error, run time. Pre-registered in `docs/REAL_DATA_PREREG.md`.

*Tells us:* whether the frozen verifier stays quiet on real noise, charging and edge contrast. *Does not tell us:* anything about recall.

### Rung 2: our own test chip with deliberate defects (real recall, exact ground truth)

*Status (2026-10-21): design done (`src/nanovate/real/testchip.py`, `scripts/real/make_testchip.py` → `results/real/testchip/<preset>/`: GDS, defect map, previews; every block's design Δβ checked by tests). Lab request drafted in `docs/LAB_REQUEST.md`, **not sent**.*

- [x] GDS test layout (line/space and via-array blocks, clean control blocks, bridges / opens / notches at a range of known sizes) + defect map CSV.
- [ ] Send `docs/LAB_REQUEST.md` (user) and agree on feature size, polarity and imaging time with the facility.
- [ ] Fabricate and image at a university nanofab / microscope facility (several doses / dwell times, a range of pixel sizes).
- [ ] Measure recall and false-alarm rate per defect type and size against the design, with true CAD this time.

*Kill condition:* if the fabricated defects below the facility's resolution cannot be confirmed by an independent image (e.g. high-dose or tilted SEM), they are dropped from the recall denominator and reported as unconfirmed, not as misses or hits.

### Real-wiring false-alarm test (M2), 2026-10-25

*`docs/REAL_DATA_M2.md`: frozen verifier on 64,536 real metal-2 tiles (318 images, commercial 128 nm IC) against validated labels. KILLER 0.31% [0.24, 0.39] (**met**), coverage 100% (met), NEAR_MISS 76.9% (**not met**: driven by bright vias the single-conductor-level model cannot represent; via-free tiles 36%). First real result meeting the killer target. Next: a two-level (track + via) conductor model, chosen on synthetic data with bright vias added.*

### Rung 3: public SEM defect datasets via reference-die mode

- [ ] Uses Phase 4's reference-die mode (clean neighbour image as the reference instead of CAD). Only after Phase 4's synthetic check passes its kill condition. *Phase 4 passed (2026-10-25): Rung 3 is unblocked.*

*Rung 3 run (2026-10-25, `docs/REAL_DATA_RUNG3.md`): reference-cell mode on the Puschner 90 nm chip, 429,232 cells: KILLER 28.0% (**not met**; fillers 43%, functional 12%), NEAR_MISS 1.1%, coverage 94.6%. False killers are extra components (dbeta0 +2 typical), likely a splice-seam artefact. Recall not measurable (positions withheld).*

### Rung 4: industry partner data

- [ ] Only after Rungs 1 and 2 have results. Under NDA; same pre-registration rules.

---

## Phase 4: Make real data possible (parallel)

- [x] **Reference-die mode:** clean die (or median of many dies) as the structural reference instead of CAD. (`src/nanovate/refdie.py`)
- [x] Validate reference-die vs CAD mode on synthetic data first (F1 cost). *2026-10-25 (`docs/PHASE4_REFDIE.md`): loss 4.6 F1 points [2.3, 6.9] (n = 1,152, val layouts 8–23): **kill condition not triggered**. The oracle arm (true reference wafer) loses the same, so the cost is the reference die's own LER, not segmentation. Particle recall 98% → 81%, filament called killer 32% → 62%, opens 70% → 79%.*
- [ ] Find a public SEM defect dataset with die-to-die references or repeated patterns; check licence, pixel size, label quality.
- [ ] Ask the university SEM facility about imaging test structures.

*Kill condition:* reference-die mode loses more than 10 F1 points vs CAD on synthetic → real-data work needs CAD access; pause and look for a data partner instead.

---

## Scrap list (decisions owed to the data)

| Candidate to scrap | Decided by | Kill condition |
|---|---|---|
| Stack path (Module 2) as default | Phase 1 | < 2% of verdicts changed, or no significant F1 gain |
| Global τ | Phase 2 | local rule beats it on val |
| Full persistent homology in the hot path | B2 | fast Betti path matches GUDHI exactly (**tested 2026-10-21: does not match, 66%; GUDHI stays**) |
| Pseudo-label hybrid | B1 | never beats verifier alone |
| Calibrated-simulation training | B1 | worse than Manhattan CNN on new layer |
| Our jitter fixes | B3 | industry-style correction already meets the target |
| Low-dose (5 e⁻/px) as a selling point | already decided | information isn't in the image; drop from pitch |
| Optical-discovery claims | already decided | defects unresolved; out of scope |

---

## Suggested order

1. **Track C first.** Send the Rung 2 lab request (`docs/LAB_REQUEST.md`) on day one, because facility access and fabrication take weeks; run Rung 1 on public images while waiting.
2. Phase 1 (+ Module 2 test) and B3 simulator additions: both validation-only, both cheap.
3. B2 fast Betti path: pure speed, zero verdict change, unblocks everything else.
4. Phase 2 core rule.
5. B1 agility comparison.
6. B3 corrections.
7. Phase 4 alongside, as data access allows (it is also Track C Rung 3).

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
