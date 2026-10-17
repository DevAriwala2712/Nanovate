# Defect Detection System Plan v4.0

Oct 17, 2026 · @Boss

v4.0 builds one thing in the hack: a training-free SEM defect verifier, measured on synthetic data against honest baselines. It flags a killer only when circuit topology differs from CAD. Everything else from v3.0 stays below, as future R&D (Part B) or parked claims (Part C).

## Part A — Implementable core

### Thesis

**A training-free SEM defect verifier that calls a defect a killer only when circuit topology differs from CAD, calibrated per layer from three scribe-line patches.**

The genuine potential is the topology check plus near-miss scoring. It needs no labels, ignores geometry style, and explains each verdict. That is what the hack builds and measures.

### What survives from v3.0

| Component | Verdict | Why |
| --- | --- | --- |
| Synthetic SEM testbed (layouts, LER, grain, bridge / open / particle / near-miss, Poisson dose) | Build (48h) | NumPy/SciPy/gdstk work; every number you show comes from it |
| Module 1: scribe-line ridge calibration | Build (48h) | 5-feature least squares; a few hours |
| Module 2: same-layout patch stack + randomized SVD residual | Build (48h) | NumPy SVD on K = 32 stacks; singletons fall back to a simple denoiser |
| Module 3: phase-correlation alignment + SDF tension energy | Build (48h) | Code already sketched; drop the Langevin curvature step |
| Module 4: persistent-homology Δβ verdict | Build (48h) | Already in the repo; harden the threshold τ |
| Bottleneck-distance near-miss rule | Build (48h) | One extra GUDHI call; the real differentiator |
| Die-to-die and small-CNN baselines | Build (48h) | Without them the numbers mean nothing |
| Self-supervised blind-spot denoiser (Noise2Void-style) | Stretch | Feasible on synthetic data; fix the Dropout2d bug first |
| Persistence-image few-shot classifier (bridge / open / particle / near-miss / nuisance) | Stretch | Synthetic topology classes only; real root-cause bins need fab data |
| GPU batched randomized SVD | Stretch | Only if a GPU is free; CPU numbers are fine for the demo |
| FNO / DOINN lithography PVB thresholds | Future R&D | Needs trained litho surrogate and validation (Part B) |
| Discrete Morse GPU homology | Future R&D | Custom CUDA engineering (Part B) |
| Multi-detector SEM (BSE / voltage contrast) fusion | Future R&D | Needs real tool data (Part B) |
| 3D Gaussian-splatting X-ray laminography | Future R&D | Needs synchrotron data (Part B) |
| ASPH-GNN + Causal-KAN cross-layer engine | Future R&D | Needs fab telemetry (Part B) |
| Optical BBP insertion point (replacing aiSIGHT) | Parked | 30–100 nm optical pixels can't resolve the wires (Part C) |

### Pipeline

&#91;embedded content: v4.0 pipeline · 4 modules, 3 verdicts\]

The die-to-die and CNN baselines run on the same patches in every experiment. The template asks for a team-drawn diagram, so redraw this one in your own style for slide 4.

### 48-hour build plan

The plan turns the existing topology repo into a measured system by hour 48. Everything after that is stretch.

1. **H0–4: Freeze the testbed spec.** 256 × 256 px at 1 nm/px. Three layout families: Manhattan lines (14 nm half-pitch), curvy spline wires, via/pillar arrays. Fix all random seeds.
2. **H4–12: Build the testbed.** Inject nuisance: Ornstein-Uhlenbeck LER (σ 0.5–2.5 nm) and Voronoi grain (contrast up to 30%). Inject defects: bridges (radius 3–10 nm), opens, particles, near-miss notches and filaments. Apply the SEM forward model: blur, edge bloom, charging ramp, Poisson dose λ 5–200 e⁻/px. Save (CAD, image, mask, label) sets.
3. **H12–18: Harden Module 4.** Sweep τ on clean vs. defect pairs and pick it per layout family. Add the bottleneck-distance near-miss rule. Write one unit test per defect class.
4. **H18–24: Modules 1 and 3.** Ridge calibration on 3 large-feature "scribe" patches. Phase-correlation alignment to CAD. SDF energy to rank candidate residuals.
5. **H24–30: Module 2.** Stack 32 same-layout patches, rank-2 randomized SVD, MAD threshold. Singleton layouts fall back to a median or Gaussian denoise.
6. **H30–36: Baselines.** Die-to-die difference with a threshold. A small CNN trained on Manhattan layouts only.
7. **H36–44: Run the six experiments** in the next section. Save CSVs and plots.
8. **H44–48: Package.** README with one results table, a demo notebook, and one command that regenerates every number.

**If time runs short**, cut in this order: Module 2, then Module 1, then the CNN baseline. Never cut the die-to-die baseline or the dose sweep.

**Stretch, after H48:** the blind-spot denoiser, the persistence-image few-shot classifier, then a GPU path.

### Experiments and the numbers to show

Six experiments give every number on the slides. The targets below are pre-registered hypotheses: state them today, fill the measured column by H44.

| # | Experiment | Sweep | Metric | Target to state | Compared against | Measured |
| --- | --- | --- | --- | --- | --- | --- |
| 1 | Nuisance stress | LER σ 0.5–2.5 nm; grain contrast 0–30% | Nuisance suppression %; bridge recall | ≥ 90% suppression and ≥ 90% recall at σ ≤ 1.5 nm | Die-to-die threshold |  |
| 2 | Dose cliff | λ 5–200 e⁻/px | F1 vs. dose; lowest dose with F1 ≥ 0.9 | F1 ≥ 0.9 at λ ≥ 50 | Die-to-die; CNN |  |
| 3 | Zero-shot geometry transfer | Calibrate or train on Manhattan; test on curvy and via layouts | F1 drop, in points | Topology drop ≤ 5 points; CNN drop reported as measured | Manhattan-trained CNN |  |
| 4 | Misalignment | Shift ±8 px plus small warp | % verdicts unchanged vs. zero shift | ≥ 95% unchanged | Same pipeline without Module 3 |  |
| 5 | Near-miss capture | Notches and filaments 1–3 nm short of shorting | Recall at a fixed false-alarm rate | ≥ 80% | Δβ rule alone |  |
| 6 | Calibration and latency | 3 scribe patches; 1,000 patches on CPU | ms per calibration; ms per patch | Calibration < 50 ms | CNN on the same hardware and patch size |  |

Revise any target after the first run if it is clearly wrong. A missed target reported honestly still scores better than an unexplained miracle number.

**Presenting numbers in today's PPT:**

- **Problem-side numbers** can come from the bottlenecks reference doc: 50,000–500,000 nuisance flags per wafer, 1–3 weeks of recipe tuning per layer, 500–5,000 labels per defect class. Verify each against a public source before quoting it.
- **Solution-side numbers** go in a "Target" column, plus any preliminary result the current repo already produces, labelled "preliminary, synthetic data".
- **Show the empty results table** with its baselines. A pre-registered table with a falsification test reads as more expert than a large multiplier.
- **Leave out** multipliers vs. KLA, dollar ROI, and the word "proven" (Part C lists them).

### Slide-by-slide mapping (BIOS template)

Each template slide draws only from Part A; Part B earns at most one line on slide 5.

| Slide | What goes on it |
| --- | --- |
| 1. Problem, stakeholder & SDGs | Challenge ID (fill in). Problem: new layers flood review SEMs with nuisance flags, and supervised filters need weeks of labels. Stakeholder: fab yield and process engineers, review-SEM operators. Evidence: verified problem-side numbers. SDG 9 primary; SDG 12 supporting (less scrap). |
| 2. Existing approaches & gap | Die-to-die subtraction with adaptive thresholds; supervised CNN filters and ADC (aiSIGHT / SEMVision class). Close with: "The gap we are addressing is: no inspection method can tell a killer defect from harmless roughness on a new layer without weeks of labelled data." |
| 3. Proposed solution | Topology verifier. Input: SEM patch + CAD clip. Output: killer / near-miss / nuisance verdict with its Δβ reason. Differentiators: training-free, geometry-agnostic, explainable. |
| 4. Technical approach | The pipeline diagram above, redrawn by the team. Stack: Python, NumPy/SciPy, gdstk, GUDHI, PyTorch (baseline CNN only). Mark which boxes the hack implements. |
| 5. Build plan, risks & impact | Build: the 48-hour plan. Risks: the list below. Impact metrics: nuisance suppression % and setup time per new layer. |

### Risks

- **Sim-to-real gap.** No real SEM data; every number is synthetic. Say so on the slide.
- **Threshold sensitivity.** One global τ can miss faint bridges; calibrate it per layer from Module 1.
- **Singleton-heavy logic.** Module 2 needs repeated layouts; non-repeating blocks fall back to a weaker denoiser.
- **Scribe-to-die transfer.** Calibration on large scribe features may mis-predict dense arrays.
- **Patch-boundary topology.** Wires may connect outside the crop; compare CAD and image in the same window with a margin.

## Part B — Theoretical R&D horizon

Every v3.0 pillar stays here as future work; none is claimed for the hack. On the PPT, these fit in one "future directions" line.

| Component (from v3.0) | What it would add | What it needs that the hack lacks |
| --- | --- | --- |
| FNO / DOINN lithography surrogate (PVB-guided thresholds) | Tighter thresholds at lithography hotspots | A trained litho model and proof it transfers to the layer's real litho conditions (High-NA EUV is not shown) |
| Self-supervised denoiser on real SEM | Lower electron dose at the same accuracy | Real SEM stacks; a patch-wise blind spot for scan-correlated noise |
| Persistence-image few-shot ADC on real root-cause bins | Semantic bins: particle, residue, resist footing, void | Labelled review-SEM images; BSE and voltage-contrast channels |
| Discrete Morse homology on GPU | Throughput at fab data rates | CUDA engineering; a benchmark against GUDHI |
| 3D Gaussian-splatting X-ray laminography | Buried GAA and backside-power voids (Δβ₂) | Synchrotron beamtime and 3D data |
| ASPH-GNN + Causal-KAN cross-layer engine | Root-cause attribution to process chambers | Fab telemetry across layers; evidence that ASPH properties hold for wafers |
| High-NA EUV stochastic resist modelling | Priors for random nano-bridges and missing contacts | Resist chemistry models and EUV data |
| Vendor integration (Stage-2 GPU blade, review-SEM ADC slot) | A real deployment path | Vendor data access and GDSII licensing |

## Part C — Parked v3.0 claims

These claims are kept for the record but stay off the PPT until an experiment backs them. Several have a measurable replacement in Part A.

| Claim (v3.0) | Why parked | What would make it usable |
| --- | --- | --- |
| 270,000× faster setup | Compares a 0.01 s fit with weeks of human tuning; not like-for-like | Report measured calibration ms and what it replaces (Experiment 6) |
| 26.9× faster execution; CNN at 8,400 ms per 10k patches | CNN baseline roughly 100× too slow; patch sizes differ | Time both on the same hardware and patch size (Experiment 6) |
| 28.7× fewer false alarms; 99.1% purity; 98.6% suppression | No experiment run | Experiment 1 |
| 13.3–40× lower e-beam dose | No denoiser built or measured | Experiment 2, with and without the denoiser |
| 96.4% 25-bin ADC accuracy from 9–50 shots | Real root-cause bins can't be simulated | Measured accuracy on synthetic topology classes only |
| 97.4% mousebite capture | Not measured | Experiment 5 |
| "100% invariant" Betti verdict | Holds for the ideal field, not noisy images | Experiment 4 agreement rate |
| 14× less VRAM; 6.7× fewer scans | Results of a cited ptychography paper, not this system | Cite as related work only |
| Dollar ROI, KLA market share, "foundry VPs deploy immediately" | Market pitch; the template discourages it | One sentence of stakeholder value on slide 1 |
| Replacing aiSIGHT on the optical BBP blade | Optical pixels can't resolve the wires | Drop; scope is SEM-resolution only |
| ASPH supercell-size convergence applied to wafers | Materials-science result; transfer unshown | Cite as inspiration only |
| Irrelevant citations (Snapdragon, helium, CSDN, L-BFGS, newsletters) and raw \[cite\] markers | Signal unedited AI output | Remove from anything submitted |

## Appendix — Review of v3.0 (why v4.0 changed)

### Bottom line

There is a strong, defensible project inside this plan, but the document itself would hurt it as a proposal. It reads as a research-program vision rather than something a team intends to build. A technical judge will likely see the overclaiming and scope before reaching the good idea.

Unproven numbers are fine at this stage. The real problem is that the doc presents them as proven, and that a few core claims contradict the physics in the project's own reference documents.

Rough scores: as a research vision, **8/10** for ambition and idea quality. As a BIOS screening proposal in its current form, about **4/10**.

### What's genuinely strong

The central insight is good and not obvious. It separates "is this anomaly a killer?" from "does this anomaly look unusual?" by checking whether circuit **topology** changed (Betti numbers versus CAD), not whether pixels changed. That turns the nuisance problem into a well-posed mathematical question.

Using bottleneck distance with Δβ = 0 to catch near-miss mousebites and partial bridges is a smart extension. It covers the obvious weakness of a binary topology check.

Scribe-line ridge-regression calibration is simple, cheap, and directly targets the cold-start story. The "where does pure math fail" audit (Gaps 1–4) shows self-critical thinking. The falsification-matrix mindset is exactly what screeners want to see, and the problem framing is well grounded in the bottlenecks reference doc.

### Problems, ranked by severity

#### 1. Scope is the biggest issue

Plan 3.0 covers optical inline filtering, SEM denoising, ADC classification, FNO lithography surrogates, X-ray ptychographic laminography with Gaussian splatting, and a fab-wide causal GNN. That is several PhD theses and multiple vendor product lines.

The BIOS template asks what "can realistically become a working prototype during the hack" and warns against Shark Tank language. This doc has a ROI section, KLA market share, and "foundry VPs will deploy immediately." Every pillar added without a build plan lowers credibility on the pillars that can be built.

#### 2. The claims language works against the project

The doc uses "proven multiplier," "indisputable breakthroughs," "empirical benchmarks prove," and "100% invariant." It reports figures like 97.4%, 96.4%, 270,000× and 99.1% purity with no experiments run. The surgical protocol's scorecard is formatted like measured results. An expert will discount every number after spotting one fabricated-looking precise figure.

The citation list makes this worse. It includes a Snapdragon launch article, a helium supply blog, a CSDN quantization post, the L-BFGS paper, an investment newsletter, and raw `[cite: 2, 3, 5]` markers. That pattern signals unedited AI-generated research, which judges screening "builders" actively look for.

ASPH (as far as I know, *atom*-specific persistent homology from materials science) is relabeled "defect-specific." Its supercell-convergence property is then cited as if it applies to wafers, which is a stretch.

**Fix:** relabel numbers as target hypotheses, give each a falsification criterion, and drop false precision ("target F1 > 0.9," not "0.96").

#### 3. Insertion Point A contradicts the project's own physics

The bottlenecks doc says optical BBP pixels are 30–100 nm and defects appear as featureless 3×3 smudges. At a 14 nm half-pitch, wires aren't resolved, so meaningful Betti numbers of the circuit can't be computed from those images. The topological verifier only makes sense at SEM-class resolution. The headline claim of replacing aiSIGHT on the optical Stage-2 blade does not hold.

The synthetic testbed hides this. Its optical mode uses an Airy radius of 8–15 px at 1 nm/px, which is 8–15 nm. A real 193 nm system at NA \~0.9 has a Rayleigh radius around 130 nm. The simulated optics are roughly 10× too sharp, so optical results from that testbed would be inflated.

**Fix:** position the method as an e-beam / review-SEM engine, where its logic holds.

#### 4. The baselines are strawmen

The doc claims a CNN takes 8,400 ms per 10k patches. A TensorRT INT8 ResNet-18 on 64×64 patches is roughly 0.15 GFLOPs each, so 10k patches is about 1.5 TFLOPs. A modern GPU handles that in well under a second. The comparison also mixes 64×64 CNN patches with 256×256 patches.

The more important missing baseline is plain die-to-die differencing. It already exploits repetition, because the same location on the neighbouring die has identical CAD. The plan needs to show what CAD-hashed within-die RPCA adds over that, on SEM-resolution data.

#### 5. Technical points a careful reviewer would raise

- **Topology moves the threshold, it doesn't remove it.** With τ = 0.35 on a percentile-normalized image, a bridge at normalized brightness \~0.5 leaves both components with lifetime ≥ 0.35. β₀ doesn't change and the bridge is missed. τ needs physical calibration, possibly from the Module 1 fit. The "homeomorphic invariance" guarantee applies to the ideal field, not a noisy image at SNR ≈ 2.
- **The stochastic-defect reasoning in surgical doc §2.1 is confused.** Random, non-repeating defects are exactly what the sparse term S captures; L comes from CAD repetition regardless. The real stochastic-EUV problem is low contrast versus noise.
- **Scribe-line calibration may not transfer.** Scribe targets are large and isolated, while in-die patterns are dense. Proximity, charging and edge-interaction effects differ, so a 5-feature linear fit may not predict dense-array appearance. This is testable and worth stating as a risk.
- **Blind-spot denoising assumes pixel-independent noise.** SEM noise is often correlated along the scan direction, which is why patch-wise blind-spot methods exist; the code implements a pixel blind spot. The blind spot does survive the dilated layers, thanks to offset parity. However, `nn.Dropout2d` on a single-channel input zeroes the entire image 15% of the time, which is a bug.
- **The thesis drifts.** The surgical protocol's identity is "training-free at t = 0." Plan 3.0 adds a pre-trained FNO, a few-shot classifier, per-patch fine-tuning and a GNN. That evolution is legitimate, but the doc needs one clear thesis.

#### 6. Fit to the BIOS template

The doc has no SDGs, no scoped build plan, no honest risk section, and no simple architecture diagram the team clearly owns. Its strongest material maps onto slides 1–2. Slides 4–5, which the template calls most important, would currently be the weakest.

### How to reposition it

Make the pitch: *"A training-free SEM defect verifier that separates killer defects from nuisance by checking circuit topology against CAD."*

Build only the synthetic SEM-mode testbed plus Modules 1, 2 and 4 (SDF energy optional). Compare against die-to-die differencing and a small CNN trained on one geometry domain and tested zero-shot on another.

That gives a live demo: inject LER, grain texture, bridges and opens, then show nuisance-suppression rate and bridge recall as dose is swept. The zero-shot transfer result, if it holds, is the headline. FNO, X-ray and GNN go in a single "future directions" line.

**SDGs:** SDG 9 (Industry, Innovation and Infrastructure) as primary; SDG 12 (scrap and waste reduction) as a reasonable supporting one.

**Genuine risks for slide 5:**

- Sim-to-real gap, with no real SEM data
- Persistence-threshold sensitivity
- Singleton-heavy logic layouts defeating RPCA
- Scribe-to-die calibration transfer

With this rescoping, the project could be one of the stronger technical entries. The core idea is sound, and the falsification framing is better than most hackathon proposals have.
