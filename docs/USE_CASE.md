# Why use this: the day-0 inspector for a new layer

**One sentence.** On the first wafers of a new layer or layout, before anyone has labelled defect images, this
verifier sorts review-SEM sites into killers, near-misses and harmless roughness from the CAD and three scribe
images, and says why for every call; today's trained classifiers do not exist yet at that point.

All evidence below is from the synthetic testbed (see the README's limitations). The numbers in the tables are
generated from the result CSVs; this page contains no hand-typed results.

## Who and when

- **Who:** a process or yield engineer bringing up a new layer (a new node, a new metal or via level, a
  curvilinear or ILT layout) on a review SEM.
- **When:** from the first wafer. Existing review-SEM classifiers (KLA aiSIGHT, AMAT SEMVision ADC; see
  `existing_semiconductor_inspection_systems_bottlenecks.md`) are trained per defect class and per layer. Until
  enough labelled images exist, the fab falls back to die-to-die differencing plus manual review.
- **Later:** once a trained classifier exists, the verifier stays useful as a cross-check grounded in the CAD:
  every call it makes comes with its reason (which wires merged or split, how close a near-miss came).

## What it replaces on day 0, and what it does not

- It replaces **die-to-die differencing + manual review** on a new layer. Die-to-die flags pixel differences,
  so roughness and grain between dies look like defects; the verifier asks whether the circuit's connectivity
  changed.
- It does **not** beat a classifier trained on the same layout, and it is slower per patch. Those losses are in
  the table below on purpose: the claim is "usable before any labels exist", not "better than everything".
- It adds a **near-miss bin** the others lack: a wire pinched but not cut, or a filament that stops short of its
  neighbour, is reported as NEAR_MISS (a reliability risk) instead of being forced into killer or harmless.

## The evidence

<!-- DAY0:START -->
_All numbers are **synthetic data**; the review queue is a **projection** from measured per-kind rates (via family, dose 50) with an assumed site mix._

#### Day 0 on a new layer: what each option gives you

|  | Verifier (ours) | CNN (Manhattan-trained) | Die-to-die (single) |
|---|---|---|---|
| Labelled images needed before first use | none (3 scribe images per layer) | a labelled training set (Manhattan here) | labelled val per dose, to fit the rule |
| F1 on a new layout: via | 0.88 [0.82, 0.93] (n=144) | 0.74 [0.66, 0.81] (n=144) | 0.56 [0.45, 0.65] (n=144) |
| F1 on a new layout: curvy | 0.84 [0.77, 0.90] (n=144) | 0.85 [0.79, 0.90] (n=144) | 0.55 [0.45, 0.66] (n=144) |
| F1 on the layout the CNN was trained on (Manhattan) | 0.81 [0.73, 0.87] (n=144) | 0.94 [0.89, 0.97] (n=144) | 0.57 [0.45, 0.67] (n=144) |
| Setup per layer | 43 ms calibration | training run (GPU) | rule fit |
| Time per patch (this CPU) | 99 ms | 3 ms | 31 ms |
| Says why it flagged a site | yes: delta-beta, bottleneck, SDF energy | no (a probability) | no (a difference map) |

Where the verifier loses: on a layout the CNN was trained on, the CNN is more accurate, and it is much faster per patch.

#### Review queue: 1,000 flagged sites on a new via layer, 5% real killers (50 sites)

| Sites (95% CI) | Verifier (no labels) | CNN (Manhattan-trained) | Die-to-die (single) | Die-to-die (double) |
|---|---|---|---|---|
| Killers caught (flagged KILLER) | 45 [42, 48] | 38 [34, 41] | 26 [22, 29] | 31 [26, 35] |
| Killers flagged NEAR_MISS (still reviewed) | 3 [1, 6] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] |
| Killers missed (passed as harmless) | 2 [0, 4] | 12 [9, 16] | 24 [21, 28] | 19 [15, 24] |
| Non-killer sites flagged KILLER (incl. near-misses) | 22 [12, 34] | 73 [28, 146] | 46 [40, 51] | 103 [61, 176] |
| Near-misses flagged NEAR_MISS (reliability bin) | 73 [61, 83] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] |
| Harmless sites flagged NEAR_MISS | 36 [0, 107] | 0 [0, 0] | 0 [0, 0] | 0 [0, 0] |
| All sites sent to an engineer | 179 [142, 250] | 111 [66, 185] | 71 [63, 78] | 133 [91, 209] |

Site mix assumed: killers split equally bridge / open / particle; harmless sites 90% roughness-only, 5% notch, 5% filament (near-misses). Only the verifier has a NEAR_MISS bin; the others must call a near-miss killer or harmless.

Other prevalences (killers missed / non-killer sites flagged KILLER):

| Real killers | Verifier (no labels) | CNN (Manhattan-trained) | Die-to-die (single) | Die-to-die (double) |
|---|---|---|---|---|
| 1% | 0 [0, 1] / 23 [12, 35] | 2 [2, 3] / 76 [29, 153] | 5 [4, 6] / 47 [41, 54] | 4 [3, 5] / 107 [64, 184] |
| 20% | 8 [0, 17] / 18 [10, 28] | 50 [36, 64] / 62 [23, 123] | 97 [83, 111] / 38 [33, 43] | 78 [61, 94] / 87 [52, 148] |

#### Label break-even: labelled images of the new layout the CNN needs to match the verifier

| Family | CNN threshold | Verifier F1 | Break-even k (seed-mean F1 >= verifier) | Clearly better k (every seed's CI above) | n |
|---|---|---|---|---|---|
| curvy | fixed_0.39 | 0.84 | 0 | 48 | 3 seeds, n=144 |
| curvy | labelled | 0.84 | 0 | 192 | 3 seeds, n=144 |
| via | fixed_0.39 | 0.88 | 192 | none | 3 seeds, n=144 |
| via | labelled | 0.88 | 192 | none | 3 seeds, n=144 |
<!-- DAY0:END -->

## What would change this conclusion

- Real SEM images behave differently from the testbed (the largest risk; see the README's sim-to-real limitation).
- A classifier that adapts to a new layout from very few labels would shrink the day-0 window; the label
  break-even above measures exactly that for a fine-tuned CNN.
- Sites the optics cannot resolve (2-3 nm necks or gaps) are confused by every image-based method, including this one.
