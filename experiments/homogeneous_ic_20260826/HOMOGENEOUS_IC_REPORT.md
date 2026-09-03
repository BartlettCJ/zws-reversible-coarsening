# ZWSLab v0.1.4 homogeneous-initial-condition report

## Executive result

At the discovery-matched global occupancy (6.4575%), the frozen v0.1.4 winner **did not spontaneously create the large-domain regime from a macroscopically homogeneous random state**. The localized-patch reference passed the preregistered strong finite-domain criterion in 16/16 seeds; the exact-inventory homogeneous condition passed in 0/16, and the weak-gradient condition passed in 0/16. The formal preregistered classification is therefore **D — localization-dependent**.

The exploratory result adds an important qualification: homogeneous random initial states at 32% global occupancy formed very large domains in 16/16 seeds and showed systematic low-wavenumber growth in 16/16. The rule can therefore organize without a supplied boundary, but not at the much lower discovery-matched density tested in the primary comparison. The most defensible description is **density-conditional seeded phase amplification**.

## Frozen design and integrity

- Frozen winner SHA-256: `d1fe99b5735d8e9be032347de2727dc5c52a7848df200ca26f67f28c8c1bc41c`
- Lattice and duration: 1024 x 1024 for 32,768 phases
- Seeds: 16 matched seeds per arm
- Completed trajectories: 64/64; failures: 0
- Exact charge conservation: 64/64 trajectories preserved `Q = 0`
- Manifest SHA-256: `847ad4251f7f467ec561cc1bc75beb1a19af841da8374feddce56f63a590e6f2`
- Primary A/B/C inventory per seed: 25,392 positive, 25,392 negative, 16,928 bound-neutral, 980,864 empty; `M = 67,712`, `Q = 0`
- Primary A/B/C occupancy: 0.0645751953125
- Exploratory D inventory: `M = 335,544`, occupancy 0.31999969482421875; it is not an inventory-matched primary arm.

Only the large-scale spatial arrangement differs among A, B, and C. Exact initial-state hashes are frozen in [HOMOGENEOUS_IC_MANIFEST.json](HOMOGENEOUS_IC_MANIFEST.json) and audited in [INITIAL_STATE_AUDIT.json](INITIAL_STATE_AUDIT.json).

## Quantitative result

| Arm | Strong domain passes | Low-k growth passes | Median peak domain ratio | Median final domain ratio | Median peak low-k ratio | Median final occupancy | Median final entropy |
|---|---:|---:|---:|---:|---:|---:|---:|
| A localized patch | 16/16 | 0/16* | 22.464 | 12.855 | 2.539 | 0.1024 | 0.319 |
| B homogeneous, matched | 0/16 | 0/16 | 1.671 | 1.289 | 1.238 | 0.0653 | 0.225 |
| C weak gradient, matched | 0/16 | 0/16 | 1.713 | 1.315 | 1.098 | 0.0655 | 0.226 |
| D homogeneous, 32% exploratory | 16/16 | 16/16 | 86.054 | 36.071 | 58.721 | 0.5105 | 0.904 |

\*The low-k test is a growth-from-small-baseline criterion. A begins with a large low-k patch, so its 0/16 is not a failure of the localized positive control.

![Arm-level domain and low-k results](figures/stage1/stage1_arm_coordinate_summary.png)

The A/B/C snapshots use the same colors and rendering scale. A retains and elaborates its imposed macroscopic patch. B remains macroscopically uniform, noisy, and sparse. C retains only weak large-scale modulation and does not enter the A-like domain regime.

![Representative matched A/B/C snapshots](figures/stage1/snapshots/abc_snapshots_seed00.png)

## Answers to the preregistered questions

### 1. Does the winner form large domains from a macroscopically homogeneous random initial state?

**Measurement:** At matched 6.4575% occupancy, B passed the strong finite-domain rule in 0/16 runs. Its median peak and final domain-scale ratios were only 1.671 and 1.289.

**Formal inference:** No, not under the primary matched-density condition.

**Qualification:** The exploratory 32%-occupancy homogeneous arm did form large domains in 16/16 runs. The negative primary result is density-specific.

### 2. If yes, how reproducibly?

The matched homogeneous result was reproducibly negative: 0/16 strong-domain passes and 0/16 systematic low-k-growth passes. At exploratory 32% occupancy, the positive result was reproducible: 16/16 for both criteria.

### 3. How does its peak/final domain scale compare with the localized-patch case?

A's median ratios were 22.464 at peak and 12.855 at final. B's were 1.671 and 1.289. Thus the localized condition's median growth ratios were about 13.4 times larger at peak and 10.0 times larger at final than the matched homogeneous condition.

### 4. Do low-wavenumber fluctuations grow from the homogeneous initial state?

Not systematically at matched occupancy. B passed the frozen low-k rule in 0/16 runs; its median peak and final low-k mean-power ratios were 1.238 and 1.212. By contrast, D passed in 16/16 with median peak and final ratios of 58.721 and 57.360.

### 5. Does the weak-gradient condition get amplified?

Not into the large-domain regime. C passed neither the strong-domain nor low-k rule in any of 16 runs. Its median peak/final domain ratios of 1.713/1.315 are close to B's 1.671/1.289, while its low-k ratios of 1.098/1.092 do not indicate sustained amplification.

### 6. Is a sharp localized boundary required?

At the discovery-matched global occupancy, a strong localized density inhomogeneity was required among the tested A/B/C conditions. It is not universally required: D organized from homogeneous random states at higher density. The evidence therefore supports a density-dependent basin or instability threshold, not a rule that literally requires a boundary.

### 7. Does total global occupancy remain matched between A/B/C?

Yes. For every matched seed, A/B/C have identical counts: 25,392 positive, 25,392 negative, 16,928 bound-neutral, and 980,864 empty, giving `M = 67,712` and exact `Q = 0`. All frozen input hashes and inventories passed the pre-run audit. D is explicitly excluded from this matched comparison.

### 8. Are the homogeneous-condition patterns irregular or simple regular geometry?

B never developed a macroscopic domain morphology; it remained sparse, noisy, and macroscopically homogeneous in the reviewed snapshots. No B trajectory met the simple-regular spectral signature. D developed large structure but likewise produced 0/16 simple-regular spectral signatures, arguing against crystallization, stripes, or checkerboards as the explanation for its low-k growth.

### 9. Is the late state still high-entropy internally?

Where a genuine dense region exists, yes. The median normalized entropy of the final dense interior was about 0.90 for A and 0.99 for D. B and C have no comparable large dense domain; their quantile-selected “dense” subsets remain sparse (about 0.08 occupied), so treating their local entropy as a domain-interior property would be misleading.

### 10. What is the best description of the phenomenon?

**Formal primary classification:** D — localization-dependent.

**Scientific description:** density-conditional seeded phase amplification. At low global density the winner elaborates a supplied macroscopic concentration but does not create one from homogeneous noise or a weak gradient. At higher global density it can spontaneously generate long-wavelength structure without an imposed boundary.

### 11. What is the strongest scientific conclusion justified?

The previously observed “golf-course” domains are not, by themselves, evidence that the winner is linearly unstable to tiny homogeneous fluctuations at the discovery-matched density. They depend strongly on the initial large-scale density arrangement. However, the same frozen rule exhibits a reproducible spontaneous large-scale instability in a denser homogeneous regime, showing that both initial localization and total density determine which dynamical regime is reached.

### 12. Does this materially change the recommendation to move primary effort to v0.2?

It narrows the v0.1 mechanism claim and reveals a useful phase boundary, but it does not justify reopening rule search or displacing v0.2 as the main program. One short, preregistered v0.1 density-threshold experiment is warranted before closure; after that, primary effort should move to v0.2.

### 13. Implications for Zero Without Stasis

**Measurement:** Exact global `Q = 0` was preserved in all 64 runs while occupancy, entropy, and domain morphology changed substantially.

**Formal inference:** Null conservation permits both homogeneous sparse evolution and high-entropy phase organization. Conservation alone does not select the macroscopic regime; density and spatial arrangement matter.

**Speculation:** This is compatible with a “Zero Without Stasis” intuition: exact cancellation need not imply inactivity or uniformity. The stronger statement—that null balance generically causes organization—is not supported. What is supported here is a conditional example in which exact null balance coexists with active, internally disordered, large-scale structure.

## Stage-2 decision

The frozen 2048 x 2048 escalation gate did not pass, because the primary matched homogeneous condition B showed neither strong domains nor systematic low-k growth. No Stage-2 run was performed, as preregistered.

## Recommended decision

**B — one further targeted v0.1 experiment is justified.**

Run one fixed homogeneous-occupancy sweep at 1024 x 1024 using the frozen winner and the same charge-balanced occupied-state proportions, with no localized patch. A preregistered grid between 6.46% and 32% occupancy can locate the onset of the spontaneous high-density regime. This is the minimum experiment needed to distinguish a sharp threshold from a gradual crossover. Do not start a new rule search, and do not start or modify v0.2 as part of that sweep.

## Evidence index

- [Machine-readable decision](artifacts/stage1/STAGE1_DECISION.json)
- [Per-trajectory coordinates](artifacts/stage1/TRAJECTORY_COORDINATES.csv)
- [Arm summaries](artifacts/stage1/ARM_SUMMARIES.csv)
- [Paired comparisons](artifacts/stage1/PAIRED_COMPARISONS.csv)
- [Spatial composition audit](artifacts/stage1/SPATIAL_COMPOSITIONS.csv)
- [Analysis artifact hashes](artifacts/stage1/ANALYSIS_ARTIFACTS.json)
- [All paired-seed plots](figures/stage1/paired/)
- [All A/B/C snapshot grids](figures/stage1/snapshots/)

## Interpretation boundary

The classification and escalation decision follow the frozen quantitative rules. Visual morphology is used only as secondary description. The exploratory D arm is evidence about density dependence, not a substitute for the exact-inventory A/B/C comparison.
