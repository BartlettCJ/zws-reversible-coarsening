# Preregistration — ZWSLab v0.1.4 Homogeneous-Initial-Condition Test

**Frozen before any outcome-generating run.** This study does not change the winner rule,
automaton semantics, v0.2, or the immutable discovery tags. It tests only whether changing the
large-scale spatial arrangement of a fixed state inventory changes the known domain regime.

## Question and hypotheses

Primary question: does the exact v0.1.4 winner spontaneously generate large irregular
excitation-rich/near-vacuum domains from a macroscopically homogeneous random state, or does it
mainly amplify the strong localized density inhomogeneity supplied by the discovery initial
condition?

- **H-A, spontaneous organisation:** the globally matched homogeneous arm develops reproducible
  finite near-vacuum domains and systematic low-wavenumber growth.
- **H-B, weak/conditional instability:** homogeneous states develop weaker but measurable domain
  or low-k growth.
- **H-C, seeded amplification:** the weak smooth gradient develops large structure but the
  homogeneous arm does not.
- **H-D, localization dependence:** only the hard localized patch enters the large-domain regime.
- **H-E, mixed/inconclusive:** the frozen rules below do not support one of A-D.

No hypothesis or threshold may be changed after Stage 1 outcomes are inspected.

## Frozen provenance

- Parent branch and commit:
  `followup/domain-mechanism-20260825` at
  `040e97452c77aa4d8063c3eea7b6ebd5f9e02553`.
- Study branch: `followup/homogeneous-initial-condition-20260826`.
- Winner rule file:
  `experiments/phase_separation_confirmation_20260825/rules/winner.json`.
- Exact winner table SHA-256 over the 256 uint8 entries:
  `d1fe99b5735d8e9be032347de2727dc5c52a7848df200ca26f67f28c8c1bc41c`.
- Immutable tags `v0.1.4-windows-discovery-freeze` and
  `discovery-d1fe99b5735d` must both remain at
  `74b4243c9518ec6f95b906de1a374bc277a2a5d5`.
- Design-generation namespace: `ZWSLab-homogeneous-ic-v1:2026082601`.
- Any derived integer seed is the low 31 bits of the big-endian integer represented by the first
  eight SHA-256 bytes of `namespace + ":" + label`.

## Frozen Stage 1 configuration

- World: 1024 × 1024 torus.
- Duration: 32,768 microscopic Margolus phases. Existing 1024² data show this comfortably spans
  the winner's growth and relaxation regime.
- Saved-sample and resumable-checkpoint interval: 256 phases.
- Paired seeds: 16 per arm.
- Coarse blocks: 16, 32, 64.
- Recurrence lags: 1 (for activity/reaction), 4, 8, 16, 32, 64, 128 phases. Lag 1 may be retained
  in machine output but is not promoted as a requested recurrence result.
- Device: one CUDA worker when available; existing resource governor, background process policy,
  file controls, and checksummed checkpoints are mandatory.
- Main arms A/B/C: 48 trajectories. Exploratory arm D: 16 trajectories. Total: 64.

### Frozen Stage 1 inventory seeds

In seed-index order 00-15:

`2019049721, 1098257983, 1091929321, 522236404, 481960949, 805102133,
1866549897, 1150730812, 1073990928, 1048271580, 1253183600, 2006309794,
714131700, 1894338784, 306028096, 700691882`.

Arm-specific arrangement seeds are derived with label
`arrangement:<arm_id>:<zero-based seed index>` and will be written explicitly into the frozen
manifest. Diagnostic shuffle seeds use label `diagnostic:<arm_id>:<seed index>`.

## Frozen initial-condition constructions

All state arrays are uint8 with semantics: 0 empty, 1 positive (+1), 2 negative (-1), and 3
bound-neutral (0). The exact inventory is measured with integer counts, not approximate fractions.

### A — localized patch (positive reference)

This reproduces the established 1024² reference construction. A centered square has side length
460 cells: `round(1024 * 0.45)` followed by truncation to the nearest lower even integer. Inside
the patch, the old balanced-random construction uses positive 12%, negative 12%, bound 8%, empty
68%, with integer flooring. Outside is empty.

Every Stage 1 A realization therefore contains exactly 25,392 positive, 25,392 negative, 16,928
bound-neutral, and 980,864 empty cells: `M = 67,712`, `Q = 0`, and global initial occupancy
0.0645751953125. Only the within-patch shuffle varies by seed.

### B — homogeneous random (primary test)

For each paired seed, take the exact flattened inventory generated for A and uniformly shuffle all
1,048,576 entries across the torus using the frozen B arrangement seed. Thus A and B have identical
positive, negative, bound-neutral, empty, `M`, and `Q` counts. No macroscopic mask, patch, or
gradient is applied.

### C — weak smooth gradient

For each paired seed, take the exact A inventory. Select its `M` occupied positions without
replacement with weights

`w(y,x) = 1 + 0.125*cos(2*pi*(x+0.5)/1024) + 0.125*cos(2*pi*(y+0.5)/1024)`.

The weight ranges smoothly from 0.75 to 1.25, has toroidal mean 1, and contains no discontinuity.
The occupied state labels are then shuffled over the selected positions. C therefore has the same
exact inventory, `M`, and `Q` as paired A/B; only its large-scale arrangement differs. Selection
and label shuffle use the frozen C arrangement seed and a documented NumPy PCG64 generator.

### D — exploratory homogeneous 32% arm

D is secondary and may not substitute for B. It uses the old patch-internal proportions across the
whole torus: exactly 125,829 positive, 125,829 negative, 83,886 bound-neutral, and 713,032 empty;
`M = 335,544`, `Q = 0`, occupancy 0.31999969482421875. It is uniformly shuffled and explicitly
excluded from inventory-matched A/B/C causal comparisons.

## Exact pre-simulation input audit

Before evolution starts, deterministic input generation must write, for every planned state:

- seed index, inventory seed, and arrangement seed;
- exact counts of states 0/1/2/3;
- exact `M` and `Q`;
- SHA-256 of the contiguous uint8 state;
- construction metadata and low-k baseline diagnostics.

The manifest generator must assert that every paired A/B/C seed has identical state counts, `M`,
and `Q=0`; B has no intentional spatial weighting; C uses the frozen weighting; and every expected
hash is unique within an arm/seed pair. The campaign must regenerate or load each state and verify
its frozen digest before the first update. An unusual-looking input is never regenerated.

## Measurements

At every saved phase, retain:

- exact `Q`, occupancy `M`, and all four state fractions;
- finite empty-domain characteristic scale (largest component excluded), finite component count,
  largest finite empty-domain area, connected-domain histogram, and largest total empty-domain
  fraction;
- peak and final finite-domain scale ratios;
- exact-state boundary/interface density and total empty-domain perimeter density;
- coarse-density variance at blocks 16/32/64;
- radial occupancy structure factor and characteristic wavelength;
- activity and reaction density;
- normalized exact-state entropy;
- exact-state and block-64 occupancy recurrence at lags 4/8/16/32/64/128;
- normalized occupancy spatial autocorrelation at displacements 1/2/4/8/16/32/64/128/256;
- directional low modes `(1,0)`, `(0,1)`, `(1,1)`, `(1,-1)`, `(2,0)`, `(0,2)` and radial bands
  `k=1..2`, `1..4`, `3..4`, `5..8`, and `9..16`, recording mean power, integrated power, and
  fraction of nonzero spectral power.

Low-k growth is evaluated from the B arm's radial `k=1..4` mean power and spectral fraction. For a
trajectory, **systematic low-k growth** requires peak mean-power ratio >= 4 and final mean-power
ratio >= 2 relative to tick 0, plus peak spectral-fraction ratio >= 2. Ratios use an explicit
`1e-30` numerical floor only if a floating value is zero; no missing value is silently imputed.

For initial, phase 4,096 (early), each trajectory's measured peak, and phase 32,768 (late), retain
or reconstruct exact states for common-scale rendering. Peak state retention is online and
metric-driven, not selected visually.

## Frozen per-trajectory domain criterion

For homogeneous states with no finite empty component at tick 0, define the initial finite-domain
scale denominator as the one-cell measurement floor. Otherwise use the measured initial scale.
This convention is fixed before outcomes and reported beside raw scales.

A trajectory is a **strong finite-domain trajectory** only if all four conditions hold:

1. peak finite-domain scale ratio >= 4;
2. final finite-domain scale ratio >= 2;
3. peak largest-finite-empty-domain area >=
   `max(64, 16 * max(initial largest-finite-empty-domain area, 1))`;
4. exact `Q` is preserved at every sampled phase.

The area gate requires an actual connected near-vacuum domain substantially larger than the
homogeneous baseline; dispersion or low-k power alone cannot pass the rule.

## Frozen Stage 1 interpretation classes

Let `pass_X` be the number of strong trajectories among 16 for arm X and let low-k passes refer to
the rule above.

- **A — STRONG SPONTANEOUS ORGANISATION:** `pass_B >= 12`, B median peak ratio >= 8, B median final
  ratio >= 4, and at least 12/16 B trajectories pass systematic low-k growth.
- **B — WEAK / CONDITIONAL INSTABILITY:** A does not apply, and either `pass_B >= 8`, or at least
  12/16 B trajectories pass low-k growth with B median peak ratio >= 2.
- **C — SEEDED AMPLIFICATION:** A/B do not apply, `pass_B < 8`, `pass_C >= 12`, and C median peak
  ratio is at least twice B's median peak ratio.
- **D — LOCALIZATION-DEPENDENT:** A/B/C do not apply, `pass_A >= 12`, `pass_B <= 4`, `pass_C <= 4`,
  B median peak ratio < 2, and C median peak ratio < 4.
- **E — MIXED / INCONCLUSIVE:** none of A-D.

These labels concern this winner, inventory, duration, and family of initial conditions; they are
not universal thermodynamic phase labels.

## Morphology and entropy review

Appearance is secondary to the frozen domain and spectral evidence. For each retained state record
radial spectral peak fraction, concentration in the strongest nonzero Fourier mode pair,
axis/diagonal low-mode concentration, and dense/interfacial/sparse state composition under the
established spatial masks. A state meets the **simple-regular spectral signature** only when the
upper-bound fraction in its strongest conjugate Fourier-mode pair is >= 0.15 **and** its strongest
radial shell contains >= 0.30 of all nonzero occupancy power. A simple regular/crystalline label
requires this repeatable sharp spectral signature in at least 12/16 B states and matching
common-scale visual geometry. Otherwise
domain-forming states are described by the measured morphology (irregular, labyrinthine, droplet,
stripe, or other) without forcing a classifier. Late dense-interior normalized entropy >= 0.8 is
the frozen threshold for “high-entropy internally.”

## Conditional Stage 2 at 2048²

Stage 2 is run only if B is meaningfully positive or surprising by one of these frozen gates:

1. `pass_B >= 8`; or
2. at least 12/16 B trajectories pass systematic low-k growth and B median peak ratio >= 2; or
3. at least 12/16 B final states meet the preregistered simple-regular spectral signature even if
   the finite-domain rule fails.

If triggered, run A/B/C at 2048 × 2048 for 32,768 phases and eight paired seeds, with sample and
checkpoint interval 256 and the same constructions scaled exactly. The reserved seeds are:

`1725827600, 1123306482, 1368436911, 634739919, 2105390133, 2128787032,
389793545, 1259535480`.

At 2048, the centered A patch side is 920 cells; B/C again receive each paired A inventory exactly.
No 4096² run is authorized by this preregistration.

## Reporting and decision

The final `HOMOGENEOUS_IC_REPORT.md` must answer all 13 requested questions and visibly separate
Measurement -> Formal inference -> Interpretation/speculation. Compact CSV/JSON evidence and
reproducible figures are committed; giant states/checkpoints stay under `results/` and out of Git.

The final study decision is one of:

- **A:** close v0.1 and move to v0.2;
- **B:** one further specifically named v0.1 experiment is justified;
- **C:** the homogeneous result is sufficiently surprising that this reversible null-conserving
  phase transition temporarily takes priority.

No v0.2 work starts automatically.
