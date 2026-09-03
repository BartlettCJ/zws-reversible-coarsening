# Preregistration — ZWSLab v0.1.4 Domain-Mechanism Follow-up

**Frozen:** 2026-08-25 JST

**Branch:** `followup/domain-mechanism-20260825`

**Parent result commit:** `970d1f118836d04fb2d5ab64448c418ca05094b2`

**Frozen discovery rule-table SHA-256:** `d1fe99b5735d8e9be032347de2727dc5c52a7848df200ca26f67f28c8c1bc41c`

**Initial frozen-plan SHA-256:** `a0e79dc61db34a44f3329d7a98d59e2f965a0f6e43b0c0382cbd18eeb4b145f9`

This document fixes the follow-up questions and analysis rules before any new trajectory is run.
It does not revise the original confirmation preregistration or its strict classification D. The
established observation to explain is reproducible, very large transient domain formation followed
by relaxation, not sustained late-time coarsening.

No v0.2 package will be installed or merged in this study.

## Evidential vocabulary

Every report will keep these levels visibly separate:

1. **Measurement:** a directly computed exact value or declared statistic.
2. **Formal inference:** a conclusion entailed by exact audits or a conclusion under a frozen
   decision rule.
3. **Interpretation/speculation:** a proposed mechanism or broader implication not uniquely forced
   by the measurements.

Images and event bookmarks remain review aids. No outcome will be described as evidence of life,
organisms, replication, heredity, agency, consciousness, primitive occurrence, ETNE, cosmology,
or physical nullity.

## Frozen questions

### H1 — Composition

What exact local states constitute the dense and sparse domains? Specifically, do positive and
negative states rise together, are neutral bound excitations enriched, are interfaces
compositionally distinct, and is occupancy rather than signed charge the principal spatial order
parameter?

### H2 — Dynamics

After the very large early/mid-time domain scale, does the system approach a nonzero plateau,
continue slowly, undergo damped or persistent oscillation, enter a larger macrocycle, return toward
the initial scale, or occupy multiple metastable regimes?

### H3 — Second exact invariant

Does the organization survive when every local update preserves both signed charge `Q` and an
independent occupancy/mass-like quantity `M` exactly?

### H4 — Same-family prevalence

Among independently frozen 24-reactive-input rules on the same rotational scaffold, is winner-like
domain growth common, moderately unusual, rare, or part of a distinct behavioural cluster?

These questions and the decision rules below will not be changed after new outcomes are inspected.

## Exact state semantics and candidate order parameters

| Code | Meaning | Charge `q(s)` | Occupancy `m(s)` |
|---:|---|---:|---:|
| 0 | empty | 0 | 0 |
| 1 | positive excitation | +1 | 1 |
| 2 | negative excitation | -1 | 1 |
| 3 | bound/neutral excitation | 0 | 1 |

For a lattice or local block,

`Q = count(state 1) - count(state 2)` and
`M = count(state 1) + count(state 2) + count(state 3)`.

`M` is not a disguised copy of `Q`: blocks can share `Q` and differ in `M` (empty-empty has
`Q=0,M=0`, while positive-negative has `Q=0,M=2`), and can share `M` while differing in `Q`.

The candidate separating fields, retained separately, are occupancy, bound-state fraction,
charged-state fraction, positive fraction, negative fraction, and local signed charge. State-
specific structure factors and conditional compositions will decide which description is most
specific; the answer will not be assumed to be generic “density.”

## Existing-data analysis fixed before extraction

The primary existing data are:

- the 4,096² discovery checkpoints every 2,048 phases and exact event clips;
- all eight retained 2,048² Stage 7A winner trajectories;
- all four retained 4,096² Stage 7B winner trajectories;
- paired rotation and matched-control scalar series;
- exact final-state batches in the confirmation checkpoints.

Because the automaton is bijective, retained final states may be streamed backward to reconstruct
intermediate frames exactly. This is analysis of already completed trajectories, not a new-seed
simulation. Reconstructed states must match the stored initial hash and every available archived
checkpoint hash.

### Spatial sampling times

For every retained winner trajectory, spatial diagnostics will be evaluated at least at tick 0,
the first quarter, the recorded maximum finite-domain scale, the third quarter, and the final tick.
Where the maximum duplicates one of those times, the nearest distinct sampled time is added. The
discovery run additionally uses every retained 2,048-phase checkpoint. No time may be selected by
visual appeal.

### Dense, sparse, and interface definitions

Let `rho_R` be toroidal box-averaged occupancy at widths `R = 16, 32, 64` cells. Dense/sparse
conditional summaries will be repeated for lower/upper empirical quantile pairs 10/90, 20/80,
and 30/70. Threshold values and realized region fractions, including ties, must be reported.

For interface measurements, a binary high-density phase is defined separately by:

1. the midpoint of the 20th and 80th percentiles of `rho_R`;
2. Otsu's threshold when it is nondegenerate; and
3. the global occupied fraction as an absolute reference threshold.

The toroidal morphological boundary is the set of phase-mask sites adjacent to the opposite class.
An interface band extends a declared distance on each side; interiors exclude that band. Results
that change sign or qualitative conclusion across `R` or threshold schemes will be labelled
threshold-sensitive.

### Frozen composition diagnostics

For dense interiors, sparse interiors, and interfaces, compute:

- counts and fractions of all four exact states;
- positive, negative, bound/neutral, charged, and occupied fractions;
- local signed-charge distribution and absolute local charge;
- normalized local state entropy;
- one-phase reaction frequency;
- all 2×2 block-code/motif frequencies;
- central-state-conditioned 3×3 neighbourhood frequencies;
- interface thickness, length, and roughness;
- signed-distance curvature only where stable under spatial resolution;
- Eulerian state-count exchange across a fixed pre-update interface mask.

The last quantity is not particle tracking. If a microscopic state identity cannot be followed
through a reactive block, no Lagrangian “particle flux” will be claimed.

### Frozen macrodynamic analysis

Retain separately as functions of phase: finite-domain scale `L(t)`, structure-factor peak and
wavelength, occupancy, coarse-density variance, interface length, domain count, largest-domain
fraction, all state fractions, reaction density, exact recurrence, and coarse recurrence.

For each winner seed report the peak time and scale, post-peak normalized decay, and final/peak
ratio. Compare constant-plateau, linear-return, exponential-relaxation, power-law-relaxation, and
damped-sinusoid descriptions using AICc on the post-peak series. A model is preferred only with
`ΔAICc >= 4`; otherwise the result is model-ambiguous. Fits with structured residuals or fewer
than eight post-peak samples are descriptive only.

Cross-correlations will be computed for occupancy versus `L`, reaction rate versus `L`, and
interface length versus occupancy, with both raw and linearly detrended series. Report the full
lag axis; do not select only the largest attractive lag.

Periodograms and autocorrelations search beyond the inherited lag-4/8 scaffold. A slower
macroperiod is called reproducible only if a non-edge spectral peak representing at least four
saved-sample intervals occurs within ±20% in at least three of four Stage 7B winner seeds and is
also visible in the corresponding autocorrelation. Otherwise it is at most a candidate oscillation.

## Double-conservation construction fixed before implementation

The chosen second invariant is nonempty-site count `M`. Every allowed 2×2 mapping must remain
inside a joint `(Q,M)` sector. There are 25 nonempty joint sectors, with sizes:

```text
(-4,4):1; (-3,3):4; (-3,4):4; (-2,2):6; (-2,3):12; (-2,4):10;
(-1,1):4; (-1,2):12; (-1,3):24; (-1,4):16;
(0,0):1; (0,1):4; (0,2):18; (0,3):28; (0,4):19;
and the sign-reflected positive-Q sectors of the same sizes.
```

The exact rule-space size is the product of the factorials of those sector sizes,
approximately `10^199.8192`, compared with `10^317.9615` for charge-only sectors. Thus rule freedom
is substantially reduced but remains enormous.

The closest sibling will retain the 90-degree rotation scaffold and, if an exhaustive constrained
construction permits it, exactly the winner's 24 reactive input locations. Its output assignment
will be a `(Q,M)`-sector permutation chosen to minimize table Hamming distance from the winner,
then lexicographically to break ties. It must remain reactive at all 24 locations. Independent
matched double-conservation controls will use the same scaffold, 12 local image swaps, 24 reactive
inputs, and independently frozen generation seeds.

Before any new trajectory runs, implementation must provide:

- exhaustive 256-input checks of permutation, `Q`, and `M` preservation;
- an independently coded exact verifier;
- forward/inverse recovery tests;
- joint-sector sizes and rule-freedom calculation;
- exact sibling and control tables, hashes, seeds, and settings in a committed
  `DOUBLE_CONSERVATION_MANIFEST.json`.

No global repair or compensating distant update is permitted.

## Frozen double-conservation screen

The first screen will use 1,024×1,024 worlds, 32,768 alternating phases, sampling every 256 phases,
and the eight primary seeds in `FROZEN_PLAN.json`. Four additional seeds are frozen as a reserved
extension and may be run only as a complete four-seed block, never selected by outcome.

Conditions are:

1. original winner;
2. pure 90-degree rotation;
3. all eight original matched controls;
4. closest double-conservation sibling;
5. eight matched double-conservation controls.

Initial states are paired exactly across every condition. They begin with `Q=0` and identical `M`.
The original charge-only conditions may change `M`; joint-conserving conditions may not.

A trajectory counts as strong domain organization when exact required invariants pass, peak finite-
domain scale is at least 8× initial, and final finite-domain scale is at least 2× initial. The joint-
conserving result is classified as:

- **disappears:** at most 2/8 sibling seeds meet the criterion and median peak ratio is below 2;
- **weakens but survives:** at least 3/8 meet it, but the comparable criterion below fails;
- **survives comparably:** at least 6/8 meet it and both median peak and final scale ratios are at
  least half the paired original-winner medians;
- **mixed:** none of the above;
- **potentially richer:** may be discussed only if at least 6/8 pass and at least two predeclared
  morphology coordinates exceed twice the original-winner median without collapse/noise flags.

“Potentially richer” is an interpretation flag, not a life or organism claim.

## Frozen same-family prevalence map

`FROZEN_PLAN.json` fixes 256 unique same-family rules and four untouched seeds. Every rule is the
same 90-degree scaffold plus 12 disjoint equal-charge image swaps, has exactly 24 reactive inputs,
and has an independent SHA-256 digest. The screen uses 1,024² worlds for 32,768 phases, sampling
every 512 phases. No optimization, evolution, replacement, or visual selection is allowed.

The winner, rotation, and the original eight matched controls will be rerun on the same four seeds
as anchors. Behavioural coordinates remain separate: peak/final domain scale, peak time,
occupancy change, dispersion, reaction density, recurrence, state composition, and
collapse/stasis/noise indicators.

For a descriptive prevalence count, a rule is “winner-like domain forming” if at least three of
four seeds have peak scale ratio at least 8, final scale ratio at least 2, and zero charge
violations. The observed fraction and Wilson 95% interval will be reported. Labels are:

- **common:** at least 20% of sampled rules;
- **moderately unusual:** at least 5% but below 20%;
- **rare tail:** below 5%;
- **distinct cluster:** used only if unsupervised behavioural clustering places the frozen winner
  in a stable cluster separated across multiple standardizations/cluster counts.

Clustering is descriptive. No universal scalar fitness score will be invented.

## Reversible macro-arrow demonstration

For the frozen discovery trajectory, plot synchronized forward and reverse microscopic phase axes
for occupancy, finite-domain scale, interface density, declared entropy measures, and all state
fractions. The reverse state at every retained boundary must match exact stored hashes. The figure
will be described as microscopic reversibility with history-dependent coarse observables, not
automatically as a thermodynamic arrow.

Candidate explanations to discuss separately are the localized initial ensemble, coarse-graining,
occupancy freedom, observational projection, and combinations of these.

## Optional perturbation study

Only after required studies are complete, apply fixed local perturbations at phases 8,192, 21,504,
and 49,152 on the frozen winner using the four perturbation seeds in `FROZEN_PLAN.json`. Compare a
Q-preserving balanced creation/removal intervention with a Q-and-M-preserving local state
permutation of matched spatial extent. Report recovery, regrowth, dissolution, displacement, or
ordinary relaxation; do not call it self-repair without evidence of maintained higher-level
organization.

## Integrity, resource, and reporting rules

- Preserve pause/resume and checksummed checkpoints.
- Use the existing resource governor and yield to foreground CPU/GPU demand.
- Never silently replace a failed trajectory, seed, or rule.
- Keep full states and other large raw artifacts outside Git; commit compact CSV/JSON, figures,
  manifests, tests, and reports.
- A diagnostic bug that changes earlier scientific semantics must be documented before repair; the
  historical result remains preserved.
- All new conservation and diagnostic logic requires unit/integration tests.
- Push only to the existing private repository on this branch; never expose credentials.

The final report will explicitly answer all 15 questions in the follow-up brief and choose a v0.2
decision. If the choice is to modify v0.2, recommendations will be written without implementing or
merging v0.2.
