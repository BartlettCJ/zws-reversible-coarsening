# ZWSLab v0.1.4 Domain-Mechanism Study

## Executive conclusion

The v0.1 phenomenon is real, reproducible pattern formation, but it is not sustained
coarsening and it is not evidence of life.  The strict confirmation classification remains
**D** because the preregistered late-coarsening gate failed.  What reproduced in all four
4096-square winner runs was a very large transient finite-domain scale (median peak
**98.4459x** initial), followed by partial relaxation to a reproducible but still macroscopic
finite-time scale (median final **16.1118x** initial).

The separating phases are not positive and negative charge domains.  They are a
high-entropy, excitation-rich mixture of positive, negative, and bound-neutral states versus
near-vacuum.  Exact signed charge `Q` remains zero because positive and negative excitations
are created in equal global numbers, while the independent occupancy `M` is free to change.
The closest exact-`Q` plus exact-`M` sibling remained reactive but lost the large domains.
Thus reversible, spatially selective excitation creation/destruction is a major causal engine;
occupancy freedom is necessary in this frozen comparison but is not sufficient by itself.

The same-family map found winner-like behaviour in **19/256 rules (7.4219%; Wilson 95%
interval 4.8026%-11.3001%)**, formally **moderately unusual**, and the frozen clustering
test placed it in a distinct behavioural cluster.  Exact reverse evolution retraced all 33
retained discovery boundaries and returned byte-for-byte to the initial state, so the striking
coarse history is compatible with exact microscopic reversibility.

## Evidential boundary and methods

This report was authored only after the frozen report-evidence assembler succeeded.  Its
machine-readable source is `artifacts/report/REPORT_EVIDENCE.json` (SHA-256
`3a5d02341d40e85398daffec0f5795ee28163e1c5887f43df5dbc9c3e747eabb`), which records
16 source-integrity entries and maps all 15 required questions to frozen evidence groups.
New simulation seeds, rule hashes, success criteria, durations, and reductions were frozen
before outcomes were evaluated.  Large raw trajectories remain outside Git; compact tables,
figures, manifests, and hashes are versioned.

The analysis covers 112/112 retained original trajectories, including 12 winner trajectories;
93/93 reconstructed spatial milestones; 152/152 frozen double-conservation trajectories;
1,064/1,064 prevalence trajectories representing 256 sampled rules plus frozen references;
and 96/96 perturbation branches with 512 paired evaluation rows.  All required exact-invariant
and input-integrity audits passed.

The report uses three labels throughout:

- **Measurement** means a direct value or comparison in the checksummed evidence.
- **Formal inference** means a consequence of an exact invariant/proof or a frozen decision rule.
- **Interpretation** means a scientifically motivated explanation that is not uniquely forced by
  the measurements.  **Speculation** is more tentative still.

## State semantics

| Code | Local state | Signed contribution to `Q` | Contribution to occupancy `M` |
|---:|---|---:|---:|
| 0 | empty | 0 | 0 |
| 1 | positive excitation | +1 | 1 |
| 2 | negative excitation | -1 | 1 |
| 3 | bound/neutral excitation | 0 | 1 |

Therefore `Q = N_positive - N_negative`, whereas
`M = N_positive + N_negative + N_bound`.  The bound state makes the independence especially
plain, although even balanced positive/negative creation can change `M` without changing `Q`.

## Principal measurements

In the four 4096-square Stage 7B winners, median occupied fraction rose from **0.0647156**
initially to **0.0964405** at each trajectory's domain-scale peak and **0.115703** finally.
Median charged fraction rose from **0.0485367** to **0.0777749** finally and median bound-state
fraction from **0.0161789** to **0.0379276**.  The four peaks occurred at phases 19,456,
19,456, 21,504, and 30,720; their peak ratios ranged from 72.66x to 136.11x, while final ratios
converged to 15.77x-16.33x.

Under the primary spatial definition (32-cell density window and 20/80 density quantiles),
Stage 7B median dense-interior occupancy was **0.320055** initially, **0.477776** at peak, and
**0.572989** finally.  At the final milestone its positive, negative, and bound-neutral fractions
were respectively **0.192542**, **0.192535**, and **0.187912**, with normalized exact-state
entropy **0.946343**.  Matching sparse interiors had zero median occupancy at all three roles.
All **72/72** estimable stage/role/field aggregates preserved the sign of the dense-minus-sparse
contrast across box widths 16/32/64 and density quantiles 10/90, 20/80, and 30/70.

Dense 16-by-16 blocks were globally balanced but locally charged: Stage 7B median mean absolute
block charge rose from 6.23 initially to 6.97 finally, and charge standard deviation from 7.82 to
8.85.  At peak, median outgoing reaction-cell fraction was **0.06333** in dense interiors,
**0.00268** at interfaces, and zero in sparse interiors.  In one phase under fixed pre-update
masks, median dense counts changed by -10,317 empty, +1,273 positive, +1,278 negative, and
+7,946.5 bound cells.  These are Eulerian count changes, not tracked-particle fluxes.

Post-peak model selection did not identify a common relaxation law: exponential was numerically
best in two seeds, power law in one, and damped sinusoid in one, with decisive separation only for
the last.  No slow macroperiod passed the frozen three-of-four-seed agreement test.  Raw median
zero-lag correlations were 0.2919 for occupancy versus domain scale and 0.2543 for reaction
density versus scale; after linear detrending they were 0.5601 and 0.6688.  These autocorrelated
associations are descriptive, not causal estimates.

The 1024-square exact-second-invariant comparison supplied the intervention: the original winner
passed the strong-domain rule in **8/8** runs (median peak 26.2277x; final 15.0221x), while the
closest exact-`Q`+`M` sibling passed **0/8** (peak 1.59449x; final 1.02746x).  None of eight
exact-`Q`+`M` matched controls or eight original-`Q` matched controls passed.  The sibling was
not inert: median mean reaction density was 0.01160 versus the winner's 0.01234.

The optional perturbation screen passed exact-`Q` application and dynamical-invariant audits for
all 96 branches.  All 48 state-swap applications conserved `M` and exact state counts; all 48
balanced-creation applications conserved `Q` and changed `M` by exactly +256.  Coordinate
deviations often persisted or grew over the measured horizon even though intensive absolute
differences remained small.  The frozen analysis deliberately defines no scalar resilience score
and no repair or self-repair classification.

## Answers to the 15 frozen questions

### 1. What exact local states make up the dense domains?

**Measurement.** Dense interiors contain all three occupied states, not one dominant species.
At the Stage 7B final milestone the primary-mask median fractions were 0.192542 positive,
0.192535 negative, and 0.187912 bound-neutral, giving occupancy 0.572989 and normalized entropy
0.946343.  Positive and negative fractions remained equal to about 3e-5 at displayed precision.

**Formal inference.** Dense domains are globally near-neutral, high-diversity excitation mixtures;
they are neither single-sign domains nor bound-state-only domains.

**Interpretation.** They are best described as regions that support a mixed excitation population,
with local charge fluctuations embedded inside global exact balance.

### 2. What exact local states make up the sparse domains?

**Measurement.** Under the primary mask the median sparse-interior occupancy was zero at initial,
peak, and final roles.  Alternative thresholds retained a positive dense-minus-sparse contrast for
every estimable occupied-state and entropy field.

**Formal inference.** The sparse phase is effectively state 0, the empty state, rather than a
different charged or bound phase.

**Interpretation.** The visually smooth “golf-course” areas are near-vacuum carved beside the
excitation-rich mixture.

### 3. What quantity/order parameter is actually phase-separating?

**Measurement.** Occupied, positive, negative, bound-neutral, charged, and exact-state entropy
fractions were all higher in dense than sparse regions under all 72 robust aggregate comparisons;
signed charge per site remained near zero.

**Formal inference.** Excitation occupancy/composition, `M`, is the primary separating order
parameter.  Signed charge `Q` is not.

**Interpretation.** This is reversible organisation between excitation-rich matter and near-vacuum,
not separation of positive matter from negative matter.

### 4. Why does occupied fraction increase while Q remains exactly zero?

**Measurement.** Positive and negative populations rise together and bound-neutral population also
rises.  Every audited update preserves `Q` exactly.

**Formal inference.** Equal creation of +1 and -1 contributions changes `M` while cancelling in
`Q`; creation of state 3 changes `M` while contributing zero to `Q`.  The invariant constrains
`N_positive-N_negative`, not the total occupied count.

**Interpretation.** The winner uses this independent occupancy direction as a reversible reservoir
for excitation creation and destruction.

### 5. What drives the enormous ~98x transient peak in domain scale?

**Measurement.** Growth coincides with increasing occupancy, reactions concentrated in dense
interiors, and strongly reduced interface density relative to occupancy.  Removing occupancy
freedom in the closest reactive sibling reduces the median peak from 26.2277x to 1.59449x at the
screening scale.

**Formal inference.** The frozen intervention establishes reversible occupancy-changing reactions
as a major causal engine in this architecture.  Reactions alone are insufficient because the
reactive exact-`Q`+`M` sibling fails; occupancy freedom alone is insufficient because all eight
`Q`-only matched controls fail.

**Interpretation.** The supported mechanism is a particular coupling of spatially selective,
balanced excitation creation/destruction to the rotational transport scaffold, which enlarges and
connects near-vacuum regions while enriching occupied interiors.

### 6. Why does the scale subsequently relax?

**Measurement.** Every Stage 7B winner falls far below its peak, with median final/peak ratio
0.164807.  No shared exponential, power-law, or damped-oscillation form is selected, and final
scales converge much more tightly than peaks.

**Formal inference.** The data establish partial relaxation but do not identify one universal
decay law or microscopic cause.

**Interpretation.** A plausible explanation is overshoot: reactions and transport transiently
join or eliminate many finite empty domains, after which continued reversible mixing fragments or
redistributes them toward a reproducible finite-time regime.  This explanation is not uniquely
proved by the retained observables.

### 7. Is there evidence of a stable characteristic macro-scale?

**Measurement.** At phase 65,536 the four Stage 7B final ratios lie in the narrow 15.77x-16.33x
band despite very different peaks.

**Formal inference.** This supports a reproducible late characteristic scale over the observed
window, but not an asymptotic plateau.  The sustained-late-coarsening gate failed.

**Interpretation.** “Transient growth followed by partial relaxation to a finite-time macroscopic
band” is warranted; “permanent stable scale” is not yet warranted.

### 8. Is there evidence of a slower macrocycle or recurrence?

**Measurement.** No eligible period passed the preregistered agreement test across at least three
of four seeds.  Only one seed decisively preferred a damped sinusoid; lag-4/8 recurrence is largely
inherited from the rotational scaffold.

**Formal inference.** There is no reproducible slower macrocycle in the measured 65,536-phase
window.

**Interpretation.** Longer cycles remain logically possible in a finite reversible system, but the
present data give no positive reason to claim one.

### 9. Does domain organisation survive a second exact local conservation law?

**Measurement.** The exact-`Q`+`M` sibling passed 0/8 strong-domain trials with median final ratio
1.02746x, versus 8/8 and 15.0221x for the original winner.  All 152 trajectories passed invariant
audits, and nine `Q`+`M` rules passed an independent exhaustive verifier.

**Formal inference.** By the frozen rule the classification is **disappears**.  The conclusion is
specific to the closest frozen sibling family, which differs from the winner at 45/256 table
entries because all 24 winner reactive mappings change occupancy; it is not a theorem about every
possible reversible `Q`+`M` automaton.

**Interpretation.** Non-conserved occupancy is a crucial engine of this particular morphology.

### 10. How common is winner-like behaviour within the matched rule family?

**Measurement.** Nineteen of 256 independently frozen same-family rules were winner-like:
7.4219%, Wilson 95% interval 4.8026%-11.3001%.  The frozen cluster-stability gate passed for
`k = 2, 3, 6, 7, 8`.

**Formal inference.** The preregistered label is **moderately unusual**, and a distinct behavioural
cluster is established by the frozen stability rule.

**Interpretation.** The winner is neither a unique miracle nor generic background behaviour.  It
is a member of a minority dynamical phase or family-level mechanism cluster.

### 11. What did coarse-density dispersion get wrong as a proxy?

**Measurement.** Non-passing control `matched_00` had final block-64 dispersion 2504.18, above the
winner's 1588.89.  The non-passing exact-`Q`+`M` sibling retained dispersion 941.33 and a
1024-cell structure-factor wavelength while its finite empty-domain scale stayed near 1x.

**Formal inference.** Dispersion and low-wave-number power can measure the broad localized support,
occupancy amplitude, or envelope without measuring connected finite-domain size.

**Interpretation.** The proxy mistook “where the initially localized material is” and “how uneven
the density is” for the topology and length scale of the carved finite domains.

### 12. Which observations are genuinely new relative to ordinary reversible cellular-automaton behaviour?

**Measurement.** Exact reversibility, conservation, recurrence from a scaffold, and visual pattern
formation are not individually novel.  The distinctive empirical package is: reproducible
72.66x-136.11x transient finite-domain amplification at 4096 square; mixed excitation-rich versus
near-vacuum composition; exact byte-for-byte reverse retracing at 33 boundaries; disappearance
under a locally exact second invariant; and 7.42% prevalence in the frozen family map.

**Formal inference.** The study provides a controlled causal account inside this architecture, not
a claim of a new universal CA principle.

**Interpretation.** The most interesting contribution is the clean demonstration that a conserved
null quantity can coexist with a large independent, reversible material reorganisation, and that
removing the non-conserved material coordinate suppresses it.

### 13. Does anything observed justify investigating higher-order identity, self-maintenance, heredity or replication?

**Measurement.** No persistent individuated object, lineage, templated copying, heritable variation,
adaptive response, or maintained return to an organism-level state was measured.  Perturbations
produced ordinary coordinate-wise propagation, divergence, or relaxation; the formal repair or
self-repair classification is `not_defined`.

**Formal inference.** Nothing observed supports identity, self-maintenance, heredity, replication,
adaptation, agency, organismality, or life.

**Interpretation.** These topics could be tested in a future system with explicit causal ecology
and entity-level observables, but morphology alone is not a sufficient reason to promote the v0.1
patterns into those categories.

### 14. What is the strongest scientific interpretation supported by the data?

**Measurement.** A localized low-occupancy ensemble reproducibly evolves into large transient
finite domains, higher mixed excitation occupancy, and a narrower late scale while preserving exact
`Q` and exact microscopic invertibility; imposing exact `M` suppresses the domain regime.

**Formal inference.** ZWSLab v0.1 is an exactly reversible, `Q`-conserving reactive-rotation system
in which the independent non-conserved occupancy coordinate is a necessary causal ingredient of
the frozen winner mechanism, though not sufficient across rules.

**Interpretation.** The strongest description is **transient reversible phase organisation between
a mixed, globally charge-balanced excitation-rich phase and near-vacuum, driven by a rule-specific
coupling of balanced excitation reactions to transport**.

### 15. What speculative implications, including possible relevance to Zero Without Stasis, are worth considering?

**Measurement.** Exact microscopic null balance does not prevent large changes in occupancy,
entropy, interfaces, domain scale, or spatial composition, and those changes can retrace exactly.

**Formal inference.** Conserving one signed aggregate leaves orthogonal degrees of freedom in which
substantial organisation can occur.  This is a mathematical statement about this state space, not
a metaphysical conclusion.

**Interpretation/speculation.** The result offers a concrete version of “zero without stasis”:
zero net charge is compatible with vigorous balanced creation, structured separation, and history-
dependent coarse appearance.  More speculatively, constraints may shape rather than eliminate the
space of possible organisation; apparent macroscopic arrows may arise from special ensembles and
projection even on exactly reversible orbits.  None of this implies cosmology, consciousness, life,
or universal emergence without further models and discriminating experiments.

## Reversibility and perturbation interpretation

The discovery trajectory's forward and reverse series contain identical coarse observables in
opposite order, and every retained reverse state matches its forward SHA-256 digest.  The apparent
macro-arrow is therefore not evidence for microscopic entropy production.  The supported sources
are the specially localized low-occupancy initial ensemble, coarse-graining/observational
projection, and the occupancy freedom that lets the orbit explore very different compositions.
Whether the reverse history is extraordinary relative to a well-defined typical-state ensemble was
not estimated and should not be asserted.

The perturbation results do not change that boundary.  Exact-`Q` local interventions can produce
long-lived coordinate differences without demonstrating a maintained higher-level entity.  Calling
ordinary relaxation “repair,” or assigning a scalar resilience score after the fact, would exceed
the frozen design.

## Limitations

- The strict confirmation result remains D; the study explains a reproduced transient morphology,
  not sustained late coarsening.
- The late 16x band is finite-time evidence, not proof of an attractor in an invertible system.
- Spatial masks are coarse definitions.  Contrast signs are robust, but interface composition,
  thickness, and roughness vary with threshold and resolution; curvature is not promoted.
- One-phase regional count changes are not Lagrangian particle fluxes.
- Correlations among autocorrelated coarse series are descriptive.
- The exact-second-invariant intervention is close but cannot be entry-identical to occupancy-
  changing winner reactions.  Its causal scope is the frozen sibling architecture.
- The prevalence fraction estimates the sampled reactive-rotation family, not all reversible CAs.
- No typical-state ensemble was sampled for a thermodynamic-arrow claim.

## v0.2 decision and recommended next step

**Recommendation B:** keep v0.1 as a completed, reproducible baseline and move primary effort to
the **v0.2 Causal Nullity Ecology**.  Do not merge or install v0.2 as part of this study.

The mechanism study has answered its main question: the v0.1 morphology is a transient,
occupancy-enabled reversible phase organisation, moderately unusual within its rule family, with
no evidence of higher-order living properties.  Further v0.1 work would be useful only for narrower
questions such as much longer finite-time relaxation or alternate exact-`Q`+`M` architectures.  The
higher-value next experiment is to give v0.2 explicit entity, causal-interaction, maintenance, and
lineage observables, preregistered so that identity or heredity can be tested rather than inferred
from appearance.  Recommendation B requires no `V020_INTEGRATION_RECOMMENDATIONS.md`, and this
branch makes no v0.2 implementation change.

## Reproducibility map

- Frozen design: `PREREGISTRATION.md`, `FROZEN_PLAN.json`,
  `DOUBLE_CONSERVATION_MANIFEST.json`, `PREVALENCE_SCREEN_MANIFEST.json`, and
  `PERTURBATION_RESPONSE_MANIFEST.json`.
- Existing dynamics: `artifacts/existing_metrics/` and `figures/existing_metrics/`.
- Spatial reconstruction/reduction: `artifacts/spatial_reduction/` and
  `figures/spatial_reduction/`.
- Second-invariant design/proof/results: `DOUBLE_CONSERVATION_DESIGN.md`,
  `DOUBLE_CONSERVATION_INDEPENDENT_AUDIT.json`, `artifacts/double_conservation_screen/`, and
  `figures/double_conservation_screen/`.
- Same-family prevalence: `artifacts/prevalence_screen/` and `figures/prevalence_screen/`.
- Exact macro-arrow: `artifacts/discovery_macro_arrow/` and
  `figures/discovery_macro_arrow/`.
- Perturbation response: `artifacts/perturbation_response/` and
  `figures/perturbation_response/`.
- Checksummed final evidence: `artifacts/report/REPORT_EVIDENCE.json` and its `.sha256` sidecar.

All paths above are relative to `experiments/domain_mechanism_20260825/` unless stated otherwise.
