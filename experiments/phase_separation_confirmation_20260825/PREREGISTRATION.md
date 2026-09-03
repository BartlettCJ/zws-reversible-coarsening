# Preregistration — Frozen ZWSLab Phase-Separation Confirmation

**Preregistered:** 2026-08-25 JST  
**Discovery freeze commit:** `74b4243c9518ec6f95b906de1a374bc277a2a5d5`  
**Frozen discovery rule-table SHA-256:** `d1fe99b5735d8e9be032347de2727dc5c52a7848df200ca26f67f28c8c1bc41c`

This file is committed before generating any confirmatory trajectory or examining any new-seed/control outcome. The original `results/observe_4096` run and all observations made before this commit are discovery data only.

## Confirmatory hypothesis

> The frozen `d1fe99...` rule reproducibly generates reaction-driven macroscopic domain formation/coarsening from the specified localized charge-balanced initial-condition family, beyond effects produced by matched controls.

## Explicitly not tested

This experiment is **not** a test of:

- life or organisms;
- replication, heredity, or open-ended evolution;
- consciousness or intelligence;
- the truth of Finitary Occurrent Actualism (FOA);
- primitive occurrence;
- ETNE, cosmology, or a fundamental process measure.

No image, event bookmark, recurrence value, or domain statistic will be promoted to one of those claims.

## Frozen conditions

All conditions use the exact same simulator, alternating Margolus phases, four-state alphabet, exact integer charge definition, RNG implementation, localized seed generator, and initial fractions as v0.1.4. No confirmation outcome may be used to tune or replace the discovery winner.

1. **Winner:** exact frozen `d1fe99...` table.
2. **Pure rotation:** 90° spatial block rotation, with no state-multiset-changing reactions.
3. **Identity:** exact identity mapping.
4. **Matched reactive-rotation controls:** eight independently generated tables based on the same 90° rotation scaffold, using 12 disjoint equal-charge image swaps and exactly 24 reactive inputs out of 256. Generation parameters and rule hashes are frozen before simulation.

An ordinary random charge-conserving rule may be retained only as a secondary reference and cannot replace the matched controls.

## Confirmation seeds

Exactly eight untouched confirmation seeds will be generated deterministically from a separately recorded manifest-generation seed. The same eight initial-condition seeds will be paired across winner, pure rotation, identity, and every matched control. Seeds will not be regenerated, discarded, or replaced because an outcome is boring, pathological, failed, or inconvenient.

The machine-readable seed and rule manifest, including all hashes, will be committed before the first confirmation trajectory is run.

## Initial-condition family

- exactly charge-balanced `balanced_random` construction;
- localized central square occupying 45% of each world dimension;
- positive fraction `0.12` inside the localized square;
- negative fraction exactly equal to the positive fraction;
- bound/neutral fraction `0.08` inside the localized square;
- all remaining cells empty;
- even world dimensions and toroidal boundary conditions.

## Frozen measurements

Measurements remain separate; no new “life,” “complexity,” or “interestingness” score will be constructed.

1. exact total charge and violation count;
2. occupied fraction and occupancy growth rate;
3. all four state populations;
4. one-phase activity;
5. state-multiset-changing reaction density;
6. normalized four-state entropy;
7. horizontal/vertical boundary density;
8. coarse-density variance relative to analytically matched and seeded shuffled same-density references;
9. connected empty-domain count, areas, largest area, perimeter, and characteristic domain scale under toroidal connectivity;
10. radial occupancy structure factor, peak wave number, and characteristic wavelength `L(t)` where robustly estimable;
11. exact-state recurrence at lags 1, 2, 4, 8, 16, 32, 64, and 128;
12. coarse occupancy correlation at the same lags;
13. failed runs, exceptions, charge violations, and non-estimable values.

Diagnostic curves will be sampled every 512 phases, including tick 0 and the final tick. Lag metrics compare exact global states at the declared physical lag, not adjacent explorer clip indices. Shuffle-reference RNG seeds are fixed from the run manifest and are not outcome-dependent.

## Stage 7A — calibration

- World: 2,048×2,048
- Duration: 32,768 alternating phases
- Seeds: all eight preregistered seeds
- Conditions: winner, pure rotation, identity, and all eight matched controls
- Total planned trajectories: `8 × 11 = 88`
- Pairing: identical initial seed across all eleven conditions
- Output: one compact scalar/curve record and explicit completion/failure record per planned trajectory

Paired plots will report:

- occupancy versus tick;
- characteristic domain scale versus tick;
- coarse-density dispersion versus tick;
- structure-factor peak/wavelength versus tick;
- lag-4 and lag-8 recurrence;
- exact charge audit.

### Continuation gate fixed before outcomes

Proceed to Stage 7B only if:

1. the winner shows reproducible macroscopic domain/coarsening behaviour in at least 6 of 8 seeds; **and**
2. the effect is not obviously explained equally well by pure rotation.

For the gate, a winner seed counts as macroscopic domain/coarsening only when all of the following hold:

- final 64×64 coarse-density dispersion ratio is at least `8.0`;
- final characteristic empty-domain linear scale is at least twice its tick-0 value, when both are estimable;
- the fitted late-half characteristic length slope is positive;
- exact charge violation count remains zero throughout.

“Explained equally well by pure rotation” is operationalized as pure rotation meeting the same per-seed criterion in at least 6 of 8 paired seeds and having a median final 64×64 dispersion at least 80% of the winner median. The full paired curves and effect sizes remain reportable regardless of this binary gate.

The winner is not required to beat every matched control; the matched-rule distribution is itself a confirmatory result.

If this gate clearly fails, Stage 7B will not run. The negative result will be preserved and reported without threshold revision.

## Stage 7B — gated large-scale confirmation

Run only if Stage 7A passes the frozen gate.

- World: 4,096×4,096
- Duration: 65,536 alternating phases
- Seeds: first four preregistered seeds; extend to six only if time/resources permit
- Conditions: winner, pure rotation, and the first four matched controls in manifest order
- No control may be selected based on Stage 7A appearance or performance.

## Planned inference and reporting

The final report will distinguish discovery from confirmation and answer:

1. Does the winner reproducibly form large spatial domains?
2. Does pure rotation do the same thing?
3. Is the winner unusual relative to same-family rules with 24 reactive inputs?
4. Is the apparent 4/8-tick oscillator mostly inherited from the rotation scaffold?
5. Does characteristic domain length grow with time, consistent with coarsening/phase separation?
6. Is the effect largely explained by reversible occupancy creation under charge-only conservation?
7. Is there evidence for anything beyond phase separation? The default answer is no unless independent lineage/heredity evidence exists.

Outcome classification is frozen to one of:

- **A.** Winner-specific or unusually strong null-conserving domain formation.
- **B.** Generic property of the reactive-rotation family.
- **C.** Mostly inherited from pure rotational kinematics.
- **D.** Initial discovery does not reproduce reliably.
- **E.** Mixed/inconclusive.

No positive classification is required.

## Analysis discipline

- Discovery observations cannot alter confirmatory thresholds or rule generation.
- All planned trajectories remain in the matrix even if they fail.
- Missingness and crashes are reported explicitly.
- Rule, seed, configuration, and artifact hashes are retained.
- Automaton, rule, RNG, and conservation semantics remain frozen.
- Diagnostic repairs are additive and tested. A repair that changes scientific semantics stops the affected experiment and is documented rather than silently substituted.

