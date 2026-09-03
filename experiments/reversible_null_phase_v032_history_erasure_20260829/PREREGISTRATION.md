# v0.3.2 state-matched history-erasure preregistration

Status: **frozen before outcome access**  
Branch: `followup/v0.3.2-history-erasure-paper-closure-20260829`  
Starting commit: `9aaca1ed0242b744f22cd32becc31bd6c69136d3`  
Frozen winner raw-`uint8` table SHA-256: `d1fe99b5735d8e9be032347de2727dc5c52a7848df200ca26f67f28c8c1bc41c`

## Question and interpretation boundary

The causal question is whether erasing spatial arrangement, while preserving the complete current inventory and update phase, changes the remaining time to the same frozen strong-domain macrostate. The manipulation is external; it is not asserted to be a natural transition of the automaton. The result is evidence about this formal model, not cosmology, life, consciousness, thermodynamic equilibrium, or universality across cellular automata.

The signed invariant and material coordinate are distinct:

- `Q = N_positive - N_negative`; the winner preserves `Q = 0` exactly.
- `M = N_positive + N_negative + N_bound`; the winner does not conserve `M`.
- Every paired intervention preserves the source state's full four-state histogram, hence both current `Q` and current `M`, exactly.

## Frozen sources

- Geometry: 1024 x 1024 periodic torus.
- Initial arrangement: homogeneous exact-inventory shuffle.
- Densities: 0.20 and 0.24.
- Seeds: 6201–6208 at each density; an exact seed-field audit found this range unused in related prior campaigns.
- Initial occupied composition: 3/8 positive, 3/8 negative, 2/8 bound-neutral, using the established nearest-unit rounding rule.
- Source horizon: 524,288 phases.
- Observation and topology interval: 2,048 phases.
- Checkpoint interval: 8,192 phases.
- Right-censor any non-hit at the horizon; do not replace or regenerate a censored source.

Each source records its exact initial hash and inventory; all observation-state hashes; `Q`, `M`, state fractions, low-k history, finite-domain scale, interface density, entropy hierarchy, and wrapping/topology events; and `T_hit`/`T_persistent` with censoring status.

`T_persistent` is the start of the first run of three consecutive strong-domain observations: the hit observation plus two followups.

If fewer than six sources at either density yield an eligible event, the intervention campaign stops as a failed positive control. Densities and seeds will not be changed.

## Frozen branch points

Branch A is the first sampled phase with `M/N >= 0.35`, provided the source remains below the frozen strong-domain criterion, has no persistent hit, has a valid state hash, and retains exact `Q=0`. The realized occupancy is not edited. A source that becomes strong-domain first is A-ineligible.

Branch B is `T_hit - 32,768`, provided `T_hit >= 49,152`, strong-domain is false at B, B lies on the frozen grid, and deterministic reconstruction or stored-state verification matches the recorded hash. B is future-indexed only as a mechanism probe; it is not a natural-incidence estimate. Its untouched residual `T_hit` is 32,768 phases by construction.

## Frozen interventions and task count

The untouched source suffix is the reference and is not duplicated as an avoidable scientific trajectory. Every eligible A and B state receives one primary global shuffle. Every eligible B state receives a 32 x 32 tile permutation and a 128 x 128 tile permutation. Seeds 6201 and 6202 at each density additionally receive a second B global shuffle and an even-translation B null control.

The maximum count is therefore:

- 16 source trajectories;
- 16 primary A global shuffles;
- 16 primary B global shuffles;
- 4 second B global shuffles;
- 16 B tile-32 permutations;
- 16 B tile-128 permutations;
- 4 B translation nulls;
- total maximum: **88 scientific trajectories**.

`INTERVENTION_MANIFEST.json` freezes each task identity, all intervention seeds, every tile permutation, every even offset, and global-permutation digests before outcomes. Ineligible conditional branches are not replaced.

## Exact intervention audit

Before evolution, reject any altered state unless dimensions, next Margolus partition parity, the four exact state counts, `Q`, `M`, and single-site entropy match the source exactly. Save source and altered hashes, seed, permutation digest, phase, and immediate finite-domain, low-k, interface, block-entropy, mutual-information, and angular-spectrum measurements.

Tile interventions apply a frozen even offset, move complete unmodified tiles, then undo the offset. They do not rotate, reflect, or internally shuffle tiles. Translation offsets are even. At each null-control audit phase, undo the translation and require a byte-identical match to the untouched source. Any mismatch stops interpretation.

## Branch evolution and calibration

Altered branches run for at most 262,144 elapsed phases, sampled every 2,048 and checkpointed every 8,192. They may end after `T_persistent + 16,384` only after the full capture window, final metrics, hashes, and explicit reason are recorded. Non-hits are censored, never assigned the horizon as an event time.

Exact states are retained at elapsed 0, 8,192, 16,384, 32,768, 65,536, 131,072, and 262,144/final, and at event-relative -16,384, -8,192, 0, +8,192, and +16,384 when available.

The primary branch criterion inherits the original source's frozen initial finite-domain denominator, baseline scale, and threshold. It is never recalibrated at intervention. Secondary absolute observables remain separate.

## Outcomes and frozen decision rules

Primary outcomes are residual `R_hit` and `R_persistent`, with Kaplan–Meier curves, RMST through 262,144, fixed-time fractions at 32,768/65,536/131,072/262,144, paired branch-minus-untouched delay, and deterministic source-cluster bootstrap uncertainty. Densities are reported separately before pooling. No exponential waiting-time model is assumed and medians are omitted when fewer than half transform.

The practical-equivalence margin is ±8,192 phases. Branch A and B are classified separately:

- **Inventory-equivalent:** the 90% interval for paired RMST delay lies wholly within the margin, no strong density contradiction exists, and at least 12/16 available primary branches support practical equivalence under the censoring-aware rule.
- **Spatial-history delay:** the lower 95% interval exceeds +8,192, both density point estimates are positive, and at least 12/16 available branches are delayed by more than one observation or censored beyond it.
- **Spatial-history inhibition:** the corresponding upper 95% interval is below -8,192 with consistent direction.
- **Mixed/inconclusive:** any other result.

The combined result is frozen as inventory-dominated throughout, late-emerging spatial memory, early accumulated spatial memory, history-inhibited transformation, scale-dependent/mixed, or inconclusive according to the two branch classifications and tile results. No outcome-dependent scientific Stage 2 is allowed.

## Representatives and stopping

For each density, figures and history-erasure videos use the eligible source nearest the within-density median untouched `T_hit`, tied by lowest seed. The hero image uses the exact-reversal Stage-7B discovery winner nearest the four-winner median phase of peak finite empty-domain scale, tied by source seed then task ID. Visual preference is not a selection criterion.

Maximum intended managed wall time is 18 hours, excluding user-activity pauses. Exact resume is required rather than protocol weakening. After the campaign, the paper is written regardless of mechanism classification.
