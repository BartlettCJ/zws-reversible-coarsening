# v0.3.1 Low-Density Nucleation and Metastability Preregistration

Frozen before any v0.3.1 trajectory outcome was generated or inspected. The study is an additive
follow-up to v0.3.0 on branch `followup/v0.3.1-low-density-nucleation-20260827`. It neither changes
the winner nor resumes rule search. The 1,932-task powered v0.3 profile and the Causal Nullity
Ecology are outside scope.

## Frozen rule, dynamics, and classifier

- Winner raw-`uint8` rule-table SHA-256:
  `d1fe99b5735d8e9be032347de2727dc5c52a7848df200ca26f67f28c8c1bc41c`.
- Periodic two-dimensional lattices use the existing four-state, exact-integer, reversible update.
- Every initial condition uses the existing exact inventory rule
  `u = floor(density*N/8 + 0.5)` and counts `(N-8u, 3u, 3u, 2u)`, hence exact `Q=0`.
- The operational v0.3 extension classifier is frozen unchanged: peak finite-empty scale ratio
  at least 4, current-prefix scale ratio at least 2, peak largest finite-empty fraction at least
  0.001, and conserved charge at all observations. The denominator is the initial finite-empty
  scale floored at one. This explicitly follows the shipped v0.3 implementation and its balanced
  `PLAN.json`; an older pre-extension document's absolute-area wording is not substituted here.
- `T_hit` is the earliest scheduled observation at which that classifier is true on the trajectory
  prefix ending at the observation. It is never inferred between observations.
- `T_persistent` is that earliest hit only if the classifier remains true at both of the next two
  scheduled observations. A later run of three true prefixes may establish a later persistent hit.
- No observed event by the applicable horizon is right-censored. The horizon is never imputed as
  an event time. A hit too close to the horizon for two follow-ups is separately marked as having
  insufficient persistence follow-up.

## Frozen task matrix

All tasks use the winner, a 2,048-phase observation interval, an 8,192-phase checkpoint interval,
and a primary horizon of 262,144 phases. The generated `FROZEN_PLAN.json` is authoritative and
records every task ID, inventory, realized density, and deterministic initial-state hash.

1. Primary homogeneous 1024x1024: target densities 0.16, 0.18, 0.20, 0.22, 0.24, 0.25, 0.26;
   seeds 5101 through 5112 (12 per density; 84 trajectories).
2. Area scaling homogeneous 512x512: densities 0.22, 0.24, 0.26; matched seeds 5101 through 5108
   (24 trajectories).
3. Seeded basin 1024x1024: densities 0.20, 0.22, 0.24, 0.26; droplet and slab arrangements with
   seeds 5101 through 5106 (48 trajectories). The matched homogeneous comparators are the primary
   trajectories with identical density and seed, so they are reused rather than rerun.

The total preregistered primary matrix is 156 new trajectories. Twelve primary seeds were selected
because the known v0.3 balanced runtime makes the matrix suitable for a managed overnight run while
improving censoring estimates over the minimum eight.

Fixed readouts are phases 32,768, 65,536, 131,072, and 262,144. The principal estimates are
Kaplan-Meier event-free survival, cumulative incidence for this single event definition, identifiable
median and quartile first-passage times, and descriptive interval hazards/exposure rates. Undefined
quantiles remain undefined. Hazard estimates are descriptive finite-ensemble quantities, not a
claim of a stationary Poisson process.

## Positive-control and Stage 2 gates

Density 0.26 at 1024x1024 is the positive control. Fewer than 2 of 12 primary hits by phase 131,072
triggers a stop-and-investigate audit of rule hash, classifier semantics, observation timing, and
initialization before scientific interpretation.

Stage 2 is considered only after every primary homogeneous trajectory passes integrity checks. It
is eligible if a tested density at or below 0.24 has at least two events and its immediately lower
tested density has event fraction below 0.25. The bracket is the lowest eligible event density and
its immediately lower tested density. Before looking at any extended outcome, `STAGE2_ADDENDUM.md`
must freeze and commit the selected task IDs. Only censored/surviving trajectories at the two bracket
densities are extended to phase 524,288 from their exact phase-262,144 final states/checkpoints; a
restart is forbidden. No optional new seed is automatic, and at most four may be added only with a
written justification in the addendum. If no event occurs at or below 0.24, no automatic extension
is made. Deadline censoring is not evidence of impossibility.

## Topology, local windows, and representative selection

At each observation, record first true times for occupied and empty x-wrap and y-wrap using the
existing exact periodic topology measurement. Every hit trajectory retains only a compressed local
event window spanning phases `T_hit-16384` through `T_hit+16384`, clipped to available observations,
plus hashes. It contains scheduled exact states and measurements, not the full history. The local
crop is a square of side `size/4`, centered at the periodic location maximizing absolute occupancy
change from the earliest available pre-hit state to the hit state; lexicographic coordinates break
ties. Local composition, domain/interface, low-k, activity/reaction, and topology readouts use that
frozen crop. This local analysis is descriptive and does not redefine the event.

For each eligible cohort, morphology representatives are selected mechanically as the earliest hit,
the hit nearest the cohort median (absolute difference, then lower time, then task ID), and the latest
hit (task ID breaks remaining ties). No image-quality selection is allowed.

## Frozen local event-shape labels

Pre-outcome correction (still before any v0.3.1 trajectory was run): the letters below follow the
mechanism labels in the task specification rather than using letters as topology-only categories.
Labels summarize measurements and are not new phases:

- A — sudden growth of one localized nucleus. At least 50% of changed occupancy sites in the
  earliest-pre-hit to hit comparison lie in the frozen `size/4` crop (which occupies 1/16 of the
  lattice), and the hit is not uniquely assigned to D.
- B — simultaneous coarsening throughout the world. The crop contains no more than 1/8 of changed
  occupancy sites and finite-empty scale grows in at least half of the pre-hit observation intervals.
- C — merger of several pre-existing low-k regions. Neither A nor B applies, the pre-hit low-k power
  is already at least twice its phase-zero value, and finite empty-component count falls by at least
  25% across the available pre-hit window.
- D — topological reconnection of an already textured state. An occupied or empty wrap first appears
  within one observation of `T_hit`, while the finite-empty scale two observations before the hit is
  already at least twice its phase-zero value.
- E — other/mixed/unresolved. None of A-D applies uniquely, required pre-hit observations are absent,
  or more than one rule applies. When rules overlap, D has priority only when its complete condition
  holds; all other overlaps are E.

The primary report must separate direct measurement, finite-sample statistical summary, physical
interpretation, and speculation. It must not claim a thermodynamic phase transition, free-energy
barrier, equilibrium metastability, or infinite-volume impossibility.

## Integrity and storage

The managed supervisor must preserve pause, checkpoint, worker unload, exact resume, rule/config/task
hashes, sidecars, and exact-Q audits. Full trajectories and checkpoints remain under ignored `results/`;
Git receives only compact code, manifests, event windows selected by the frozen rule when reasonably
sized, tables, figures, reports, and hashes. The frozen v0.3 tags and prior discovery artifacts are
immutable. Eshkol is not rerun unless rule application or update semantics change.
