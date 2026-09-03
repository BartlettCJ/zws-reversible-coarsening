# Preregistration Amendment 002 - translation-invariant torus topology

Status: frozen before the corrected campaign rerun.

## Why this amendment is mandatory

The first sealed analysis of the completed v0.3.2 campaign exposed a failure of
the even-translation null at the **metric/event-clock** level. The altered and
reference lattices remained byte-identical after undoing the frozen even
translation at every audited phase, but one null branch was classified at
`R_hit = 22,528` while its untouched reference was classified at
`R_hit = 32,768`.

The discrepancy was traced to the former 3x3-tiled implementation of toroidal
component analysis. Its central-tile component labels depended on the chosen
coordinate origin. For the exact source state at phase 129,024 from source task
`a53add2a25fda6e24aea`, translating by the frozen offset `(454, 420)` changed:

- component count from 50,126 to 50,130;
- finite-component count from 50,125 to 50,129;
- largest finite empty fraction from 0.0009775161743164062 to
  0.0018444061279296875;
- characteristic finite empty scale from 3.4730135246880893 to
  4.203471310432729.

This violates the frozen null-control requirement. The pre-amendment mechanism
classification and publication outputs are therefore invalid for scientific
interpretation, even though their underlying trajectories, exact inventories,
Q/M audits, checkpoint hashes, and state-conjugacy audits remain preserved.

## Frozen correction

The topology implementation is replaced by an exact periodic union-find over
open-boundary components. Integer tile-offset potentials identify non-contractible
winding cycles without reference to the coordinate origin.

For the same failing state, the corrected implementation gives identical values
before and after translation:

- component count: 50,088;
- finite-component count: 50,087;
- wrapping in x and y: true and true;
- largest finite empty fraction: 0.0018444061279296875;
- characteristic finite empty scale: 4.251048608639074.

The branch runner now audits both byte-exact conjugacy and exact equality of all
translation-invariant primary topology fields at elapsed zero and every sampled
phase. The final analyzer additionally requires translation-null `R_hit` and
`R_persistent` to equal the untouched reference clocks exactly.

## What remains frozen

No scientific threshold, decision rule, equivalence margin, seed, density,
horizon, observation interval, checkpoint interval, intervention seed,
permutation algorithm, representative-selection rule, or plotting rule changes.
The winner rule remains unchanged.

The complete 16-source plus up-to-72-branch task campaign will be rerun from its
original frozen initial conditions in:

`<local>/results\reversible_null_phase_v032_history_erasure_20260829_topology_fix_001`

The corrected source results will determine Branch A/B eligibility and Branch B
phases under the original frozen definitions. Realized task identities may
therefore change, but the planned scientific design and maximum of 88 distinct
trajectories do not. This is a technical validity rerun, not a new exploratory
campaign or an added replicate set.

The original run directory remains untouched as a historical failure record.
The corrected run alone may supply the final report, figures, manuscript, and
mechanism classification.

