# Erratum: distinction between Q and M

Status: additive v0.3.2 correction; frozen v0.3.1 artifacts remain unmodified.

## Correct definitions

`Q = N_positive - N_negative`. The frozen winner preserves `Q = 0` exactly.

`M = N_positive + N_negative + N_bound`. The frozen winner does **not** conserve `M`; material occupancy changes substantially. The separate exact-Q+M sibling conserves both coordinates, but its fixed `M` is the positive, nonzero initial occupancy. The phrase “Q = M = 0” is correct only for the empty state.

This is a wording correction. It does not change any stored state, trajectory, hash, measurement, or statistical result.

## Frozen public-facing statements affected

1. `experiments/reversible_null_phase_v031_nucleation_20260827/NUCLEATION_REPORT.md`, line 156: “In an exactly reversible lattice constrained to exact Q = M = 0 ...”

The same wording also occurred in campaign working records that are not part of this release.

## Corrected wording

Where that statement says “exact Q = M = 0,” read:

> In an exactly reversible lattice constrained to exact Q = 0, with occupancy M free to change, ...

A clearly versioned corrected derivative record is provided under `corrected_v031/`:

- `NUCLEATION_REPORT_QM_CORRECTED_V032.md` corrects the report claim.

The original statement and its existing artifacts remain byte-for-byte historical records.
