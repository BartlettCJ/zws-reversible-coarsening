# Preregistration — ZWSLab v0.1.4 Homogeneous Density-Threshold Map

**Frozen before generating or inspecting any density-threshold outcome.** This final planned
v0.1 mechanism experiment changes only the homogeneous initial occupancy. It does not change the
winner, automaton, diagnostic definitions, strong-domain rule, low-k rule, discovery tags, or
v0.2.

## Question

As homogeneous global occupancy increases, is there a reproducible finite-size onset region for
the frozen winner's spontaneous large excitation-rich / near-vacuum domains?

The endpoints already established by the parent study are 0/16 strong-domain and 0/16 low-k
passes near 6.46% occupancy, versus 16/16 and 16/16 at 32%. This experiment maps the interval
without searching for another rule or changing the success criteria.

## Frozen provenance

- Parent branch: `followup/homogeneous-initial-condition-20260826`.
- Parent commit: `0f6de8dd325a53b66f4c7e972e993d496918ccfb`.
- Study branch: `followup/homogeneous-density-threshold-20260826`.
- Winner file: `experiments/phase_separation_confirmation_20260825/rules/winner.json`.
- Winner table SHA-256 over 256 uint8 entries:
  `d1fe99b5735d8e9be032347de2727dc5c52a7848df200ca26f67f28c8c1bc41c`.
- Immutable tags `v0.1.4-windows-discovery-freeze` and `discovery-d1fe99b5735d`
  must remain at `74b4243c9518ec6f95b906de1a374bc277a2a5d5`.
- Broad-grid design namespace: `ZWSLab-density-threshold-broad-v1:2026082601`.
- A derived seed is the low 31 bits of the big-endian integer represented by the first eight
  SHA-256 bytes of `namespace + ":" + label`.

## Broad-grid configuration

- World: 1024 x 1024 torus.
- Duration: 32,768 microscopic Margolus phases.
- Saved sample and resumable checkpoint interval: 256 phases.
- Early snapshot: phase 4,096; per-seed metric peak and final snapshots are also retained.
- Broad seeds: 12 per density, blocked across every density.
- Total broad trajectories: 14 densities x 12 seeds = 168.
- Coarse blocks: 16, 32, and 64.
- Recurrence lags: 1, 4, 8, 16, 32, 64, and 128 phases; formal requested recurrence reporting
  uses lags 4/8/16/32/64/128.
- Device: one CUDA worker when available. The established resource governor, background process
  priority, exact input preflight, checksummed checkpoint, pause/resume, and stop controls are
  mandatory.

### Broad density targets

The target grid is frozen as:

`0.0646, 0.08, 0.10, 0.12, 0.14, 0.16, 0.18, 0.20, 0.22, 0.24, 0.26, 0.28, 0.30, 0.32`.

Actual realized occupancies are written into the frozen manifest before evolution and used as
the density coordinate in all analyses.

### Broad blocked seed identities

In seed-index order 00-11:

`2019049721, 1098257983, 1091929321, 522236404, 481960949, 805102133,
1866549897, 1150730812, 1073990928, 1048271580, 1253183600, 2006309794`.

For each seed index, one arrangement seed is derived with `arrangement:seedXX`, and one diagnostic
shuffle seed with `diagnostic:seedXX`. The same two derived values are reused at every density.

## Exact inventory and homogeneous construction

For a world with `N` sites and target occupancy `p`, define the integer composition unit

`u = floor(p*N/8 + 0.5)`.

This is deterministic round-half-up to the nearest eight occupied sites. Then set:

- positive count = `3*u`;
- negative count = `3*u`;
- bound-neutral count = `2*u`;
- total occupancy `M = 8*u`;
- empty count = `N - 8*u`.

Thus positive and negative counts are exactly equal, exact global `Q = 0`, and the occupied-state
composition is exactly 3/8 positive, 3/8 negative, and 2/8 bound-neutral at every density. The
realized occupancy is `8*u/N`; no count is adjusted after outcome inspection.

For each seed, construct a single deterministic random permutation of all `N` site indices. At a
given density the first `3*u` source ranks are positive, the next `3*u` negative, the next `2*u`
bound-neutral, and all remaining source ranks empty; the common permutation maps those ranks onto
the torus. Consequences:

- every state is spatially homogeneous in construction, with no mask or gradient;
- exact inventories and `Q=0` are guaranteed;
- the occupied support is nested as density rises within a seed;
- seed identity and site priority are paired across all densities;
- occupied labels can change at inventory boundaries as `u` changes, so the design is blocked,
  not a claim that individual particles persist across counterfactual densities.

Before any evolution, the generator must write every target/realized density, all three seeds,
exact state counts, `M`, `Q`, initial low-k diagnostics, coarse-density baselines, and SHA-256 of
the contiguous uint8 state. The audit must regenerate every state, verify every digest, and verify
nested occupied support for all adjacent densities within every seed. No atypical-looking input is
regenerated.

## Reused frozen measurements

At all 129 saved boundaries, retain the established definitions of:

- exact `Q`, all four state populations/fractions, occupancy, and occupancy change;
- finite empty-domain characteristic scale, finite component count, finite largest area,
  connected-domain log2 histogram, quantiles, perimeter, and largest total empty-domain fraction;
- peak and final finite-domain scale ratios;
- exact-state boundary density and occupancy interface density;
- reaction and activity density;
- coarse-density variance at blocks 16/32/64 with analytic and seeded-shuffle baselines;
- radial occupancy structure factor and characteristic wavelength;
- directional Fourier modes and radial low-k bands, including `k=1..4` mean power and spectral
  fraction;
- normalized occupancy spatial autocorrelation at lags 1/2/4/8/16/32/64/128/256;
- normalized exact-state entropy;
- exact-state and block-64 occupancy recurrence at 4/8/16/32/64/128.

At peak and final snapshots, use the existing width-32, 20/80-quantile spatial masks to measure
dense/interfacial/sparse state composition and entropy. “Dense-interior entropy” is reported only
when that trajectory passes the frozen strong-domain rule; quantile-selected subsets of non-domain
states are not promoted as genuine dense phases.

No composite fitness, interestingness, life, or resilience score is defined.

## Frozen trajectory classification

The strong-domain and low-k criteria are copied unchanged from the homogeneous-IC manifest.

A trajectory is `strong_domain = true` only if:

1. peak finite-domain scale ratio >= 4;
2. final finite-domain scale ratio >= 2;
3. peak largest finite empty-domain area >=
   `max(64, 16 * max(initial largest finite empty-domain area, 1))`;
4. exact `Q = 0` at every saved phase.

The initial scale denominator is the measured initial finite-domain scale, floored at one cell,
exactly as before.

A trajectory is `low_k_growth = true` only if radial `k=1..4` peak mean-power ratio >= 4,
final mean-power ratio >= 2, and peak spectral-fraction ratio >= 2, using only a `1e-30`
numerical floor for an exact floating zero.

Every density reports the independent pass count and fraction for both classifications. The
principal transition observable is the strong-domain pass fraction.

## Frozen finite-size response classification

Let `f(d)` be the strong-domain pass fraction at realized density `d`. A **25-to-75 bracket** is
an ordered pair `d_low < d_high` with `f(d_low) <= 0.25` and `f(d_high) >= 0.75`. The primary
bracket is the eligible pair with minimum width, breaking ties toward the lower pair.

A **severe reversal** exists if an earlier density exceeds a later density's pass fraction by at
least 0.50, or if a density at/above 0.75 is followed by a later density at/below 0.50.

Classification priority is:

1. **E — NON-MONOTONIC / MULTIPLE REGIMES** if a severe reversal exists.
2. **A — NO TRANSITION IN TESTED RANGE** if `max(f)-min(f) <= 0.25`, or all `f <= 0.25`.
3. If a primary 25-to-75 bracket exists:
   - **D — VERY SHARP FINITE-SIZE TRANSITION REGION** if width <= 0.03;
   - **C — SHARP FINITE-SIZE TRANSITION REGION** if width <= 0.06;
   - **B — GRADUAL CROSSOVER** if width > 0.06.
4. **F — MIXED / INCONCLUSIVE** otherwise.

These labels are descriptive finite-size dynamical classes. None is automatically a
thermodynamic phase-transition claim.

## Frozen broad-to-refinement rule

The full 14-density broad grid must complete and pass integrity checks before any scientific
outcome is read. The frozen broad analyzer then selects the refinement interval:

1. use the primary 25-to-75 bracket if one exists;
2. otherwise use the adjacent density pair with the largest increase in `f(d)`, breaking ties
   toward the lower interval, but only if that increase is at least 0.25;
3. if neither exists, no refinement run is justified and the absence is reported.

Before refinement, create and commit `STAGE2_ADDENDUM.md`, a new input manifest/audit, every new
density, 16 new blocked seeds per density, and every initial-state hash. No Stage-2 state evolves
before that commit.

- Use 0.005 spacing strictly inside a selected interval of width <= 0.03.
- Otherwise use 0.01 spacing strictly inside the interval.
- Exclude densities already present in the broad grid.
- Values are generated arithmetically from the lower endpoint and rounded to six decimal places;
  no density is selected from an individual seed outcome.

## Frozen onset-curve analysis

A logistic onset curve is fitted only when all are true:

- at least one strong-domain pass and one failure exist;
- no severe reversal exists;
- Spearman rank correlation between density and pass fraction is >= 0.8.

The response is each trajectory's binary strong-domain result. Density is centered at 0.20 and
scaled by 0.10. Fit a binomial logistic curve by maximum a posteriori optimization with independent
Normal(0, 10) weak regularization on intercept and slope and constrain the slope positive. This
weak regularization makes separated finite samples estimable and is reported explicitly; the fit
is descriptive, not microscopic theory.

Report densities at predicted probabilities 0.25, 0.50, and 0.75 and the 25-to-75 width.
Uncertainty uses 2,000 deterministic blocked nonparametric bootstrap replicates. Broad seed
identities are resampled as whole cross-density blocks; refinement seed identities are resampled
as whole cross-density blocks within their stage. Percentile 2.5/97.5 intervals are reported;
fewer than 1,800 successful refits invalidates bootstrap intervals but not raw pass fractions.
Bootstrap namespace: `ZWSLab-density-threshold-fit-v1:2026082601`.

If the monotonicity gate fails, no logistic curve is forced. Continuous medians versus density are
still reported for peak/final scale ratio, low-k amplification, final-minus-initial occupancy,
mean reaction density, dense-interior entropy when defined, interface density, and largest empty
domain fraction.

## Morphology and exploratory cluster geometry

Common-scale state-color grids are generated after final 1024 analysis at deterministic nearest
available densities: clearly below onset, nearest predicted/empirical 25% point, nearest 50%
point, nearest 75% point, clearly above onset, and 32%, with duplicates removed. Each grid includes
initial, early, per-seed metric peak, and final states.

The previously frozen simple-regular spectral signature remains: strongest conjugate mode-pair
upper-bound fraction >= 0.15 and strongest radial shell fraction >= 0.30. A repeatable simple
regular morphology requires that signature in at least 75% of trajectories at a density plus
matching visual geometry. Otherwise visual terms such as droplets, labyrinths, channels, or
amorphous domains remain explicitly descriptive.

Exploratory, snapshot-only cluster diagnostics apply the existing exact toroidal four-neighbour
component reducer to both occupied and empty masks at initial/peak/final roles. Report component
counts, largest occupied-cluster fraction of all occupied sites, largest occupied-cluster fraction
of the world, largest empty-cluster fraction, and cluster-size histograms. No toroidal winding
test is added, so “percolation” is not formally asserted from edge touching or cluster size alone.

## Optional small finite-size gate

A finite-size check is permitted only after the completed broad and refinement stages if:

- the final response is C or D;
- the monotonic logistic fit is valid;
- the fitted 25-to-75 width is <= 0.06.

If triggered, first commit `FINITE_SIZE_ADDENDUM.md` and a new exact input manifest/audit. Select
the nearest available/fitted densities to the 25%, 50%, and 75% points, deduplicated and frozen.
Run 8 blocked seeds per density at 512 square and 1024 square, and 4 blocked seeds per density at
2048 square, all for 32,768 phases. This is only a location/sharpening check, not finite-size
scaling. It may not run at 4096 square.

If the gate fails, no size check is run and question 12 is answered as not measured.

## Failure, inspection, and reporting policy

- Preserve every failed completion; never silently replace or cherry-pick it.
- The campaign may pause/resume from checksummed checkpoints but may not regenerate inputs.
- During a running stage, operational health may be inspected, but partial scientific outcomes
  are not read.
- Giant checkpoints, states, and trajectories stay under `results/` and outside Git.
- Compact CSV/JSON, code/tests, hashes, and figures are versioned.
- The final report separately labels broad-grid, refinement, optional-size, and exploratory
  evidence and explicitly answers all 15 requested questions using Measurement, Formal inference,
  Interpretation, and Speculation boundaries.

## Final decision

The final v0.1 recommendation is exactly one of:

- **A:** close v0.1 as a completed baseline and move to v0.2;
- **B:** one precisely specified v0.1 follow-up is scientifically necessary;
- **C:** the phenomenon warrants a separate dedicated study before v0.2.

Additional work is not recommended merely because more plots are possible. No v0.2 work begins
automatically.
