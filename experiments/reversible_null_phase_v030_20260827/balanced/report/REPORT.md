# Reversible Null Phase Study Report

Generated 2026-08-27T10:46:30.765656+00:00.

## Executive summary

The 1024² onset sweep did not bracket a 50% midpoint: its lowest sampled density (0.2675) already passed 7/8. The fitted p50=0.26460 is an extrapolation, not a measured threshold.

This report is a purpose-built follow-up to the frozen v0.1.4 discovery. It tests whether the apparent ~27.5% onset is a finite-time nucleation boundary, a linear spectral instability, a seeded/bistable regime, a topology change on the torus, or a reaction-pair-specific effect. It does not perform a new rule search.

## Completed trajectories

- Total: **684**
- Exact-charge violations recorded in compact results: **0**

## Spectral precursor

- Winner median maximum early shell growth rose from 1.520e-05 per phase at density 0.08 to 1.809e-04 at 0.16.
- The rotation control remained at or below 7.895e-19 in absolute median growth.
- None of these 8,192-phase precursor trajectories passed the strong-domain criterion.

This is evidence of reaction-specific low-k amplification well below the later strong-domain window, not by itself proof of a persistent phase or a sharp linear-instability threshold. See `figures/spectral_precursor_growth.png`.

## Strong-domain onset and first passage

- 512²: lowest sampled density 0.2600 passed 8/12; p50=0.25911, sample range 0.2600–0.2900, midpoint bracketed=false.
- 1024²: lowest sampled density 0.2675 passed 7/8; p50=0.26460, sample range 0.2675–0.2825, midpoint bracketed=false.

Both fitted midpoints lie below their sampled grids, so their apparent finite-size shift and width must not be read as converged thresholds. See `figures/onset_probability.png` and `figures/first_passage.png`.

## Horizon and topology

- At 1024² and density 0.2600, all 6/6 long-horizon trajectories passed by 131,072 phases; median first passage was 63,488.
- At that density, final empty- and occupied-wrap fractions were 0.00/0.00 in x. At density 0.2800, final occupied wrapping reached 1.00 in x and 1.00 in y.

The long-horizon passes at the lowest tested density support a waiting-time/nucleation component, while the density-dependent wrapping change shows that the late morphology also changes topology. These are compatible mechanisms, not mutually exclusive labels.

## Inventory-matched seeded/homogeneous comparison

- The finite-empty strong-growth classifier recorded 0/96 passes across all slab and droplet trajectories.
- Across densities 0.20–0.28, droplet median final/initial largest occupied-cluster retention ranged from 1.207 to 1.213.
- Across densities 0.20–0.28, slab median final/initial largest occupied-cluster retention ranged from 1.218 to 1.229.
- Homogeneous controls at densities 0.20 and 0.24 recorded 0/16 strong-growth passes.

The strong-growth classifier is normalized to the initial finite-empty scale and therefore is not a seed-survival statistic. The direct occupied-cluster measurement shows that both seeded shapes persist and grow where homogeneous runs do not cross the preregistered strong-domain criterion. This supports operational basin dependence, not an equilibrium free-energy claim or proof of first-order bistability. See `figures/seeded_cluster_retention.png` and `figures/hysteresis_seed_dependence.png`.

## Reaction-pair ablation

- Winner: 10/16 strong-domain passes.
- Exact Q+M sibling: 0/16; rotation scaffold: 0/16.
- Single-pair ablations with zero passes: ablate_00, ablate_02, ablate_03, ablate_04, ablate_07, ablate_10.

The effect is rule-specific and distributed: the occupancy-changing winner outperforms both exact controls, while removal of several distinct reaction pairs abolishes the regime in this protocol. See `figures/reaction_pair_ablation.png`.

## Files

- `TRAJECTORIES.csv`: one row per trajectory.
- `SUMMARY.csv`: grouped pass fractions and medians.
- `FIT.json`: descriptive logistic fits.
- `dashboard.html`: local visual browser for exact initial/peak/final states.

## Interpretation boundary

The study can establish formal dynamical properties of one finite reversible exactly-Q-conserving automaton. It cannot establish primitive occurrence, cosmology, consciousness, life, or universality across rule families. Bold interpretations may be discussed, but the measurements and exact inferences remain separately identified.
