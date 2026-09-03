# Independent review of the v0.3.2 event classifier

Date: 2026-08-31 (JST)

Status: independent post-campaign verification from frozen and retained v0.3.2 data. No new simulation was run and no historical result was edited.

## Conclusion

The detector-failure diagnosis is confirmed. The state-matched intervention was implemented correctly, but the inherited finite-empty-domain strong-event classifier was not invariant to global shuffling. Global shuffling preserved the full four-state inventory, signed charge `Q`, occupancy `M`, single-site entropy, lattice size, update phase, and subsequent rule while removing more than 99% of measured long-wavelength organisation. Many shuffled states nevertheless satisfied the inherited classifier at elapsed phase zero or at the first few observations.

Because every altered branch was then terminated 16,384 phases after its frozen persistent-event time, the experiment did not measure the time required for genuine large-scale organisation to re-form. The v0.3.2 conclusion that spatial history inhibited transformation is therefore withdrawn. The direction of any causal effect of pre-existing spatial correlations on genuine macroscopic reorganisation remains unresolved.

This is a construct-validity failure of the endpoint under the intervention, not a software-run failure. The exact intervention audits, state preservation, branch execution, checkpoint/restart behaviour, and retained measurements remain valid.

## Authoritative inputs

All compact paths below are relative to `experiments/reversible_null_phase_v032_history_erasure_20260829/`.

| File | SHA-256 |
|---|---|
| `RESIDUAL_FIRST_PASSAGE.csv` | `7b4a6245bab17c2db5d9c7e6ea9280bc51b14fd58651039deef92df306527816` |
| `INTERVENTION_AUDIT.csv` | `ce0e9cbad3ec6891cd58de8422193a24238264658d150ebf757cd3fdff875ee1` |
| `MULTISCALE_MEMORY.csv` | `b6c79fd827da6ad67dc53ab3599f14c2e4b4ad2ad67b57cbd7c0c90e32d54051` |
| `INTERVENTION_MANIFEST.json` | `19f89585642a05c8ec9977d9641167cdf3fbf7c512f74ded3ede13bbfbaa1b17` |
| `FIGURE_MANIFEST.json` | `d8f3ddbffd20470c1ed50fa151e10de56dcab5ea6c790c46092e06f0f31f1f07` |
| `REPRESENTATIVE_STATE_INDEX.csv` | `51a21e26663415648755fa04dd2a9d33efce6510e81f3aa50918e399019101a5` |

The manifest contains 72 altered branches: 16 Branch-A primary global shuffles; 16 Branch-B primary global shuffles; four additional Branch-B global-shuffle realisations; 16 Branch-B tile-32 branches; 16 Branch-B tile-128 branches; and four even-translation nulls. Its frozen payload hash is `bb68512730a77912e14bb4e3d2f8ee6d52ff519ffeb58f2e4dd7a8a066874f64`. All 72 rows in `INTERVENTION_AUDIT.csv` are valid.

The retained raw branch directory used for the termination and exact-state cross-check was `<local>/results\reversible_null_phase_v032_history_erasure_20260829_topology_fix_001`. It is the corrected authoritative run; the earlier invalidated run was not used.

## Exact first-passage checks

### Branch B primary global shuffles

The 16 primary rows are `branch_point=B`, `intervention=global_shuffle`, `realization=1`.

- `R_hit == 0`: 9/16.
- `R_hit <= 4,096`: 14/16.
- `R_hit <= 6,144`: 15/16.
- Remaining row: `R_hit = 20,480`.
- Mean `R_hit`: 2,432 phases.

The supplied reviews expected 14/16 at or below 6,144. Direct enumeration of the frozen CSV gives 15/16; 14/16 is the count at or below 4,096. This one-row correction does not alter the diagnosis.

### Branch A primary global shuffles

The 16 primary rows are `branch_point=A`, `intervention=global_shuffle`, `realization=1`.

- `R_hit == 2,048`: 14/16.
- `R_hit == 4,096`: 2/16.

Thus every Branch-A global shuffle crossed by the second sampled observation.

### Negative untouched residual

Branch-A source `49461bd54e39582ab4fb` (density 0.20, seed 6202) has `untouched_R_hit = -28,672`, `untouched_hit_observed = false`, and later `untouched_R_persistent = 83,968`. The negative residual records an earlier transient hit before the Branch-A intervention even though Branch A excluded a prior persistent event. An earliest-hit comparison is therefore ambiguous for this progenitor. This is a secondary problem because the global-shuffle endpoint already fails construct validity independently.

## Branch termination

All 72 altered rows in `RESIDUAL_FIRST_PASSAGE.csv` have `completion_reason = post_persistent_capture_complete`. Direct parsing of all 72 retained raw `branch_result.json` files confirms that every branch ends exactly 16,384 phases after its recorded `R_persistent`. In particular, branches falsely classified at elapsed zero were generally retained only through elapsed 16,384. They were not continued until natural-scale low-k organisation returned.

## Immediate structure destruction

Means below come from `INTERVENTION_AUDIT.csv`. Nearest-neighbour mutual information is the mean at offset one. Branch A contains its 16 primary shuffles. For Branch B the main values use all 20 frozen global-shuffle realisations because the preregistered four-replicate subset is part of the global-shuffle audit; the primary-16 low-k means are also shown.

| Branch / metric | Source mean | Altered mean | Diagnostic change |
|---|---:|---:|---:|
| A low-k power, n=16 | 5,544.268 | 43.719 | 99.211% removed |
| A interface density, n=16 | 0.364832 | 0.455946 | increased |
| A spectral anisotropy, n=16 | 0.549951 | 0.159870 | strongly reduced |
| A nearest-neighbour mutual information, n=16 | 0.0208694 | 0.00000405 | effectively erased |
| B low-k power, all global shuffles, n=20 | 7,246.311 | 45.650 | 99.370% removed |
| B interface density, n=20 | 0.366745 | 0.466956 | increased |
| B spectral anisotropy, n=20 | 0.591044 | 0.179172 | strongly reduced |
| B nearest-neighbour mutual information, n=20 | 0.0240205 | 0.00000455 | effectively erased |
| B low-k power, primary only, n=16 | 7,800.923 | 45.257 | 99.420% removed |

The intervention therefore did what it was intended to do: it preserved the one-site state exactly while destroying the measured long-wavelength and nearest-neighbour organisation.

## Post-shuffle low-k recovery

Branch-B means from `MULTISCALE_MEMORY.csv`, including all frozen global-shuffle realisations, are:

| Elapsed phases | n | Mean low-k power | Mean low-k fraction |
|---:|---:|---:|---:|
| 0 | 20 | 45.650 | 0.0001866 |
| 8,192 | 20 | 729.492 | 0.0028191 |
| 16,384 | 20 | 1,127.719 | 0.0043519 |
| 32,768 | 2 | 2,215.712 | 0.0085149 |

The matched source mean before shuffling was 7,246.311. At the end of the standard 16,384-phase post-event capture, the altered mean remained only 15.6% of that already pre-transformation source structure. Only two branches survived to 32,768 phases, and even their mean was 30.6% of the source mean. The retained branch windows do not show recovery of the natural macrostate.

## Exact-state and visual cross-check

`FIGURE_02_HISTORY_ERASURE_BRANCHES` synchronizes exact saved uint8 states. In both displayed densities, the source and untouched panels retain large diagonal, continent-like occupancy structure. The globally shuffled panels become fine-grained, macroscopically homogeneous noise and remain visually far less organised throughout their retained windows. `FIGURE_05_MULTISCALE_MEMORY` independently shows that global shuffling retains essentially none of the source low-k precursor or spatial mutual information.

One exact Branch-B example is task `ed5f7a72ad06541c16fa` (density 0.24, seed 6202). Its retained `branch_result.json` has SHA-256 `2aeb572427e7570b86f9f4b174642eced8e57daef4359efb6333cb85e742c918`. At elapsed zero it records:

- exact altered-state SHA-256 `eeef37afa687e43c5279e5fe5977bc24084e124bfaffc0b4cf8a7a19b59959f7`;
- inventory `[619163, 145846, 145846, 137721]` and occupancy 0.409520;
- `Q = 0`;
- finite-empty scale and current scale ratio 157.6567;
- largest finite empty fraction 0.191215;
- low-k fraction 0.00024348;
- nearest-neighbour mutual information approximately `5.57e-6`;
- `strong_domain = true` and `R_hit = R_persistent = 0`.

This is the clearest construct-validity contradiction: a spatially shuffled low-k state is classified as already transformed because its finite-empty geometry passes the inherited thresholds.

## Geometric diagnosis

The classifier was calibrated on natural trajectories whose initial finite-empty scale was often one. At Branch B, the example above is 40.95% occupied and 59.05% empty after a uniform spatial shuffle. The empty cells form many connected finite clusters, including a very large finite component, rather than the smoothly organised excitation-rich/near-vacuum macrostate visible in untouched trajectories. With an inherited denominator of one, that random connected-component geometry gives a scale ratio of 157.7 and a largest-finite-empty fraction of 0.191, easily passing thresholds of 4.0, 2.0, and 0.001 even though long-wavelength power is negligible.

This supports the diagnostic explanation that high-occupancy random shuffles can trip the finite-empty-component endpoint through connected empty-cluster geometry. It does not prove an equilibrium percolation mechanism: the campaign contains no dedicated equilibrium ensemble or classifier-validating winding/percolation analysis sufficient for that stronger claim.

## Methodological conclusion and withdrawal

The frozen v0.3.2 classifier measured whether the inherited finite-empty-domain criterion fired, not whether genuine large-scale organisation had re-formed after shuffling. Under the intervention those constructs diverged. The originally reported timing differences are valid descriptions of a metric response but cannot support either acceleration or inhibition of genuine organisation.

Withdrawn conclusion: **spatial history inhibits transformation** (including the frozen labels `SPATIAL-HISTORY INHIBITION` and `D. HISTORY-INHIBITED TRANSFORMATION` when used as present scientific conclusions).

Replacement conclusion: the state-matched intervention exposed a non-intervention-invariant endpoint. It validly established exact inventory preservation and severe destruction of multiscale spatial organisation. It did not answer whether pre-existing spatial correlations accelerate, delay, or do not affect genuine macroscopic reorganisation.

