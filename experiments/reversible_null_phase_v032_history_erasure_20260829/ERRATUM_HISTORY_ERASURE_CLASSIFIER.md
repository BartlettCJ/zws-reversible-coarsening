# Erratum: v0.3.2 history-erasure event classifier

Date: 2026-08-31 (JST)

This is an additive correction to the frozen v0.3.2 record. It does not alter `HISTORY_ERASURE_REPORT.md`, the preregistration, frozen plan, manifests, decision JSON, result tables, state hashes, or original figures.

## What the frozen endpoint classified

The v0.3.2 analysis inherited a strong-domain endpoint based on growth and current scale of finite connected empty components, a largest-finite-empty-component threshold, and exact charge preservation. Under the frozen decision rules, earlier residual crossings in state-matched global-shuffle branches were labelled `SPATIAL-HISTORY INHIBITION`; agreement at Branches A and B produced `D. HISTORY-INHIBITED TRANSFORMATION`.

Those labels are preserved in the historical files as the output of the frozen classifier. They are withdrawn as scientific conclusions about genuine macroscopic organisation.

## What cross-checking revealed

Independent post-campaign inspection compared the residual-time table with the immediate intervention audit, multiscale memory table, exact branch states, and synchronized figures. Global shuffling correctly preserved every one-site quantity while destroying the macrostructure the event was intended to represent. The inherited endpoint nevertheless fired immediately or within a few observation intervals on most shuffled states.

The endpoint was therefore not intervention-invariant: after global shuffling, satisfying it no longer meant that the excitation-rich/near-vacuum, long-wavelength organised macrostate had formed.

## Exact verified numbers

From `RESIDUAL_FIRST_PASSAGE.csv`:

- Branch-B primary global shuffles (`n=16`): `R_hit == 0` in 9/16; `R_hit <= 4,096` in 14/16; `R_hit <= 6,144` in 15/16; the remaining value was 20,480; mean `R_hit` was 2,432 phases.
- Branch-A primary global shuffles (`n=16`): `R_hit == 2,048` in 14/16 and `R_hit == 4,096` in 2/16.
- Every one of the 72 altered branches ended with `completion_reason = post_persistent_capture_complete`.
- Direct inspection of the 72 retained raw `branch_result.json` files confirms that every branch ended exactly 16,384 phases after its recorded persistent hit.
- Branch-A source `49461bd54e39582ab4fb` (density 0.20, seed 6202) had `untouched_R_hit = -28,672`: an earlier transient hit before the intervention, followed by `untouched_R_persistent = 83,968`.

From `INTERVENTION_AUDIT.csv`:

| Branch / metric | Source mean | Altered mean |
|---|---:|---:|
| A low-k power, 16 primary shuffles | 5,544.268 | 43.719 |
| A interface density | 0.364832 | 0.455946 |
| A spectral anisotropy | 0.549951 | 0.159870 |
| A nearest-neighbour mutual information | 0.0208694 | 0.00000405 |
| B low-k power, all 20 global shuffles | 7,246.311 | 45.650 |
| B interface density | 0.366745 | 0.466956 |
| B spectral anisotropy | 0.591044 | 0.179172 |
| B nearest-neighbour mutual information | 0.0240205 | 0.00000455 |

The low-k reduction was 99.211% at Branch A and 99.370% across all Branch-B global-shuffle realisations. Using only the 16 primary Branch-B realisations gives 7,800.923 to 45.257, a 99.420% reduction.

From `MULTISCALE_MEMORY.csv`, the Branch-B mean low-k powers across all frozen global-shuffle realisations were 45.650 at elapsed zero (`n=20`), 729.492 at +8,192 (`n=20`), 1,127.719 at +16,384 (`n=20`), and 2,215.712 at +32,768 (`n=2`). The matched pre-shuffle source mean was 7,246.311. Most branches therefore ended while their measured long-wavelength structure remained far below the natural source macrostate.

The value expected in two external reviews for `R_hit <= 6,144` was 14/16. Direct enumeration gives 15/16; 14/16 is the count for `R_hit <= 4,096`. The corrected repository value is used here.

## Why the endpoint was invalid under the intervention

The classifier inherited its baseline from natural progenitor trajectories. A global shuffle creates a high-occupancy random arrangement with connected finite empty clusters that can be very large relative to the small inherited baseline. It can therefore pass finite-empty scale and largest-finite-component thresholds even while low-k power, anisotropy, and spatial mutual information show that long-wavelength organisation has been erased.

For example, Branch-B task `ed5f7a72ad06541c16fa` was 40.952% occupied at elapsed zero, had finite-empty scale 157.6567, largest finite empty fraction 0.191215, and low-k fraction only 0.00024348. The classifier nevertheless set `strong_domain = true` with `R_hit = R_persistent = 0`. The exact state was visually fine-grained rather than continent-like.

This supports a connected-empty-cluster geometric diagnosis. It does not prove an equilibrium percolation mechanism; no dedicated equilibrium or classifier-validating percolation analysis was performed.

## What v0.3.2 still established

The campaign validly established that:

- all 72 altered branches passed exact intervention audits;
- global and tile interventions preserved the complete four-state histogram, `Q`, `M`, single-site entropy, lattice size, and Margolus phase;
- the translation-null implementation passed exact conjugacy checks;
- global shuffling removed more than 99% of the measured low-k structure and essentially all nearest-neighbour mutual information while increasing interface density;
- tile interventions retained scale-dependent structure;
- checkpoint, pause/unload, exact-resume, implementation parity, and artifact integrity gates passed;
- the inherited event classifier was not robust to this scientifically relevant intervention.

This is a valid methodological negative result and an endpoint-robustness lesson.

## Withdrawn conclusion

Withdrawn: **spatial history inhibits transformation**, including the unqualified scientific labels `SPATIAL-HISTORY INHIBITION`, `HISTORY-INHIBITED TRANSFORMATION`, and the lay claim that accumulated history holds organisation back.

The residual-time numbers remain correct measurements of when the frozen finite-empty-domain score fired. They are not estimates of when genuine large-scale organisation formed after shuffling.

## Unresolved causal question

The current campaign does not determine whether pre-existing spatial correlations accelerate, delay, or do not affect genuine macroscopic reorganisation. Answering that question would require a separately validated intervention-invariant endpoint and a new preregistered campaign. No such campaign is authorized for the present publication cycle.

## Review trail

The review reports behind this erratum are retained by the author and are not part of this release.
