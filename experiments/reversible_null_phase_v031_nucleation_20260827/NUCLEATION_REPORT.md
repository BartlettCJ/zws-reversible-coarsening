# v0.3.1 Low-Density Nucleation and Metastability Report

## Top line

The complete preregistered campaign passed integrity audit: **156/156 trajectories**, exact winner and config hashes, and no recorded exact-Q violation. Positive control: **12/12 hits by 131,072 and 12/12 by horizon**.

This is a finite-size, finite-time first-passage study. Right-censoring means ‘not observed by the deadline,’ never ‘cannot happen.’

## Primary homogeneous first passage

| density | N | T_hit events / censored | median T_hit | q25 / q75 T_hit | T_persistent events / censored | median T_persistent | q25 / q75 T_persistent | earliest / latest T_hit |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.16 | 12 | 0 / 12 | &gt;262,144 | &gt;262,144 / &gt;262,144 | 0 / 12 | &gt;262,144 | &gt;262,144 / &gt;262,144 | — / — |
| 0.18 | 12 | 8 / 4 | 495,616 | 466,944 / &gt;524,288 | 8 / 4 | 495,616 | 466,944 / &gt;524,288 | 286,720 / 512,000 |
| 0.20 | 12 | 12 / 0 | 321,536 | 296,960 / 325,632 | 11 / 1 | 323,584 | 305,152 / 325,632 | 194,560 / 360,448 |
| 0.22 | 12 | 12 / 0 | 206,848 | 174,080 / 215,040 | 12 / 0 | 206,848 | 174,080 / 215,040 | 151,552 / 243,712 |
| 0.24 | 12 | 12 / 0 | 108,544 | 96,256 / 129,024 | 12 / 0 | 108,544 | 96,256 / 129,024 | 79,872 / 151,552 |
| 0.25 | 12 | 12 / 0 | 83,968 | 75,776 / 92,160 | 12 / 0 | 83,968 | 75,776 / 92,160 | 69,632 / 112,640 |
| 0.26 | 12 | 12 / 0 | 65,536 | 40,960 / 81,920 | 12 / 0 | 65,536 | 40,960 / 81,920 | 36,864 / 96,256 |

Values shown with a leading `>` are censoring bounds: the corresponding quantile was not reached by the horizon.

## Fixed-time transformation counts

| density | N | by 32,768 | by 65,536 | by 131,072 | by 262,144 |
| --- | --- | --- | --- | --- | --- |
| 0.16 | 12 | 0 | 0 | 0 | 0 |
| 0.18 | 12 | 0 | 0 | 0 | 0 |
| 0.20 | 12 | 0 | 0 | 0 | 2 |
| 0.22 | 12 | 0 | 0 | 0 | 12 |
| 0.24 | 12 | 0 | 0 | 10 | 12 |
| 0.25 | 12 | 0 | 0 | 12 | 12 |
| 0.26 | 12 | 0 | 6 | 12 | 12 |

## Descriptive hazard inspection

| density | events | untransformed phase-exposure | descriptive λ / phase | descriptive λ / cell / phase | early / middle / late λ | shape |
| --- | --- | --- | --- | --- | --- | --- |
| 0.16 | 0 | 3,145,728 | 0.0000e+00 | 0.0000e+00 | 0.0000e+00 / 0.0000e+00 / 0.0000e+00 | insufficient-events |
| 0.18 | 8 | 5,726,208 | 1.3971e-06 | 1.3324e-12 | 0.0000e+00 / 4.9156e-07 / 4.3895e-06 | strongly-nonstationary |
| 0.20 | 12 | 3,682,304 | 3.2588e-06 | 3.1079e-12 | 0.0000e+00 / 6.9875e-06 / 9.1553e-05 | strongly-nonstationary |
| 0.22 | 12 | 2,398,208 | 5.0037e-06 | 4.7719e-12 | 0.0000e+00 / 2.9653e-06 / 2.6634e-05 | strongly-nonstationary |
| 0.24 | 12 | 1,349,632 | 8.8913e-06 | 8.4794e-12 | 9.6055e-07 / 3.5649e-05 / 0.0000e+00 | strongly-nonstationary |
| 0.25 | 12 | 1,044,480 | 1.1489e-05 | 1.0957e-11 | 7.0280e-06 / 1.0316e-04 / 0.0000e+00 | strongly-nonstationary |
| 0.26 | 12 | 792,576 | 1.5141e-05 | 1.4439e-11 | 1.4036e-05 / 1.1268e-04 / 0.0000e+00 | strongly-nonstationary |

These rates are finite-cohort descriptions, not equilibrium nucleation constants or an assumed exponential fit.

## Topology timing relative to first passage

| density | wrap event | observed / hit trajectories | before / same / after T_hit | median offset |
| --- | --- | --- | --- | --- |
| 0.18 | occupied x | 0 / 8 | 0 / 0 / 0 | — |
| 0.18 | occupied y | 0 / 8 | 0 / 0 / 0 | — |
| 0.18 | empty x | 8 / 8 | 8 / 0 / 0 | -477,184 |
| 0.18 | empty y | 8 / 8 | 8 / 0 / 0 | -477,184 |
| 0.20 | occupied x | 4 / 12 | 0 / 0 / 4 | 180,224 |
| 0.20 | occupied y | 4 / 12 | 0 / 0 / 4 | 180,224 |
| 0.20 | empty x | 12 / 12 | 12 / 0 / 0 | -322,560 |
| 0.20 | empty y | 12 / 12 | 12 / 0 / 0 | -322,560 |
| 0.22 | occupied x | 1 / 12 | 0 / 0 / 1 | 67,584 |
| 0.22 | occupied y | 0 / 12 | 0 / 0 / 0 | — |
| 0.22 | empty x | 12 / 12 | 12 / 0 / 0 | -207,872 |
| 0.22 | empty y | 12 / 12 | 12 / 0 / 0 | -207,872 |
| 0.24 | occupied x | 11 / 12 | 0 / 0 / 11 | 110,592 |
| 0.24 | occupied y | 11 / 12 | 0 / 0 / 11 | 131,072 |
| 0.24 | empty x | 12 / 12 | 12 / 0 / 0 | -109,568 |
| 0.24 | empty y | 12 / 12 | 12 / 0 / 0 | -109,568 |
| 0.25 | occupied x | 12 / 12 | 0 / 0 / 12 | 104,448 |
| 0.25 | occupied y | 12 / 12 | 0 / 0 / 12 | 124,928 |
| 0.25 | empty x | 12 / 12 | 12 / 0 / 0 | -84,992 |
| 0.25 | empty y | 12 / 12 | 12 / 0 / 0 | -84,992 |
| 0.26 | occupied x | 12 / 12 | 0 / 0 / 12 | 102,400 |
| 0.26 | occupied y | 12 / 12 | 0 / 0 / 12 | 97,280 |
| 0.26 | empty x | 12 / 12 | 12 / 0 / 0 | -66,560 |
| 0.26 | empty y | 12 / 12 | 12 / 0 / 0 | -66,560 |

## Matched area scaling

| density | size | matched N | T_hit events / censored | median T_hit | median T_persistent | λ / cell / phase | shape |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0.22 | 512² | 8 | 7 / 1 | 172,032 | 176,128 | 1.8261e-11 | strongly-nonstationary |
| 0.22 | 1024² | 8 | 8 / 0 | 190,464 | 190,464 | 4.7516e-12 | strongly-nonstationary |
| 0.24 | 512² | 8 | 8 / 0 | 110,592 | 110,592 | 3.1504e-11 | strongly-nonstationary |
| 0.24 | 1024² | 8 | 8 / 0 | 108,544 | 108,544 | 8.4474e-12 | strongly-nonstationary |
| 0.26 | 512² | 8 | 8 / 0 | 57,344 | 61,440 | 6.1831e-11 | strongly-nonstationary |
| 0.26 | 1024² | 8 | 8 / 0 | 65,536 | 65,536 | 1.4273e-11 | strongly-nonstationary |

Only seed identities present in both the 512² and 1024² cohorts are included.

## Matched seeded-basin comparison

| density | arrangement | matched N | T_hit events / censored | median T_hit | persistent events | occupied retention | final empty scale | final interface | final low-k | final occupancy | mean reaction |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.20 | homogeneous | 6 | 1 / 5 | &gt;262,144 | 1 | 804.9 | 2.092 | 0.3526 | 6.741e+09 | 0.3863 | 0.03695 |
| 0.20 | droplet | 6 | 0 / 6 | &gt;262,144 | 0 | 1.273 | 1.061 | 0.1254 | 8.547e+10 | 0.2459 | 0.02992 |
| 0.20 | slab | 6 | 0 / 6 | &gt;262,144 | 0 | 1.308 | 1.065 | 0.1294 | 6.406e+10 | 0.2523 | 0.03026 |
| 0.22 | homogeneous | 6 | 6 / 0 | 212,992 | 6 | 3473 | 12.4 | 0.3745 | 6.944e+09 | 0.4827 | 0.04606 |
| 0.22 | droplet | 6 | 0 / 6 | &gt;262,144 | 0 | 1.265 | 1.064 | 0.1369 | 8.943e+10 | 0.2685 | 0.03274 |
| 0.22 | slab | 6 | 0 / 6 | &gt;262,144 | 0 | 1.29 | 1.067 | 0.1397 | 6.661e+10 | 0.2732 | 0.0331 |
| 0.24 | homogeneous | 6 | 6 / 0 | 110,592 | 6 | 1.21e+04 | 67.4 | 0.3856 | 5.626e+09 | 0.5636 | 0.05499 |
| 0.24 | droplet | 6 | 0 / 6 | &gt;262,144 | 0 | 1.268 | 1.064 | 0.1502 | 9.317e+10 | 0.2944 | 0.03582 |
| 0.24 | slab | 6 | 0 / 6 | &gt;262,144 | 0 | 1.301 | 1.065 | 0.1528 | 6.897e+10 | 0.2987 | 0.0361 |
| 0.26 | homogeneous | 6 | 6 / 0 | 59,392 | 6 | 1.639e+04 | 21.93 | 0.3896 | 3.628e+09 | 0.6297 | 0.06281 |
| 0.26 | droplet | 6 | 0 / 6 | &gt;262,144 | 0 | 1.261 | 1.066 | 0.1618 | 9.593e+10 | 0.3169 | 0.03874 |
| 0.26 | slab | 6 | 0 / 6 | &gt;262,144 | 0 | 1.278 | 1.064 | 0.1632 | 7.075e+10 | 0.3198 | 0.03891 |

Only seed identities present in homogeneous, droplet, and slab cohorts are included; all arrangements have identical global inventories by construction.

## Statistical and scientific questions

1. **MEASUREMENT — T_hit vs density.** Events by horizon: 0.18 (8/12), 0.20 (12/12), 0.22 (12/12), 0.24 (12/12), 0.25 (12/12), 0.26 (12/12). The survival curves and censoring-aware summaries quantify the full dependence.

2. **MEASUREMENT — T_persistent vs density.** Persistent-event counts and censoring-aware KM medians/quartiles are tabulated separately; a transient T_hit is never promoted to persistence.

3. **FORMAL INFERENCE — old ~27.5% threshold.** Hits below 27.5% at the longer horizon show that the earlier boundary was at least partly finite-observation-time dependent.

4. **MEASUREMENT — horizon dependence.** The apparent boundary moves downward with time: density 0.22 has 0/12 hits by phase 131,072 but 12/12 by 262,144; density 0.20 has 2/12 by 262,144 but 12/12 after exact continuation; density 0.18 reaches 8/12 only in Stage 2.

5. **FORMAL INFERENCE — lower hard boundary.** Censored cohorts establish only non-observation through 262,144 phases, never impossibility or an infinite waiting time.

6. **SCIENTIFIC INTERPRETATION — longer waits.** The presence of lower-density events together with rising event-free times supports progressively longer waiting times rather than a hard boundary within the event-bearing densities tested.

7. **MEASUREMENT — identifiable medians.** KM median T_hit is identifiable at densities 0.18, 0.20, 0.22, 0.24, 0.25, 0.26, and KM median T_persistent at 0.18, 0.20, 0.22, 0.24, 0.25, 0.26; other entries show the explicit horizon lower bound rather than an observed-only median.

8. **MEASUREMENT — growth of waiting time.** The characteristic-time figure compares restricted mean event-free time through the common phase-262,144 horizon, while KM medians use each cohort's full frozen horizon (including exact Stage 2 continuations). The descriptive hazard table inspects early/middle/late rates without assuming exponential waiting times.

9. **MEASUREMENT — width.** Waiting times broaden markedly at lower density: the density-0.18 interquartile interval runs from 466,944 to beyond 524,288 with 4/12 censored, whereas density-0.26 spans 40,960 to 81,920 with no censoring.

10. **SCIENTIFIC INTERPRETATION — mechanism.** Frozen event labels are {'B': 67, 'E': 24}. The dominance of label B and absence of label A support distributed/global coarsening rather than sudden growth of one localized nucleus; label E retains the unresolved remainder.

11. **MEASUREMENT — area ordering.** The 1024² worlds are not systematically faster: median T_hit is 172,032 versus 190,464 at density 0.22 (512² versus 1024²), 110,592 versus 108,544 at 0.24, and 57,344 versus 65,536 at 0.26.

12. **FORMAL INFERENCE — area-normalized compatibility.** The smaller lattice has approximately 3.7 to 4.3 times the descriptive event rate per cell at every matched density, so the cohorts do not show the collapse expected from a simple area-proportional local Poisson process. Small cohorts and nonstationary hazards still preclude a universal scaling law.

13. **MEASUREMENT — seeded persistence.** At the common phase-262,144 horizon, matched homogeneous worlds hit in 19/24 cases; droplet/slab worlds hit in 0/48 but retain and modestly grow their imposed occupied macrostructure, with median retention ratios from 1.26 to 1.31.

14. **SCIENTIFIC INTERPRETATION — multiple basins.** Operational basin dependence is supported where identical global inventories follow persistently separated arrangement-dependent trajectories: imposed droplet/slab macrostructure survives and grows modestly without entering the strong-domain class, while matched homogeneous worlds coarsen toward or into it. This is not an equilibrium free-energy measurement.

15. **MEASUREMENT — topology timing.** Among observed primary hits, median recorded wrap offset is -83,968 phases relative to T_hit. The density/type table reports whether each winding clock precedes, coincides with, or follows T_hit; all four remain independent of the strong-domain definition.

16. **MEASUREMENT — around first passage.** Across frozen representatives, median finite-empty scale rises sharply at T_hit while low-k power is already growing before passage and continues afterward; the morphology sequences show distributed textured coarsening rather than one isolated expanding nucleus.

17. **SCIENTIFIC INTERPRETATION — strongest claim.** The jointly preregistered evidence supports a finite-size, finite-time density-dependent kinetic onset dominated by distributed coarsening, together with operational arrangement-dependent basins. It does not support simple local Poisson nucleation or establish an equilibrium first-order transition.

18. **SCIENTIFIC INTERPRETATION — alternatives.** Live alternatives include global coarsening, finite-size topology, a classifier-crossing artifact, broad deterministic seed variability, and observation-grid aliasing.

19. **SPECULATION — paper case.** Yes. The complete integrity-audited result is sufficiently self-contained and unusual for a dedicated study or paper, provided its claims remain explicitly finite-size, finite-time, and operational rather than thermodynamic.

20. **DECISION — further v0.3 work.** C — The nucleation/metastability result justifies a standalone dedicated study or paper before primary effort moves to the Causal Nullity Ecology; freeze this campaign and do not launch the broad powered profile.

## Final A/B/C decision

**C. The nucleation/metastability result justifies a standalone dedicated study or paper before primary effort moves to the Causal Nullity Ecology; freeze this campaign and do not launch the broad powered profile.**

Scientific claim: In an exactly reversible lattice constrained to exact Q = M = 0, density controls a steep, strongly nonstationary waiting-time spectrum for distributed macroscopic coarsening, while identical global inventories can occupy long-lived arrangement-dependent basins; the earlier approximately 27.5% onset is therefore primarily a finite-observation-time kinetic boundary, not a demonstrated hard density threshold.

Rationale: The positive control reproduced (12/12 at density 0.26 by phase 131,072). Exact continuations found all 12 density-0.20 trajectories transformed by phase 360,448 and 8/12 density-0.18 trajectories transformed by phase 524,288, while density 0.16 remained right-censored at phase 262,144. Hazards were strongly nonstationary. Frozen event labels were 67 global-coarsening (B), 24 unresolved (E), and zero localized-nucleus (A). The 512-squared versus 1024-squared comparison did not collapse after division by area: per-cell descriptive rates were roughly four times higher in the smaller lattice, arguing against a simple local Poisson nucleation model. At the common phase-262,144 horizon, matched homogeneous worlds hit in 19/24 cases, whereas matched droplet and slab worlds hit in 0/48 while retaining and modestly growing their imposed occupied macrostructure (median retention ratios 1.26 to 1.31). Those jointly preregistered measurements support a paper-scale claim of density-dependent kinetic onset, distributed coarsening, and operational basin dependence, while leaving equilibrium first-order-transition language and infinite-time boundaries unclaimed.

## Evidence layers

- **MEASUREMENT:** task-level first passage, persistence, topology, event windows, exact inventories, and fixed-time states.
- **FORMAL INFERENCE:** Kaplan-Meier survival/cumulative incidence, identifiable quantiles, restricted means, and descriptive exposure rates.
- **SCIENTIFIC INTERPRETATION:** nucleation-like timing and operational metastability only where multiple preregistered comparisons agree.
- **SPECULATION:** equilibrium phase-transition analogies remain hypotheses, not results of this experiment.

## Reproducibility

The report includes 12 hashed figures, all requested CSV products, an embedded local dashboard, and the integrity audit. Raw trajectories/checkpoints remain outside Git.
