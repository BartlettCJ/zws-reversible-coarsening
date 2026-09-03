# ZWSLab v0.1.4 Homogeneous Density-Threshold Map

## Top-line result

For the unchanged reversible winner rule, homogeneous exactly `Q=0` initial conditions show a
reproducible, very sharp **finite-size dynamical onset** of spontaneous large-domain formation near
27.5% global occupancy.

- At 1024 x 1024 for 32,768 phases, the frozen strong-domain pass fraction was 1/12 at 26%, 4/16
  at 27%, 8/12 at 28%, 16/16 at 29%, and 12/12 at 30%.
- The preregistered combined response class is
  **D — VERY SHARP FINITE-SIZE TRANSITION REGION**.
- The descriptive logistic midpoint is **0.274723** occupancy (blocked-bootstrap 95% interval
  **0.270854–0.278420**).
- The fitted 25%-to-75% width is **0.013242** occupancy, or 1.32 percentage points (95% interval
  **0.010481–0.016285**).
- Low-wavenumber growth turns on much earlier: 2/12 low-k passes at 10% and 12/12 at 12%, while
  strong domains do not become common until about 28%. Low-k amplification is therefore a broad
  precursor, not the strong-domain onset itself.
- The gated 512/1024/2048 check keeps the response in the same narrow density neighbourhood, but
  its three densities and small seed counts do not establish finite-size scaling. The 2048 sample
  changed from 1/4 passes at 27.47% to 4/4 at 28.13%.
- Above onset, dense interiors remain extremely disordered (median normalized exact-state entropy
  0.970 at 28%, 0.977 at 29%, and 0.990 at 32%). No trajectory had the frozen simple-regular
  spectral signature: 0/200.

This is strong evidence for a density-controlled finite-size instability in this automaton. It is
not, by itself, proof of a thermodynamic phase transition.

![Combined pass probability and descriptive fit](figures/final_1024/density_pass_probability_and_fit.png)

## Frozen design and integrity

The winner table was never changed:

`d1fe99b5735d8e9be032347de2727dc5c52a7848df200ca26f67f28c8c1bc41c`

Every initial state was a homogeneous random shuffle of an exact inventory with positive and
negative counts equal, fixed occupied composition 3/8 positive, 3/8 negative, 2/8 bound-neutral,
and exact global `Q=0`. The deterministic rounding rule was
`u=floor(target*N/8+0.5)` and counts `(N-8u,3u,3u,2u)`.

The scientific outcomes of a stage were not opened until the whole stage completed and its files
verified. The study comprises:

| Stage | World | Densities | Seeds per density | Trajectories | Status | Failures |
|---|---:|---:|---:|---:|---|---:|
| Preregistered broad grid | 1024² | 14 | 12 | 168 | complete | 0 |
| Frozen Stage-2 refinement | 1024² | 2 | 16 | 32 | complete | 0 |
| Gated finite-size check | 512² | 3 | 8 | 24 | complete | 0 |
| Gated finite-size check | 1024² | 3 | 8 | 24 | complete | 0 |
| Gated finite-size check | 2048² | 3 | 4 | 12 | complete | 0 |
| **Total** |  |  |  | **260** | **complete** | **0** |

All 200 primary 1024 trajectories preserved exact `Q=0` at every saved phase. The final 1024
integrity audit verified frozen initial hashes, trajectory hashes, required compact fields, unique
run identities, 1,200 spatial-composition rows, and 600 exploratory cluster rows. The finite-size
campaign passed 143/143 completion/checkpoint/snapshot/trajectory hash checks before analysis.

The two immutable discovery tags remained at
`74b4243c9518ec6f95b906de1a374bc277a2a5d5`. No rule search, v0.2 edit, 4096² run, or post-outcome
criterion change occurred.

## Preregistered broad grid

The broad grid used the same 12 blocked seed identities at every density. `L` is the established
finite empty-domain characteristic scale; ratios are relative to the measured initial scale with
the original denominator floor.

| Target occupancy | Realized occupancy | Strong domain | Low-k growth | Median peak L ratio | Median final L ratio |
|---:|---:|---:|---:|---:|---:|
| 0.0646 | 0.064598 | 0/12 | 0/12 | 1.750 | 1.289 |
| 0.0800 | 0.080002 | 0/12 | 0/12 | 1.635 | 1.381 |
| 0.1000 | 0.099998 | 0/12 | 2/12 | 1.541 | 1.411 |
| 0.1200 | 0.120003 | 0/12 | 12/12 | 1.560 | 1.450 |
| 0.1400 | 0.139999 | 0/12 | 12/12 | 1.540 | 1.468 |
| 0.1600 | 0.160004 | 0/12 | 12/12 | 1.467 | 1.402 |
| 0.1800 | 0.180000 | 0/12 | 12/12 | 1.474 | 1.438 |
| 0.2000 | 0.199997 | 0/12 | 12/12 | 1.483 | 1.445 |
| 0.2200 | 0.220001 | 0/12 | 12/12 | 1.562 | 1.462 |
| 0.2400 | 0.239998 | 0/12 | 12/12 | 1.878 | 1.655 |
| 0.2600 | 0.260002 | 1/12 | 12/12 | 2.342 | 2.037 |
| 0.2800 | 0.279999 | 8/12 | 12/12 | 4.765 | 3.709 |
| 0.3000 | 0.300003 | 12/12 | 12/12 | 15.294 | 11.266 |
| 0.3200 | 0.320000 | 12/12 | 12/12 | 77.132 | 34.059 |

The frozen broad classifier returned class C with a 26–30% empirical bracket. This deterministically
triggered Stage 2 at the only new 0.01-spaced interior targets, 27% and 29%.

## Preregistered Stage-2 refinement

Stage 2 used 16 newly frozen blocked seeds per target. Its manifest, input hashes, and exact
inventories were committed before any refinement evolution.

| Target occupancy | Realized occupancy | Strong domain | Low-k growth | Median peak L ratio | Median final L ratio |
|---:|---:|---:|---:|---:|---:|
| 0.2700 | 0.269997 | 4/16 | 16/16 | 3.375 | 2.417 |
| 0.2900 | 0.290001 | 16/16 | 16/16 | 7.103 | 5.348 |

The combined pass fractions have no severe reversal. Spearman correlation between density and
pass fraction is 0.8669, above the preregistered 0.8 fit gate.

## Combined onset fit

The fit uses each trajectory's binary strong-domain result, a positive-slope logistic MAP model,
density centred at 0.20 and scaled by 0.10, and weak independent Normal(0,10) regularization. The
uncertainty is a deterministic blocked nonparametric bootstrap that resamples whole seed curves
within the broad and refinement stages.

| Quantity | Point estimate | Blocked-bootstrap 95% interval |
|---|---:|---:|
| p25 occupancy | 0.268102 | 0.263402–0.272770 |
| p50 occupancy | 0.274723 | 0.270854–0.278420 |
| p75 occupancy | 0.281344 | 0.277593–0.284848 |
| p25-to-p75 width | 0.013242 | 0.010481–0.016285 |

All 2,000/2,000 bootstrap refits succeeded. The model is descriptive; its sigmoid form is not a
microscopic theory.

The raw combined empirical 25%-to-75% bracket is 27–29%, width 0.020004 after exact inventory
rounding. Both the empirical and fitted widths meet the preregistered class-D cutoff of 0.03.

## Continuous observables

The binary onset is embedded in a rapid but continuous increase of domain scale. Reaction density,
interface density, exact-state entropy, and net occupancy change do not show a comparable jump at
27–28%; they rise smoothly. Thus the sharp classifier response is not merely a discontinuity in
the overall reaction count.

| Target | Strong pass | Median peak L ratio | Median final L ratio | Median peak low-k ratio | Final–initial occupancy | Mean reaction density | Dense entropy, passes only | Largest empty-domain fraction |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.24 | 0/12 | 1.878 | 1.655 | 64.390 | +0.0990 | 0.03930 | — | 0.9362 |
| 0.26 | 1/12 | 2.342 | 2.037 | 70.973 | +0.1229 | 0.04513 | 0.953 | 0.9028 |
| 0.27 | 4/16 | 3.375 | 2.417 | 66.313 | +0.1355 | 0.04795 | 0.962 | 0.8766 |
| 0.28 | 8/12 | 4.765 | 3.709 | 67.170 | +0.1480 | 0.05076 | 0.970 | 0.8445 |
| 0.29 | 16/16 | 7.103 | 5.348 | 76.464 | +0.1606 | 0.05361 | 0.977 | 0.8004 |
| 0.30 | 12/12 | 15.294 | 11.266 | 60.520 | +0.1703 | 0.05605 | 0.983 | 0.7208 |
| 0.32 | 12/12 | 77.132 | 34.059 | 53.986 | +0.1904 | 0.06131 | 0.990 | 0.1328 |

Above onset the final saved occupancy is also the median observed peak: 0.428 at 28%, 0.451 at
29%, 0.470 at 30%, and 0.510 at 32%. Net material occupancy therefore continues to rise during
the measured horizon. This does not prove what happens beyond 32,768 phases.

The system is not static above onset. Median final one-phase activity is 0.556 at 28%, 0.575 at
29%, and 0.621 at 32%; corresponding one-phase reaction density is 0.0550, 0.0584, and 0.0662.
Exact-state recurrence is high but not exact (for example median lag-128 recurrence is 0.921 at
28%, 0.915 at 29%, and 0.894 at 32%).

![Continuous observables across density](figures/final_1024/density_continuous_observables.png)

## Morphology and exploratory geometry

The common-scale, frozen-seed morphology grid supports the following descriptive sequence:

- at low density: sparse, fine-grained noise;
- from about 12% through 24%: low-k-amplified, weakly textured droplets/patches without strong
  finite-domain growth;
- at 26–27%: coarsening droplets and labyrinth precursors, with seed-dependent strong-domain
  success;
- at 28–30%: excitation-rich amorphous labyrinths and broad near-vacuum channels become common;
- at 32%: very coarse excitation-rich regions separated by large low-occupancy channels.

The visual change is a coarsening/amplification of irregular texture, not a switch to stripes,
checkerboards, or a simple crystal. The frozen simple-regular spectral signature was absent in
all 200 primary trajectories. Dense-interior entropy rises toward one rather than falling toward a
small periodic-state alphabet.

![Representative density morphology](figures/final_1024/representative_density_morphology_grid.png)

Exploratory four-neighbour cluster geometry does not show an ordinary occupied-cluster
percolation event coincident with the strong-domain onset. Around onset, the median largest
occupied cluster remains only 0.00349 of the world at 27%, 0.00407 at 28%, and 0.00587 at 29%,
while the largest empty cluster changes smoothly from 0.521 to 0.483 to 0.441 of the world. The
strong-domain classifier instead responds to connected empty-domain scale and persistence.

No toroidal winding test was performed. “Percolation” is therefore not asserted; these diagnostics
only argue that simple geometric occupied-site percolation is not a sufficient explanation.

![Exploratory cluster geometry](figures/final_1024/density_exploratory_cluster_geometry.png)

## Gated small finite-size check

The class-D result, valid main fit, and width below 0.06 triggered the preregistered check at fitted
p25/p50/p75 targets rounded to 0.2681, 0.2747, and 0.2813. All inputs were frozen and committed
before the 60 trajectories ran.

| World | 26.81% | 27.47% | 28.13% | Descriptive fitted p50 | Support note |
|---:|---:|---:|---:|---:|---|
| 512² | 0/8 | 4/8 | 5/8 | 0.279835 | Point p50 is in range; 95% interval extends above sampled range |
| 1024², new seeds | 2/8 | 3/8 | 4/8 | 0.287376 | Extrapolative; bootstrap is effectively unbounded and not interpreted |
| 2048² | 0/4 | 1/4 | 4/4 | 0.277730 | p50 and its 95% interval 0.271183–0.281088 are in sampled range |

The raw 2048 response has an empirical 25%-to-75% bracket from realized 0.274700 to 0.281300,
width 0.006599. The 512 and independent 1024 subsets do not reach 75% inside the three sampled
densities, so their fitted widths require extrapolation and are not evidence for a width trend.

Among worlds whose p50 point estimate lies inside the sampled window (512 and 2048), point p50
differs by 0.00210 occupancy. More conservatively, all three raw curves change within the same
26.81–28.13% neighbourhood and show no monotonic size-dependent displacement. The 2048 curve is
consistent with sharpening, but the small sample and lack of an in-window 75% point at 512/1024
prevent a claim that sharpening with size has been demonstrated.

![Finite-size raw pass fractions](figures/finite_size/finite_size_pass_probability.png)

## Explicit answers to the 15 scientific questions

### 1. At what homogeneous occupancy does spontaneous large-domain organisation first become common?

**Measurement:** The first tested density with a majority of strong-domain passes is 28% (8/12).
At 27% the fraction is 4/16, exactly 25%; at 29% it is 16/16.

**Formal inference:** The fitted majority crossing is 0.274723 occupancy.

### 2. What occupancy corresponds approximately to 50% strong-domain probability?

**Measurement/model estimate:** 0.274723, with blocked-bootstrap 95% interval
0.270854–0.278420. This is a descriptive finite-size estimate at 1024² and 32,768 phases.

### 3. How broad is the 25%-to-75% onset interval?

**Measurement:** The raw tested bracket is 0.269997–0.290001, width 0.020004.

**Model estimate:** p25 0.268102, p75 0.281344, width 0.013242 (95% interval
0.010481–0.016285).

### 4. How should the change be classified?

**Formal frozen classification:** **D — VERY SHARP FINITE-SIZE TRANSITION REGION.** There is no
severe reversal. This label does not assert a thermodynamic phase transition.

### 5. Does low-k amplification turn on at the same density as finite-domain growth?

**Measurement:** No. Low-k passes are 2/12 at 10% and 12/12 from 12% upward. Strong-domain passes
remain 0/12 through 24% and become common only around 28%.

**Interpretation:** Low-k amplification is an early large-scale spectral precursor. A second,
higher-density condition is needed for sustained finite-domain scale growth.

### 6. Does occupancy itself continue to increase dynamically above onset?

**Measurement:** Yes. Median final-minus-initial occupancy is +0.148 at 28%, +0.161 at 29%,
+0.170 at 30%, and +0.190 at 32%. The final saved value is also the median observed peak.

**Limitation:** The experiment ends at 32,768 phases, so later saturation is not measured.

### 7. Do reaction rates change sharply near onset?

**Measurement:** No comparable jump is resolved. Median mean reaction density rises smoothly from
0.0451 at 26% to 0.0479, 0.0508, 0.0536, and 0.0561 at 27–30%.

**Interpretation:** The sharp domain-probability rise is a nonlinear organisation/coarsening event,
not simply a sudden increase in the aggregate number of reactions.

### 8. Does morphology change qualitatively across onset?

**Measurement:** Yes, descriptively: small droplets/textured noise give way to coarse amorphous
labyrinths and broad excitation-rich/near-vacuum regions. Quantitatively, median peak L ratio rises
from 2.34 at 26% to 3.37, 4.77, 7.10, and 15.29 at 27–30%.

**Interpretation:** The change is rapid coarsening of an irregular morphology, not crystallization.

### 9. Are above-threshold dense regions still internally high-entropy and irregular?

**Measurement:** Yes. Median dense-interior normalized entropy among strong passes is 0.970 at
28%, 0.977 at 29%, 0.983 at 30%, and 0.990 at 32%.

### 10. Is there evidence of simple crystallization or a regular periodic lattice?

**Measurement:** No. The frozen simple-regular spectral signature occurs in 0/200 primary
trajectories, and the matched images show amorphous rather than periodic geometry.

### 11. Does onset resemble geometric percolation, a dynamical instability, or a combination?

**Measurement:** Strong-domain probability changes sharply while largest occupied/empty cluster
fractions vary smoothly near onset, and occupied clusters remain a tiny fraction of the world.

**Formal boundary:** No winding test was run, so percolation cannot be formally accepted or ruled
out.

**Interpretation:** The evidence favours a reaction-driven dynamical coarsening instability with
geometry as a contributing constraint, rather than ordinary independent-site percolation alone.

### 12. How sensitive is the onset to lattice size?

**Measurement:** All three finite-size curves change in the same 26.81–28.13% target window. The
2048 p50 is 0.277730 (95% 0.271183–0.281088). The 512 in-window p50 point estimate is 0.279835.
The independent three-point 1024 fit is extrapolative and not used as a precise size estimate.

**Interpretation:** No strong systematic location shift is resolved. The 2048 raw curve is sharper,
but monotonic sharpening is not established because 512/1024 do not bracket 75% within the sampled
window. This was deliberately a small check, not finite-size scaling.

### 13. What is the strongest scientific conclusion justified?

**Measurement:** Under one frozen reversible rule, homogeneous exact-`Q=0` worlds change from
almost never passing the strong-domain criterion below 27% to consistently passing at 29% and
above, with a well-resolved 1024 midpoint near 27.5%. The produced domains are dynamic,
high-entropy, and non-periodic.

**Formal inference:** Density is a control parameter for a very sharp finite-size dynamical onset
of spontaneous large-domain organisation over the measured sizes and time horizon.

**Not inferred:** A thermodynamic phase transition, universal critical exponent, equilibrium
critical point, or applicability to physical matter.

### 14. What is the strongest relevance to Zero Without Stasis?

**Measurement:** Exact global `Q=0` held at every saved phase in every primary trajectory, while
above-threshold worlds developed large moving domains, retained substantial one-phase activity and
reaction density, and remained internally disordered.

**Formal inference:** In this finite reversible automaton, global signed nullity does not entail a
blank, frozen, or structurally trivial state. The same exact-zero constraint supports qualitatively
different dynamical regimes selected by unsigned material density.

**Interpretation:** This is a concrete computational example of “zero as balanced cancellation,
not stasis”: nullity can coexist with motion, reaction, memory, and spontaneous organisation.

**Speculation:** It may be a useful conceptual toy model for emergence under balance constraints.
It is not evidence about cosmology, biology, quantum fields, or the physical origin of a universe.

### 15. Is there a scientific reason to keep v0.1 active after this experiment?

**Decision:** Do not keep v0.1 open for rule search or indefinite plotting. Freeze it as a completed
baseline. The newly established two-stage behaviour—low-k amplification near 11% followed by a
very sharp amorphous-domain onset near 27.5% that remains visible at 2048²—does justify a separate,
purpose-built scientific study before v0.2.

## Limitations

- The principal classification is finite-size and finite-time: 1024², 32,768 phases.
- The logistic curve is descriptive and weakly regularized, not a derived microscopic law.
- The finite-size check has only three densities and 8/8/4 seeds per density; several edge
  estimates extrapolate outside its sampled window.
- Cluster diagnostics are snapshot-only and exploratory; no toroidal winding test was added.
- Morphological labels are descriptive. The frozen strong-domain and spectral tests carry the
  formal classifications.
- The experiment locates an onset for one exact rule and initial-state ensemble; it does not claim
  universality across rules.

## Artifact index

| Artifact | SHA-256 |
|---|---|
| `BROAD_MANIFEST.json` | `9453e0b02415d2b9ac5fda68792427aa4dc86ac1b3bf503297161a21ab1d0ff1` |
| `BROAD_INPUT_AUDIT.json` | `e44ef9408658d8bd90ffbccb13554a67b9411d3617922a3c90c00ac988b8e40a` |
| `REFINEMENT_MANIFEST.json` | `aec4862a401b92d74de54a93dd0bfbb4744ce1b97699ca44b3d68b68b1cc791e` |
| `REFINEMENT_INPUT_AUDIT.json` | `3adc190b1084f51a48de455a8adc8965b196bfc20330d72a46fc7d004ba96d83` |
| `FINITE_SIZE_MANIFEST.json` | `3e69a14d3c22d602c82f730ffde14fa742b98b1e6336361f18aca0aafdb9db15` |
| `FINITE_SIZE_INPUT_AUDIT.json` | `f0208e4044789b26626394f937909da067af8b839bb0b287964034922c3dbb8a` |
| `DENSITY_RESULTS.csv` | `5f88448ce385c925117f3e2cf4dd8150e4c3351085a734b125b65658fa3f60c3` |
| `DENSITY_SUMMARY.csv` | `2f015629aa0458b07e3f327349f3e9f847b5cd77b987d318c0f7d2612802a828` |
| `DENSITY_THRESHOLD_FIT.json` | `da2d43202af28d0e300e6df4110837e75ad20b1bd2fa1ef27cc54b04f2fe0bd0` |
| `FINITE_SIZE_RESULTS.csv` | `2a7eb2eaa04854a851b9a36a5e867380a16979fc9a599c83031d82105aee8c5c` |
| `FINITE_SIZE_SUMMARY.csv` | `8de4895302dfd7be963b00bc001801c9886f37b925dc1ee63419fb8848e22ce3` |
| `FINITE_SIZE_COMPARISON.json` | `36637c82d084724d636e2f519a058de536ef05db66983fa5b4b4605a6ff07231` |

Figure hashes are machine-readable in `FIGURE_MANIFEST.json` and
`FINITE_SIZE_FIGURE_MANIFEST.json`; aggregate compact artifact indices are under
`artifacts/broad`, `artifacts/refinement`, `artifacts/final_1024`, and
`artifacts/finite_size`. Large checkpoints, snapshots, and raw trajectories remain under
`results/homogeneous_density_threshold_20260826/` outside Git.

## Final v0.1 recommendation

**C. THE DENSITY-DEPENDENT REVERSIBLE NULL-CONSERVING PHENOMENON IS SUFFICIENTLY UNUSUAL THAT IT
WARRANTS A SEPARATE DEDICATED STUDY BEFORE v0.2.**

The precise unanswered result is why low-k amplification becomes universal near 12% while the
strong, persistent amorphous-domain regime turns on only near 27.5%, and whether that second onset
remains sharp under longer horizons and a genuinely powered finite-size/winding analysis. That is
a mechanism question about the frozen rule, not a reason to resume rule search or redefine v0.1.
