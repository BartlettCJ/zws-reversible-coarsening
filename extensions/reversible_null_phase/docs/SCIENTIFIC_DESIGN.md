# Scientific design

## Starting result

The frozen winner preserves signed charge

\[
Q=N_+-N_-=0
\]

while unsigned occupancy

\[
M=N_++N_-+N_{\mathrm{bound}}
\]

is free to change. Earlier interventions showed that the closest exact-`Q+M` sibling remains reactive but loses the large-domain regime. A homogeneous density sweep then found a sharp finite-size onset near 27.5% occupancy, preceded by low-k amplification near 12%.

## The central ambiguity

A sharp finite-time classifier can arise for several very different reasons:

1. **Linear instability:** an infinitesimal mode has positive growth above a density.
2. **Nucleation:** the regime exists below the apparent threshold but takes longer than the observation window to appear.
3. **Bistability:** homogeneous and organised states occupy different basins at the same exact inventory.
4. **Topology:** a connected near-vacuum network begins wrapping or changing genus.
5. **Reaction synergy:** several local occupancy-changing transpositions jointly create a feedback loop.
6. **Metric artefact:** the binary classifier crosses a threshold although continuous dynamics vary smoothly.

The stages are designed to separate these.

## Stage families

### Precursor

Measure the early log-growth rate of declared Fourier shells. A positive rate over the frozen fit window is evidence of an amplifying spectral mode. It is not yet evidence of persistent domains.

### Onset

Measure strong-domain pass probability and first-passage time over a denser occupancy grid and several sizes. The important quantity is not only whether a trajectory passes, but **when**.

### Horizon

Extend selected densities to 131,072 phases. If below-threshold pass probability rises substantially with horizon, the previous “threshold” was partly a waiting-time boundary.

### Hysteresis/basin dependence

Compare homogeneous, slab and droplet arrangements with the **same exact state inventory**. A seeded state that persists or grows where homogeneous states rarely nucleate demonstrates basin dependence. The word hysteresis is used operationally; no equilibrium free-energy interpretation is assumed.

### Reaction-pair ablation

The winner differs from its 90° rotation scaffold at 24 inputs forming twelve exact transpositions. Restore each pair separately and rerun matched conditions. This identifies necessary reactions and distributed synergy without searching a new rule.

## Topology

The torus analysis uses a 3×3 periodic tiling and exact four-neighbour labelling. A cluster wraps when a central-tile site is connected to its translated copy. Finite-domain scale excludes wrapping components, preventing one world-spanning vacuum from being mistaken for an enormous finite droplet.

## Primary order parameters

No universal scalar is declared. The package retains separately:

- finite empty-domain characteristic scale;
- pass probability and first-passage time;
- low-k power and shell growth rates;
- occupied/empty wrapping probabilities;
- largest finite and wrapping component fractions;
- occupancy and exact state composition;
- reaction fraction and predicted local \(\Delta M\) flux;
- state entropy and activity.

## Exactness

Every rule is exhaustively checked over all 256 local inputs. The winner and ablations must remain permutations and preserve exact local charge. The generated `Q+M` sibling additionally preserves exact local occupancy. Exact global charge is checked at every topological sample. Checkpoints and compact results carry SHA-256 sidecars.
