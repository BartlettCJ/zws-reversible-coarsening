# Existing-Trajectory Spatial Findings

This note records the interpretation made **before** evaluating either frozen new-rule screen.
It separates direct measurement, formal inference, and interpretation. Complete row-level results,
thresholds, motif tables, neighbourhood tables, input hashes, and figure hashes are in this
directory and `SUMMARY.json`.

## Exact state semantics

| Code | Name | Signed charge `Q` | Occupancy `M` |
|---:|---|---:|---:|
| 0 | empty | 0 | 0 |
| 1 | positive excitation | +1 | 1 |
| 2 | negative excitation | -1 | 1 |
| 3 | bound/neutral excitation | 0 | 1 |

The bound state is independently occupied even though its signed charge is zero. Therefore
`Q = N_positive - N_negative` and `M = N_positive + N_negative + N_bound` are not disguised copies.

## Measurement

- Spatial states were reconstructed for 93/93 preregistered milestones: all 33 discovery
  checkpoints, 40 Stage 7A milestones, and 20 Stage 7B milestones. Every retained state matched
  its frozen SHA-256 digest, and all three reconstructed trajectories returned exactly to their
  initial state under reverse evolution.
- Dense/sparse contrasts were repeated at box widths 16, 32, and 64 with 10/90, 20/80, and 30/70
  density quantiles. All 72 stage/role/field aggregates in `CONTRAST_ROBUSTNESS.csv` had a positive
  dense-minus-sparse contrast under every estimable definition. This includes positive, negative,
  bound/neutral, charged, occupied, and normalized exact-state entropy fractions.
- For the four 4096-square Stage 7B winners, the median dense-interior composition under the
  primary preregistered definition (32-cell density window, 20/80 quantiles, midpoint phase
  threshold) changed as follows:

  | Role | positive | negative | bound/neutral | occupied | normalized entropy |
  |---|---:|---:|---:|---:|---:|
  | initial | 0.120019 | 0.120029 | 0.080021 | 0.320055 | 0.702071 |
  | scale peak | 0.163663 | 0.163688 | 0.150425 | 0.477776 | 0.877644 |
  | final | 0.192542 | 0.192535 | 0.187912 | 0.572989 | 0.946343 |

  The matching primary sparse interiors had median zero occupancy at all three roles. Primary
  interfaces were intermediate but closer to the sparse phase: median occupied fraction was
  0.00161 initially, 0.02389 at the scale peak, and 0.04051 finally. Interface composition is more
  threshold-sensitive than the sign of the dense/sparse contrast; all alternative masks remain in
  the machine-readable tables.
- Positive and negative fractions rise together. In the Stage 7B dense interiors they are equal to
  within about `3e-5` at the displayed precision, while the median signed charge per site remains
  near zero (`4.0e-6` initially, `-2.7e-5` at peak, and `7.7e-6` finally). The dense phase is not
  dominated by either charge sign or by the bound state alone.
- Dense 16-by-16 blocks can nevertheless be locally charged. Across the four Stage 7B runs, the
  median of the mean absolute block charge was 6.23 initially, 6.55 at peak, and 6.97 finally;
  the corresponding signed-charge standard deviations were 7.82, 8.26, and 8.85. Sparse blocks
  were identically empty under the primary masks. Thus local charge fluctuations coexist with
  exact global charge balance.
- One-phase outgoing dynamics at each Stage 7B scale peak were concentrated in dense interiors:
  median reaction-cell fraction was 0.06333 there, 0.00268 at interfaces, and 0 in sparse
  interiors. Under the fixed pre-update masks, the median dense-interior empty-state count changed
  by -10,317 in that phase, while positive, negative, and bound counts changed by +1,273, +1,278,
  and +7,946.5 respectively. These are Eulerian count changes, not tracked particle fluxes.
- The discovery macro-arrow reconstruction verified byte-for-byte equality at all 33 retained
  reverse boundaries. Occupancy, exact-state entropy, finite-domain scale, and the state fractions
  retrace the same states in reverse order, while exact signed charge remains zero. The apparently
  asymmetric coarse history therefore coexists with exact microscopic reversibility.
- Signed curvature is not promoted: the discrete masks change enough with spatial resolution to
  make such an estimate unstable. Interface length, thickness, and roughness remain descriptive
  mask statistics rather than a claimed continuum surface tension.

## Formal inference

The separating order parameter is excitation occupancy/composition, not signed charge. The dense
phase is a high-entropy mixture of positive, negative, and neutral bound excitations; the sparse
phase is effectively the empty state. Exact `Q=0` constrains the two charged populations to balance
globally but leaves the independent `M` direction free. Local reactions exploit that direction by
converting empty sites into balanced charged and bound excitation content, predominantly inside the
already occupied phase.

This establishes a concrete candidate engine for the morphology: spatially localized reversible
excitation creation coupled to the rotational transport scaffold. It does **not** establish that
occupancy freedom is sufficient, uniquely causal, or thermodynamically irreversible. Those claims
require the preregistered exact-`Q`+exact-`M` sibling comparison.

## Interpretation

The most specific conservative description is **reversible phase organisation between a mixed,
globally charge-balanced excitation-rich phase and near-vacuum**. The large domains are not objects
made from a single state and are not positive/negative charge domains. The visual “golf-course”
regions are principally low-excitation or empty territory carved beside an increasingly mixed
excitation-supporting phase.

The coarse arrow is consistent with a special low-occupancy localized initial ensemble moving into
a much larger, higher-occupancy region of its exactly reversible orbit under coarse projection. It
is not by itself a thermodynamic arrow, self-maintenance, heredity, replication, or life. The new
double-conservation screen is the decisive test of whether removing the free occupancy direction
suppresses the organisation.
