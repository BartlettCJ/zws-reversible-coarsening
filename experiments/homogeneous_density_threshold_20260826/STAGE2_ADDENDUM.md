# Stage-2 Addendum — Frozen Density-Threshold Refinement

**Frozen after the complete 168-trajectory broad campaign and its frozen analysis, and before
any refinement initial state evolves.** Stage 2 changes neither the frozen winner nor any
measurement, trajectory classification, response classification, or fit definition.

## Broad-stage gate and deterministic selection

- Broad manifest SHA-256:
  `9453e0b02415d2b9ac5fda68792427aa4dc86ac1b3bf503297161a21ab1d0ff1`.
- Broad decision SHA-256:
  `e6d247363d38351b5b5b9b322957be95f10f99580305df05d9ef8b75dabc25a1`.
- Broad density-summary SHA-256:
  `a542872d7bd31880d41ebaefff74d14c5050460e89db3fe62639c8f525c75e75`.
- All 14 broad arms, 168 trajectories, 14 checkpoints, and 56 snapshots completed; the
  campaign-wide post-run audit verified 271/271 completion and file-hash checks before outcome
  inspection.
- The frozen broad analyzer classified the response as
  `C_SHARP_FINITE_SIZE_TRANSITION_REGION` with no severe reversal.
- Its narrowest eligible 25%-to-75% bracket runs from target occupancy 0.26 (1/12 strong-domain
  passes; realized 0.26000213623046875) to target occupancy 0.30 (12/12 passes; realized
  0.3000030517578125), width 0.04000091552734375.
- Because the selected width exceeds 0.03, the preregistered spacing is 0.01. Arithmetic interior
  targets are 0.27, 0.28, and 0.29; 0.28 is already in the broad grid and is excluded by the
  frozen rule. Therefore the complete Stage-2 target set is exactly **0.27 and 0.29**.
- No density or seed was selected from an individual trajectory outcome.

## Frozen run design

- World: 1024 x 1024 torus.
- Duration: 32,768 microscopic Margolus phases.
- Sample/checkpoint interval: 256 phases; early snapshot at phase 4,096; metric-peak and final
  snapshots retained.
- Trajectories: 16 new blocked seed identities at each density, 32 total.
- The same arrangement seed is reused across both densities at each seed index, producing nested
  occupied supports. The same diagnostic seed identity is also blocked across both densities.
- Exact inventory rule remains `u=floor(p*N/8+0.5)` with counts
  `(empty, positive, negative, bound)=(N-8u,3u,3u,2u)`, so exact global `Q=0` and occupied-state
  composition 3/8, 3/8, 2/8 are preserved.
- Winner table SHA-256 remains
  `d1fe99b5735d8e9be032347de2727dc5c52a7848df200ca26f67f28c8c1bc41c`.

### Frozen exact inventories

| Target | Realized occupancy | Empty | Positive | Negative | Bound-neutral | Occupied | Q |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.27 | 0.26999664306640625 | 765464 | 106167 | 106167 | 70778 | 283112 | 0 |
| 0.29 | 0.29000091552734375 | 744488 | 114033 | 114033 | 76022 | 304088 | 0 |

### Frozen seed namespace and identities

Namespace:

`ZWSLab-density-threshold-refinement-v1:9453e0b02415d2b9ac5fda68792427aa4dc86ac1b3bf503297161a21ab1d0ff1:0.270000,0.290000`

Seeds are the deterministic SHA-256 derivations documented in the parent preregistration.

| Index | Inventory seed | Arrangement seed | Diagnostic seed |
|---:|---:|---:|---:|
| 00 | 538400382 | 853295840 | 1955884548 |
| 01 | 196753363 | 2087727786 | 173479063 |
| 02 | 1252314677 | 624896509 | 1723127438 |
| 03 | 2096493550 | 742512652 | 1959046485 |
| 04 | 182889977 | 1323421504 | 1619713968 |
| 05 | 1206547360 | 570634011 | 26440245 |
| 06 | 848987510 | 226793050 | 2061736050 |
| 07 | 1487073699 | 894146515 | 1661851754 |
| 08 | 1326416507 | 528394660 | 279927727 |
| 09 | 870666264 | 740528704 | 84846190 |
| 10 | 1352056701 | 1569707882 | 1046079936 |
| 11 | 1814221666 | 1986125183 | 1522288860 |
| 12 | 177166605 | 1595198365 | 1491660976 |
| 13 | 224243627 | 1830888421 | 163921243 |
| 14 | 1384049224 | 1514164260 | 1418075212 |
| 15 | 498601118 | 273610563 | 1513715837 |

## Frozen analysis and inspection policy

The exact strong-domain and systematic-low-k-growth criteria remain those in
`PREREGISTRATION.md`. Every refinement trajectory is retained. No refinement outcome may be read
until all 32 trajectories complete and all completion hashes verify. The combined broad and
refinement response is then classified with the original A–F definitions. A descriptive logistic
MAP fit and 2,000 blocked bootstrap replicates run only if the preregistered monotonicity gate
passes. The optional 512/1024/2048 check remains separately gated and requires another frozen
addendum and manifest before any such run. No 4096 world is permitted.

The machine-readable refinement manifest and input audit generated from this addendum must freeze
all 32 initial-state hashes and pass exact regeneration, nesting, inventory, and `Q=0` checks.
This addendum, that manifest, and that audit must be committed and pushed before evolution begins.
