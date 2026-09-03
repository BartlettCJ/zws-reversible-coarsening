# Gated Finite-Size Addendum — Frozen Location/Sharpening Check

**Frozen after the complete broad and refinement campaigns and the combined final 1024 analysis,
and before any finite-size-check initial state evolves.** This is the preregistered small
location/sharpening check, not a finite-size-scaling study and not a thermodynamic-limit test.

## Gate and deterministic target selection

- Final 1024 decision SHA-256:
  `e139ff91edd4bcd5b96eb9a81c6459d6062fab8be0d89f7789d9b4898b56e36b`.
- Final analysis artifact-index SHA-256:
  `0b74f2401e2764a768ade48a438b4c361cb5939346bf5c60d20c47e7db4d4272`.
- Frozen logistic-fit SHA-256:
  `da2d43202af28d0e300e6df4110837e75ad20b1bd2fa1ef27cc54b04f2fe0bd0`.
- Combined 1024 evidence: 200/200 trajectories complete, zero failures, exact `Q=0` in every
  trajectory, and no severe pass-fraction reversal.
- Frozen response class: `D_VERY_SHARP_FINITE_SIZE_TRANSITION_REGION`.
- The monotonic logistic gate passed (Spearman 0.8668740325072; required >= 0.8).
- Descriptive logistic point estimates are p25 = 0.26810208185874823, p50 =
  0.27472299929128063, p75 = 0.28134391672381304, and p25-to-p75 width =
  0.013241834865064805. The 2,000/2,000 successful blocked bootstrap refits give a 95% width
  interval of [0.010480740675156638, 0.016284535479733995].
- The preregistered gate requires class C/D, a valid monotonic fit, and width <= 0.06; all three
  conditions pass.
- Applying the frozen deterministic four-decimal round-half-up selector gives exactly **0.2681,
  0.2747, and 0.2813**. No target or seed was selected from a finite-size outcome.

## Frozen run design

- Worlds: 512 x 512, 1024 x 1024, and 2048 x 2048 tori. No 4096 run is permitted.
- Duration: 32,768 microscopic Margolus phases for every trajectory.
- Sample/checkpoint interval: 256 phases; early snapshot at phase 4,096; metric-peak and final
  snapshots retained.
- Seeds per density: 8 at 512, 8 at 1024, and 4 at 2048.
- Planned trajectories: 24 at 512, 24 at 1024, and 12 at 2048; 60 total.
- The same arrangement seed is blocked across all three densities within each world and seed
  index, yielding nested occupied supports. World-specific arrangements are independent.
- Winner, exact-inventory construction, exact-Q audit, diagnostics, strong-domain criterion,
  low-k criterion, and analysis definitions remain unchanged.
- Exact inventory rule: `u=floor(p*N/8+0.5)` and counts
  `(empty, positive, negative, bound)=(N-8u,3u,3u,2u)`.
- Winner table SHA-256:
  `d1fe99b5735d8e9be032347de2727dc5c52a7848df200ca26f67f28c8c1bc41c`.

## Frozen exact inventories

| World | Target | Realized occupancy | Empty | Positive | Negative | Bound | Occupied | Q |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 512 | 0.2681 | 0.268096923828125 | 191864 | 26355 | 26355 | 17570 | 70280 | 0 |
| 512 | 0.2747 | 0.274688720703125 | 190136 | 27003 | 27003 | 18002 | 72008 | 0 |
| 512 | 0.2813 | 0.28131103515625 | 188400 | 27654 | 27654 | 18436 | 73744 | 0 |
| 1024 | 0.2681 | 0.268096923828125 | 767456 | 105420 | 105420 | 70280 | 281120 | 0 |
| 1024 | 0.2747 | 0.27469635009765625 | 760536 | 108015 | 108015 | 72010 | 288040 | 0 |
| 1024 | 0.2813 | 0.28130340576171875 | 753608 | 110613 | 110613 | 73742 | 294968 | 0 |
| 2048 | 0.2681 | 0.2681007385253906 | 3069808 | 421686 | 421686 | 281124 | 1124496 | 0 |
| 2048 | 0.2747 | 0.2747001647949219 | 3042128 | 432066 | 432066 | 288044 | 1152176 | 0 |
| 2048 | 0.2813 | 0.2812995910644531 | 3014448 | 442446 | 442446 | 294964 | 1179856 | 0 |

## Frozen seed namespaces and identities

Base namespace:

`ZWSLab-density-threshold-finite-size-v1:e139ff91edd4bcd5b96eb9a81c6459d6062fab8be0d89f7789d9b4898b56e36b:0.2681,0.2747,0.2813`

Each world's arrangement and diagnostic namespace appends `:world512`, `:world1024`, or
`:world2048`. The shared inventory identity and world-specific arrangement/diagnostic triplets
are frozen below.

### 512 x 512

| Index | Inventory seed | Arrangement seed | Diagnostic seed |
|---:|---:|---:|---:|
| 00 | 503651774 | 1212017389 | 591408275 |
| 01 | 1542097456 | 1836965201 | 168793124 |
| 02 | 220605579 | 392422155 | 2006265395 |
| 03 | 1707946189 | 459721562 | 1511875248 |
| 04 | 728137290 | 244213078 | 72376587 |
| 05 | 2094936405 | 987387861 | 1796879953 |
| 06 | 1072085560 | 1019839612 | 88640616 |
| 07 | 212166987 | 1274886855 | 507652316 |

### 1024 x 1024

| Index | Inventory seed | Arrangement seed | Diagnostic seed |
|---:|---:|---:|---:|
| 00 | 503651774 | 192846197 | 978805476 |
| 01 | 1542097456 | 607769375 | 1870021722 |
| 02 | 220605579 | 128564605 | 490466725 |
| 03 | 1707946189 | 13432971 | 1328134725 |
| 04 | 728137290 | 1793211905 | 397241628 |
| 05 | 2094936405 | 286176727 | 907618215 |
| 06 | 1072085560 | 878770046 | 1344985527 |
| 07 | 212166987 | 1732407134 | 1470950696 |

### 2048 x 2048

| Index | Inventory seed | Arrangement seed | Diagnostic seed |
|---:|---:|---:|---:|
| 00 | 503651774 | 834674248 | 1837347820 |
| 01 | 1542097456 | 1653895994 | 214429879 |
| 02 | 220605579 | 1343851709 | 2010707532 |
| 03 | 1707946189 | 1080890518 | 531678724 |

## Analysis and inspection boundary

All 60 trajectories and all three world stages must complete and pass manifest, checkpoint,
snapshot, and trajectory-hash checks before any finite-size outcome is read. Each world reports
independent strong-domain and low-k pass counts at all three densities. A per-world descriptive
logistic fit runs only where the unchanged monotonic gate passes; it uses the frozen 2,000 blocked
bootstrap replicates. Comparisons are limited to whether fitted onset location shifts strongly and
whether fitted width appears to sharpen. Three target densities and these small seed counts do not
support scaling exponents or a thermodynamic phase-transition claim.

The generated machine-readable manifest and audit must freeze every initial-state hash and verify
exact regeneration, nested supports, exact inventories, and `Q=0`. This addendum, the manifest,
the audit, completed refinement artifacts, and combined 1024 evidence must be committed and pushed
before the finite-size campaign starts.
