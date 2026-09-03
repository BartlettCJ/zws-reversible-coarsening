# ZWSLab v0.1.4 density-threshold Stage-2 refinement report

## Frozen selection

The full 168-trajectory broad grid completed before this refinement was selected. The frozen rule selected the primary_25_to_75_bracket interval from 0.260002136 to 0.300003052, then inserted targets at spacing 0.010. No density was selected from an individual seed outcome.

Refinement manifest SHA-256: `aec4862a401b92d74de54a93dd0bfbb4744ce1b97699ca44b3d68b68b1cc791e`.
Stage-2 addendum SHA-256: `dd5419405071c1f18ddaec7f76f4b3ed124a9947e46e85505e389b15ff0328fa`.

## Results

| Target occupancy | Realized occupancy | Strong domain | Low-k growth | Median peak L ratio | Median final L ratio |
|---:|---:|---:|---:|---:|---:|
| 0.2700 | 0.269996643 | 4/16 | 16/16 | 3.375 | 2.417 |
| 0.2900 | 0.290000916 | 16/16 | 16/16 | 7.103 | 5.348 |

Completed trajectories: 32/32; failures: 0. Exact Q=0 held in 32/32 runs.

These measurements are the preregistered refinement stage. The combined broad-plus-refinement classification, logistic onset estimate, continuous curves, and morphology are reported in `DENSITY_THRESHOLD_REPORT.md`; this document does not independently redefine the transition class.

## Integrity and runtime

The sum of per-arm recorded process intervals was 1424.0 seconds. All trajectory and snapshot hashes were verified by the campaign loader before analysis. Large raw trajectories remain outside git.
