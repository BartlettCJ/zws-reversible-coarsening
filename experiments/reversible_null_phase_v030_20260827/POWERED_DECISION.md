# Powered-profile decision

## Decision

**No-go for launching the frozen powered profile unchanged as the next run.** It was not launched.

The powered profile's scientific objectives remain worthwhile—especially 2048² finite-size coverage, more seeds, a 1024² precursor stage, and a Q+M precursor control—but the balanced result shows that its present density allocation is unlikely to spend the next large block of GPU time efficiently.

## Evidence

- Both balanced onset fits extrapolate below their sampled grids. At 512², the lowest density 0.2600 already passed 8/12; at 1024², the lowest density 0.2675 already passed 7/8. Neither sweep brackets 50%.
- At 1024² and density 0.2600, all 6/6 131,072-phase horizon trajectories passed, with median first passage 63,488. This makes lower-density and waiting-time coverage more urgent than denser sampling of the already-saturated upper window.
- The frozen powered profile contains 1,932 trajectories versus 684 balanced trajectories. Its onset minima are 0.2600 at 512², 0.2650 at 1024², and 0.2700 at 2048², so it may still fail to bracket the transition while adding substantially more computation.
- The balanced run used 4.742 wall-clock hours including the deliberate pause on the faster installed RTX 5060 Ti. Thermals were excellent (61 °C maximum) and there were no governor events, so hardware safety is not the blocker; scientific allocation is.

## Recommended next gate

Keep `rtx3060_powered.yaml` frozen. Before authorizing it, add a separate, explicitly versioned lower-density scout that brackets the 50% response at 512² and 1024² and probes whether the 2048² window also needs to move downward. Use those results to decide whether to run the full powered matrix unchanged, supplement it, or replace it in a later version. Do not resume rule search and do not alter the frozen winner.
