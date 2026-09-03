# Architecture

## Search data flow

```text
resolved config + exact seed
   |
   v
initial rule population -- 256-entry charge-sector permutations
   |
   +--> candidate chunk [rules × replicates]
   |       |
   |       +--> exact Q=0 initial states and charge-preserving perturbations
   |       +--> PyTorch Margolus engine (CUDA or CPU)
   |       +--> separate behavioural metrics + representative final state
   |       |
   |       v
   |    CPU partial-generation arrays
   |       |
   |       +--> atomic checksummed checkpoint
   |       +--> file controls / signal latch
   |       +--> CPU/GPU/RAM resource governor
   |       +--> sleep or workload hibernation/restart
   |
   v
complete-generation novelty + ranking
   |
   +--> evolutionary selection/mutation
   +--> search history
   +--> score/novelty/metric champion archive
   |
   v
frozen ranked rules
```

The checkpoint contains the population, generation, next candidate, completed metric arrays, representative exact states, history, Python random state, PyTorch generator state, champion index, and configuration hash. CUDA tensors are not serialized mid-kernel.

## Holdout-analysis flow

```text
frozen top rule + independent holdout seeds
   |
   +--> exhaustive local audit
   +--> inverse trace audit
   +--> CPU/GPU parity
   +--> controls
   +--> exact saved trajectory
   +--> state/coordinate conjugacies
   +--> finite-window exhaustions
   +--> typed coarse-graining
   +--> exact cycle search
   +--> automated visual-review bookmarks
   +--> Eshkol oracle export
   |
   v
JSON + Markdown + HTML + PNG/GIF + exact states
   |
   +--> checksummed uint8 trajectory bundle
   +--> local canvas explorer (zoom/pan/playback/inspection/capture)
```

Analysis writes a stage checkpoint before each expensive phase. A managed worker that hibernates during analysis resumes from completed stages and the stored trajectory.

## Why PyTorch is primary

The NVIDIA GPU is most useful for evolving many lattices under many rule tables simultaneously. The workload is integer lookup, reshape, gather, roll, histogram, and reduction. PyTorch provides a mature CUDA path and keeps the simulator concise enough to audit.

Candidate chunking is a scheduling boundary, not a change to the evaluated population: each rule sees the same deterministic replicate seeds as in an unchunked generation. The exact-resume regression test compares an interrupted run with an uninterrupted run byte-for-byte at the winning rule and generation-history level.

## Why Eshkol is secondary

Eshkol is a cross-language exact local oracle because its exact integer arithmetic and a 256-case exhaustive validator are easy to inspect. The discrete rule-permutation family has no useful differentiable parameterization, so forcing the GPU search through Eshkol would add integration risk without exploiting its principal automatic-differentiation strength.

## Resource separation

The worker owns CUDA execution. The managed supervisor does not initialize a CUDA context; it watches the worker exit code and resource samples. On sustained external pressure the worker checkpoints and exits, releasing its CUDA context and most RAM. The supervisor waits and starts a fresh worker with `--resume`.

No code path invokes Windows sleep, suspend, shutdown, or system hibernation.

## State and rule schema

A rule JSON has:

```json
{
  "schema": "zws-nullity-rule-v1",
  "name": "...",
  "alphabet": {
    "states": ["empty", "positive", "negative", "bound"],
    "charges": [0, 1, -1, 0]
  },
  "table": [256 integers],
  "metadata": {},
  "valid": true,
  "reaction_fraction": 0.0
}
```

The lookup table is serialized in input-code order and is sufficient to reproduce every local transition.
