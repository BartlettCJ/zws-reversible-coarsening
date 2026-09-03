# Frozen Discovery Summary

## Classification

`results\observe_4096` is **discovery data**. It was inspected before the 2026-08-25 preregistration and must never be relabelled as confirmatory evidence.

## Frozen rule and run

- Rule name: `g0001_elite_000`
- Canonical rule-table digest: `d1fe99b5735d8e9be032347de2727dc5c52a7848df200ca26f67f28c8c1bc41c`
- Rule JSON file SHA-256: `6c2f21f7d08494ebb74c838e19503aae09ef58bddaa051d6d9a09dc36ab20f19`
- Valid permutation: yes
- Exact charge-sector preservation: yes
- Reactive lookup inputs: 24 of 256 (`reaction_fraction = 0.09375`)
- World: 4,096×4,096 torus
- Duration: 65,536 alternating Margolus phases
- Initial condition: localized, exactly charge-balanced seed
- Sampling: every 16 phases
- Full-state checkpoints: every 2,048 phases, 33 total
- Selected local event clips: 20
- Performance: 801.06 simulation seconds, 81.81 phases/s, 0.656 GiB peak CUDA allocation

## Discovery observations

- Occupied fraction increased from 6.472% to 11.596%.
- Positive and negative populations each increased from 407,155 to 653,292 while net charge remained exactly zero.
- Bound/neutral cells increased from 271,437 to 638,983.
- Event-level activity increased from about 0.1106 at tick 976 to 0.1328 at tick 64,336.
- Reaction density increased from about 0.0110 to 0.0148.
- Normalized four-state entropy increased from about 0.2584 to 0.3503.
- Boundary density increased from about 0.1111 to 0.1334.
- Late local states showed persistent macroscopic sparse/dense domains (“golf-course” morphology).
- A same-density shuffled reference gave 64×64 occupancy-block dispersion ratios of approximately 6.36 early and 56.10 late.
- Local exact-state agreement peaked strongly at lags 4/8/16; late exact agreement was about 0.9145 at lag 8 and 0.8696 at lag 128.
- Late 8×8-coarsened occupancy correlation remained about 0.9797 at lag 128.

These observations support an exploratory description of persistent domain formation/coarsening with a fast stroboscopic oscillator. They do not establish life, organisms, replication, intelligence, consciousness, open-ended evolution, primitive occurrence, ETNE, or FOA.

## Known discovery confounds

1. The selected rule came from a tiny smoke search rather than a deep rule search.
2. This was one rule and one initial-condition seed.
3. Charge is conserved, but particle number and an energy-like quantity are not; reversible occupancy creation may explain much of the phenomenon.
4. The event ranker mostly sampled a smooth activation trend rather than 20 independent discontinuities.
5. The explorer concatenates 385-frame local clips. Apparent jumps at clip boundaries are omitted-time splices, not one-phase global events.
6. The all-history 4,096² trajectory was intentionally not retained; exact periodic full checkpoints and reconstructed local clips were retained.
7. Matched same-family controls and independent untouched seeds had not yet been run.

## Integrity evidence

`discovery_integrity.json` verifies every one of the 33 full-world checkpoint payloads, the resume payload, all 20 exact local `.npy` clips, and the 2.02 GB explorer trajectory binary against their recorded SHA-256 values. All 55 checks passed over 4.354 GiB, and the recorded rule digest matches the frozen expected digest.

The compact evidence bundle is retained by the author and is not part of this release. Multi-gigabyte trajectories and checkpoint collections remain only under `results\observe_4096` and are excluded from Git/source snapshot artifacts.

