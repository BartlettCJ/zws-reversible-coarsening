# ZWSLab Stage 7A Confirmation Report

## Outcome

**Frozen continuation decision: PROCEED to Stage 7B.**

- Winner qualifying seeds: **8/8** (required 6/8).
- Pure-rotation qualifying seeds: **0/8**.
- Pure rotation explains the result equally well: **False**.
- Outcome classification: **E — The winner passed the continuation gate, but its position within the matched-rule distribution was mixed rather than clearly exceptional or generic.**
- Failed or incomplete planned trajectories: **0**.

## What was tested

This is the preregistered confirmation of whether frozen rule `d1fe99...` reproducibly produces
macroscopic domain formation/coarsening from the localized balanced initial-condition family,
compared with pure rotation, identity, and eight independently matched 24-reactive-input controls.
All 88 planned Stage 7A trajectories used the eight seed values frozen before outcomes.

The finite empty-domain scale is `sqrt(sum(area_i^2)/sum(area_i)) across exact four-neighbour toroidal empty components after excluding the largest background component`. The largest empty
component is reported but excluded from that scale so the initial exterior of a localized seed is
not relabelled as a newly formed finite domain.

## Matched-control distribution

| Condition | Qualifying seeds | Median final dispersion ratio (64×64) |
|---|---:|---:|
| matched_00 | 0/8 | 2692.166406905383 |
| matched_01 | 0/8 | 1.0070467328495842 |
| matched_02 | 0/8 | 1.224874810740443 |
| matched_03 | 0/8 | 0.9832438640848931 |
| matched_04 | 0/8 | 0.9734601674785992 |
| matched_05 | 0/8 | 0.9958001765604422 |
| matched_06 | 0/8 | 0.9683033115712496 |
| matched_07 | 0/8 | 0.9820669827682389 |

## Interpretation boundary

This experiment can support or weaken a claim about reversible null-conserving spatial domain
formation. It does **not** test life, organisms, replication, heredity, intelligence,
consciousness, FOA, primitive occurrence, ETNE, cosmology, or a fundamental process measure.
High recurrence can be inherited from the rotational scaffold; occupancy can grow while exact
signed charge remains zero because empty and bound states both carry zero charge.

## Artifacts

- Gate record: `STAGE7A_GATE.json` (`e7a371e77363ef23c524947f1dc02f90aa958951ed65d39bf5d05c7debdee55e`)
- Manifest: `<local>/ZWSLab\experiments\phase_separation_confirmation_20260825\MANIFEST.json` (`2350e6880f96b42186bcd3b0dfba1efd811366afeb288c5b1f84fbd23e7f773a`)
- Complete trajectory records read: `88`
- Plots: `7` paired-seed curve figures with SHA-256 values in `REPORT_INDEX.json`.
