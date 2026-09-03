# ZWSLab v0.1.4 Overnight Independent Confirmation

## Bottom line

**Classification D: The winner did not reproduce the full sustained-coarsening criterion in at least three of four Stage 7B seeds.**

This strict classification does not mean that the large domains vanished: all four 4,096² winner
seeds produced strong finite-domain growth. Their characteristic scale peaked near the midpoint
and then relaxed, so they failed the registered requirement for a positive late-half slope.

The original `observe_4096` trajectory remains discovery evidence. All rows below are
confirmatory trajectories from seeds and rules frozen in commit `750deffe1ce466947f65061b3651efc22d58a236`
before outcomes. The discovery rule is unchanged at
`d1fe99b5735d8e9be032347de2727dc5c52a7848df200ca26f67f28c8c1bc41c`.

## Discovery data (not confirmatory)

The original 4,096² × 65,536-phase run showed localized sparse/dense morphology, increasing
occupancy, and strong lag-4/8 recurrence. Its exact inverse recovered the initial 16,777,216-cell
state byte-for-byte after all 65,536 phases, with zero mismatches at every retained checkpoint
boundary. These observations motivated, but are not counted in, the matrix below.

## Confirmatory matrix

| Stage | Condition | Coarsening criterion | Median final dispersion | Median finite-scale ratio | Late recurrence 4 | Late recurrence 8 | Charge violations |
|---|---|---:|---:|---:|---:|---:|---:|
| stage7a | identity | 0/8 | 1057.27 | 1 | 1 | 1 | 0 |
| stage7a | matched_00 | 0/8 | 2692.17 | 0.849262 | 0.953222 | 0.95819 | 0 |
| stage7a | matched_01 | 0/8 | 1.00705 | 0.788091 | 0.767756 | 0.786698 | 0 |
| stage7a | matched_02 | 0/8 | 1.22487 | 0.81119 | 0.757466 | 0.765395 | 0 |
| stage7a | matched_03 | 0/8 | 0.983244 | 0.788225 | 0.760424 | 0.774352 | 0 |
| stage7a | matched_04 | 0/8 | 0.97346 | 0.788851 | 0.773681 | 0.796814 | 0 |
| stage7a | matched_05 | 0/8 | 0.9958 | 0.788532 | 0.742097 | 0.748568 | 0 |
| stage7a | matched_06 | 0/8 | 0.968303 | 0.788575 | 0.734794 | 0.731548 | 0 |
| stage7a | matched_07 | 0/8 | 0.982067 | 0.789661 | 0.737753 | 0.744977 | 0 |
| stage7a | rotation | 0/8 | 1057.27 | 1 | 1 | 1 | 0 |
| stage7a | winner | 8/8 | 1779.43 | 27.2034 | 0.979094 | 0.98589 | 0 |
| stage7b | matched_00 | 0/4 | 2796.03 | 0.786096 | 0.951956 | 0.956815 | 0 |
| stage7b | matched_01 | 0/4 | 1.01265 | 0.785885 | 0.767834 | 0.786782 | 0 |
| stage7b | matched_02 | 0/4 | 1.34995 | 0.828307 | 0.755827 | 0.763939 | 0 |
| stage7b | matched_03 | 0/4 | 1.0086 | 0.786563 | 0.761018 | 0.774846 | 0 |
| stage7b | rotation | 0/4 | 1073.03 | 1 | 1 | 1 | 0 |
| stage7b | winner | 0/4 | 2057.43 | 16.1118 | 0.977148 | 0.984118 | 0 |

All 112 / 112 planned
trajectory records passed the frozen seed, rule, sample-count, endpoint, manifest, paired-initial-
state, and exact-charge checks: **True**. Failed runs: **0**.

Paired winner-minus-control effect sizes and deterministic 20,000-draw percentile-bootstrap 95%
intervals are in `CONFIRMATION_EFFECTS.csv`. With only 8 or 4 paired seeds, these intervals are
descriptive and should not be treated as high-powered population estimates.

## Explicit scientific questions

### 1. Does the winner reproducibly form large domains?

Yes: all 4/4 Stage 7B winner seeds retained both large dispersion and at least twofold finite-domain growth; the median final scale ratio was 16.1118. The stricter sustained-coarsening count was 0/4 because the late slope was negative.

### 2. Does pure rotation do the same thing?

Pure rotation counts were 0/8 and 0/4; equal-explanation status at 4096² was False.

### 3. Is the winner unusual among matched rules?

Winner final-dispersion rank was 2/9 against the eight Stage 7A matched controls and 2/5 against the four Stage 7B controls, because matched_00 had still greater coarse dispersion. However, the winner's median final finite-domain scale ratio was 16.1118 while every Stage 7B matched-control median was below 1. The winner was unusual in domain growth but not uniquely largest in coarse dispersion. Strict classification is D: The winner did not reproduce the full sustained-coarsening criterion in at least three of four Stage 7B seeds.

### 4. Is the 4/8-tick oscillator scaffold-inherited?

Yes, largely: Stage 7B median late lag-4/8 recurrence was 0.977148/0.984118 for the winner and 1/1 for pure rotation. Recurrence shared with rotation is scaffold-inherited, not an independent signature of organization.

### 5. Does domain length grow with time?

Early-to-mid-time growth was reproducible and strong (median peak scale ratio 98.4459), but it was not sustained: the scale relaxed to a median 16.1118 times initial and every Stage 7B winner late-half slope was negative. This supports domain formation followed by saturation/relaxation, not unbounded or sustained late-time coarsening.

### 6. Can reversible occupancy creation explain it?

The Stage 7B winner median occupied-fraction change was 0.050987 while every charge audit stayed exact, and the discovery run reversed byte-for-byte. The observations are consistent with reversible balanced occupancy creation, but this campaign does not identify a unique causal fraction.

### 7. Is there evidence beyond phase separation?

No. There is no lineage, heredity, reproduction, adaptive fitness, or organism test.

## Interpretation limits

The campaign tests reversible null-conserving spatial domain formation. It does not test or
establish life, organisms, replication, heredity, intelligence, consciousness, open-ended
evolution, FOA, primitive occurrence, ETNE, cosmology, or a fundamental process measure.
Coarse-density dispersion alone is insufficient: a rule can make a static localized density
contrast without growth of finite empty-domain length.

## Machine-readable artifacts

- `CONFIRMATION_SUMMARY.json`: complete summaries, seeds, hashes, gate evaluations, effects, answers.
- `CONFIRMATION_TRAJECTORIES.csv`: one row per confirmatory trajectory.
- `CONFIRMATION_EFFECTS.csv`: paired contrasts and uncertainty intervals.
- `CONFIRMATION_INTEGRITY.json`: frozen-matrix audit.
- `reports/phase_separation_confirmation_20260825`: paired curves and Stage 7A/7B reports.
