# Confirmatory and Statistical Plan

## Exploratory stage

The evolutionary search is exploratory. It evaluates many rules, uses a composite objective, and returns ranked candidates. Search-generation metrics cannot be treated as unbiased estimates of a winner’s general behaviour.

## Frozen-rule holdout stage

Before holdout evaluation:

1. freeze the selected rule table;
2. freeze metric definitions and thresholds;
3. generate holdout seeds from the separately declared seed range;
4. evaluate selected and control rules on the same holdout lattices and paired perturbations.

`analysis.holdout_replicates` is:

- 2 for smoke plumbing only;
- 8 for the first workstation run;
- 16 for deep confirmation.

The JSON retains every per-replicate metric, not only mean and standard deviation.

## Paired comparison

For each control, the report calculates the fraction of holdout replicates on which the selected rule has a higher declared search-quality score. This is a diagnostic, not a null-hypothesis test and not a biological effect size.

Substantive interpretation should focus on componentwise paired differences:

- selected reaction minus identity/rotation reaction;
- selected spatial mutual information minus random-conserving control;
- selected temporal mutual information minus random-conserving control;
- perturbation distance relative to both disappearance and saturation;
- exact tests, which are pass/fail and need no sampling inference.

## Confirmatory rerun

For a result intended for a paper supplement:

1. choose one rule from the initial search using the archived protocol;
2. publish/hash the rule before new evaluation;
3. create a confirmatory config with at least 32 holdout seeds and at least three lattice sizes;
4. run without further search or threshold edits;
5. report all seeds and sizes;
6. treat any later metric change as a new exploratory version.

## Multiple metrics

No familywise p-value is claimed. The metrics are a diagnostic profile with known overlaps. A rule should not be promoted because one of many metrics happens to be extreme.

## Exact tests

The following are exhaustive/deterministic and should be reported separately from sampled behaviour:

- 256-block permutation and charge audit;
- inverse table correctness;
- forward/reverse trace equality for each tested trace;
- state-conjugacy equality;
- additive coarse-grain composition;
- exact state equality in cycle detection.
