# Experiment Protocol

## 1. Research target

The primary target is the paper’s Experiment A, Signed Life: a finite cellular or graph-rewrite model whose local states carry signed values and whose permitted local updates preserve a declared exact invariant. The paper asks whether persistent motifs, collisions, perturbation robustness, heredity, and selection can arise when the signed values affect dynamics rather than acting as inert labels.

The present model is a finite reversible cellular automaton. It does not implement literal variable-cardinality graph splitting. State-multiset-changing 2×2 reactions are the fixed-lattice analogue; a variable graph-rewrite engine is a later experiment.

## 2. Formal model

### Alphabet

| Code | State | Charge |
|---:|---|---:|
| 0 | empty neutral | 0 |
| 1 | positive | +1 |
| 2 | negative | -1 |
| 3 | bound/neutral excitation | 0 |

Two zero-charge states are essential. The invariant records less than the complete state, so zero does not collapse the model to an empty label.

### Space and update

The space is an even `H × W` torus. At phase 0, disjoint 2×2 blocks begin at even coordinates. At phase 1, the lattice is shifted by `(1,1)` and repartitioned. Alternating partitions permit information to move between prior block boundaries without a global neighbourhood count.

A block code is a base-4 integer in `0..255`. A rule `F` is a lookup table satisfying:

1. `F` is a permutation;
2. for all block codes `b`, `Q(F(b)) = Q(b)` exactly.

Thus every phase is a bijection, every full tick is a bijection, and total lattice charge is exactly conserved.

## 3. Predeclared hypotheses and criteria

### H1 — Exact local conservation

**Criterion:** exhaustive validation over all 256 blocks finds no charge-sector crossing and the table is a permutation.

**Failure:** implementation/rule invalid. Stop interpretation.

### H2 — Exact reversible dynamics

**Criterion:** a holdout state run forward for the declared number of phases and then under the inverse rule returns byte-for-byte to the initial state.

**Failure:** implementation defect or mismatch in phase history. Stop interpretation.

### H3 — Nontrivial signed interaction

**Criterion:** on holdout seeds, the selected rule has a positive local state-multiset reaction rate and exceeds both identity and pure block rotation. Its table must differ nontrivially from all simple baselines.

**Failure:** the invariant is dynamically inert in the selected witness.

### H4 — Organisation rather than stasis or random soup

No single scalar decides this. Inspect jointly:

- nonzero but nonsaturating activity;
- nonzero spatial mutual information;
- temporal mutual information and recurrence;
- block entropy below the maximally random regime but above trivial uniformity;
- perturbation distance that neither disappears immediately nor saturates the whole lattice;
- repeated local patches, treated only as a motif/replication proxy.

**Positive exploratory result:** selected rule outperforms identity/rotation and is not merely indistinguishable from a random-conserving control across all components.

**Failure:** report the rule as static, periodic texture, or chaotic soup. Do not rescue it by visual selection.

### H5 — Harmless-redescription covariance

**Criterion:** for declared conjugacies, transformed initial state plus conjugated rule equals the transformed original trajectory at every cell and frame. Metric discrepancies must be zero up to deterministic floating reductions.

Tested transformations:

- swap the two zero-charge labels;
- globally exchange positive and negative;
- translate coordinates by complete Margolus block periods.

**Failure:** metric or simulator depends on a detectable coding accident.

### H6 — Finite-window robustness

Candidate intensive statistics are measured under balanced, time-first, space-first, width-first, height-first, and shifted-centre window exhaustions. These vary order of expansion, rectangular shape, and sampled centre. Tail relative range below 5% is labelled stabilized for that candidate and sequence.

**Failure to stabilize is a result.** It disqualifies the candidate on this trace; it is not repaired with an infinite-limit assertion.

### H7 — Cross-level additive composition

**Criterion:** total charge at 2×2 and 4×4 levels equals microcharge exactly, and 4×4 direct coarse-graining equals composition through 2×2 blocks.

Macro-symbol conditional entropy and predictive accuracy are reported separately. They test approximate closure of a particular vocabulary, not metaphysical identity.

### H8 — Finite rootless-cycle witness

On a small torus, search for an exact repeated full-tick state. Because the map is a finite bijection, every state is on a cycle; finding the period is computationally contingent.

**Criterion:** if found, transient must be zero. A nonzero transient is an implementation error.

A cycle has no dynamically privileged first state. It does not instantiate an unbounded beginningless process and cannot establish strict potentialism.

## 4. Search and holdout separation

Evolution uses generation-specific training seeds. The final analysis uses seeds offset from every search seed. The selected rule is frozen before holdout metrics are calculated.

Search uses a mixture of:

- identity and block rotations;
- deterministic low-complexity reactive rotations;
- sparse mutations of simple rules;
- fully random charge-preserving permutations.

This avoids making the search cross an overwhelmingly chaotic rule space by tiny mutations alone.

## 5. Controls

| Control | Purpose |
|---|---|
| Identity | detects scores that reward static state diversity |
| 90° block rotation | detects kinematic movement without state-multiset reaction |
| Random conserving permutation | detects hot signed noise that satisfies the invariant |
| Random unconstrained permutation | confirms exact audits detect invariant failure |

## 6. Search disclosure

Every run archives:

- resolved configuration;
- hardware/software manifest;
- seed;
- population and generations;
- generation history;
- top rules, not only the winner;
- checkpoint rules;
- final holdout report.

The searched volume is the number of evaluated rule–replicate combinations, not a claim to cover the astronomical rule space.

## 7. Result classes

### Exact formal result

Examples: local charge preservation, bijectivity, inverse recovery, conjugacy, additive coarse-grain composition.

### Trace-level empirical result

Examples: activity, reaction rate, conditional entropy, cycle period under a cap, window stabilization.

### Proxy

Examples: mutual information, repeated patches, perturbation spread, search quality.

### Not addressed

Occurrence, physical instantiation, ETNE, consciousness, strict potentialism, and a unique process measure.
