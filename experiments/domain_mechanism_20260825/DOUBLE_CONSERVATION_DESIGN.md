# Double-Conservation Sibling Design

This document records the local construction before any double-conservation trajectory is run.
The scientific comparison remains frozen by `PREREGISTRATION.md`.

## Two independent additive invariants

For cell state `s`:

| State | Meaning | `q(s)` | `m(s)` |
|---:|---|---:|---:|
| 0 | empty | 0 | 0 |
| 1 | positive excitation | +1 | 1 |
| 2 | negative excitation | -1 | 1 |
| 3 | bound/neutral excitation | 0 | 1 |

For every 2×2 block, `Q = Σq(s)` and `M = Σm(s)`. `M` is independent of `Q`: for example,
four empty cells have `(Q,M)=(0,0)`, a positive-negative pair plus two empty cells has `(0,2)`,
and two bound excitations plus two empty cells also has `(0,2)` while differing in exact state
composition.

The new rule family consists only of permutations within joint `(Q,M)` sectors. Therefore each
local update preserves both integer sums exactly. Disjoint Margolus blocks then preserve global
`Q` and `M`; alternating the partition does not change this proof. Because every local table is a
permutation, the inverse has the same joint-sector property.

No global repair, tolerance, or distant compensation is used.

## Joint-sector sizes and remaining freedom

The 256 block codes form 25 nonempty joint sectors:

```text
(-4,4):1
(-3,3):4  (-3,4):4
(-2,2):6  (-2,3):12 (-2,4):10
(-1,1):4  (-1,2):12 (-1,3):24 (-1,4):16
( 0,0):1  ( 0,1):4  ( 0,2):18 ( 0,3):28 ( 0,4):19
(+1,1):4  (+1,2):12 (+1,3):24 (+1,4):16
(+2,2):6  (+2,3):12 (+2,4):10
(+3,3):4  (+3,4):4
(+4,4):1
```

The exact table count is the product of the factorials of these sizes, approximately
`10^199.8192446`. Charge-only sectors allowed approximately `10^317.9614652` tables. The second
invariant removes a very large amount of freedom without reducing the family to kinematics.

## Relationship to the frozen winner

The winner is a 90-degree block rotation with 12 image swaps, producing 24 reactive inputs. Exact
enumeration shows:

- the winner is a valid permutation and preserves `Q` at all 256 inputs;
- all 24 winner reactive entries change `M`;
- the other 232 entries are the occupancy-preserving rotational scaffold.

Thus no joint-conserving rule can retain any of the winner's 24 reactive input-output mappings.
The input support also cannot be retained wholesale: within several joint sectors represented by
the winner, `(Q,M)` already fixes the state histogram, so inputs can only be spatially permuted and
cannot be reactive with one another.

## Closest sibling optimization

To remain within the same 12-swap reactive-rotation architecture, the sibling is selected by an
exact binary integer program:

1. enumerate every pair of input codes in the same `(Q,M)` sector whose state histograms differ;
2. select exactly 12 pairs;
3. require pairs to be disjoint;
4. maximize how many of the 24 winner-support inputs are endpoints;
5. minimize the sum of deterministic candidate ranks to break equal-overlap ties.

The optimum overlaps only 3 of the winner's 24 reactive input locations. Consequently the sibling
differs from rotation at 24 inputs and from the winner at 45 inputs (`24 + 21`). This is the minimum
winner distance within the frozen 12-disjoint-swap architecture, not a claim of minimum distance
over every imaginable joint-sector permutation.

The result remains scientifically close in the defensible architectural senses: same four states,
same charge, same torus, same Margolus schedule, same 90-degree scaffold, same 12 swaps, same 24
reactive inputs, exact invertibility, and identical initial-condition family. It cannot preserve
the winner's particular reactions because those reactions are precisely the occupancy-changing
freedom under test.

## Matched controls and exhaustive verification

Eight matched controls are generated independently from frozen seeds. Each is the same rotation
plus 12 disjoint randomly ordered image swaps restricted to joint `(Q,M)` sectors and required to
change state histograms. Every sibling/control must pass:

- exact permutation over `0..255`;
- exhaustive `Q(input) == Q(output)` for 256 inputs;
- exhaustive `M(input) == M(output)` for 256 inputs;
- exactly 24 reactive inputs;
- exact inverse validation;
- forward/inverse trace recovery;
- an independent verifier that does not import ZWSLab's alphabet, rule, or audit implementation.

Hashes, generation seeds, tables, settings, and the eight primary comparison seeds are frozen in
`DOUBLE_CONSERVATION_MANIFEST.json` before evaluation.
