# Metrics and Their Limits

## Exact quantities

### Total signed charge

`Q(S) = count(positive) - count(negative)` using integer arithmetic. Bound and empty cells both carry zero but remain distinct states.

### Local reaction

A block update is reactive when the four-state histogram of the output differs from the input histogram. Pure spatial permutations are nonreactive even when every cell moves.

### Table distance

The normalized Hamming distance from identity and the three block rotations estimates rule-description complexity relative to declared simple baselines. It is not Kolmogorov complexity.

## Normalized descriptive metrics

### State entropy

Shannon entropy of the four-state frequency distribution, divided by `log(4)`. Static random initial states can score highly; therefore it is never interpreted alone.

### Block entropy

Entropy of 2×2 block codes divided by `log(256)`. High values can mean rich local variation or random soup.

### Spatial mutual information

Average mutual information between a cell and its right/down toroidal neighbours, divided by `log(4)`. It detects local dependence, not function or agency.

### Temporal mutual information

Mutual information between cell state before and after one phase. Identity scores highly, so the dynamics gate and reaction control are mandatory.

### Recurrence

Fraction of cells matching a sampled frame two sample intervals earlier. This is a soft periodicity statistic, not an exact cycle detector.

### Perturbation distance

Hamming distance between an original and a minimally charge-preserving perturbed trajectory. Zero suggests insensitivity; near one suggests saturation. Neither alone is desirable.

### Patch repetition

Fraction of nonempty toroidal 4×4 patch instances whose exact code occurs at least twice in a frame. This is explicitly a **replication proxy**. Repetition can be caused by symmetry, tiling, or chance.

## Search quality

The quality score is a weighted sum of smooth target bands, multiplied by a gate requiring actual activity and state-multiset reaction. It is combined with behavioural novelty and a mild preference for compact departure from simple kinematics.

It is not:

- a process measure;
- a life score;
- an observer measure;
- evidence of consciousness;
- a publication endpoint.

The score’s only role is to allocate search compute. Reports disclose all components.

## Finite-window candidates

### Event count

Number of changed cell transitions. Extensive and strongly scale-dependent.

### Activity density

Event count divided by available cell-transitions. A simple intensive candidate that may or may not stabilize across windows.

### Conditional entropy

`H(X_{t+1}|X_t)` under the four-state coding. It is invariant under bijective state relabelling but can change under coarse-graining.

### Compression ratio

Zlib-compressed bytes divided by raw bytes. Included deliberately as a coding-sensitive foil, not as a preferred measure.

## Redescription deltas

The report compares activity density, conditional entropy, state entropy, block entropy, and spatial mutual information before and after exact state/coordinate conjugacies. These structural statistics should remain invariant. Raw zlib compression ratio is reported separately as a deliberately coding-sensitive foil; a nonzero delta is a reason to reject it as an objective process weight, not a simulator failure.
