# Claim–evidence ledger — companion paper v0.2.1

Date: 2026-08-31  
Rule raw-uint8 SHA-256: `d1fe99b5735d8e9be032347de2727dc5c52a7848df200ca26f67f28c8c1bc41c`

This ledger separates exact audit, measurement, bounded inference, interpretation, and withdrawn claim. “Strong-domain” always means the stored operational endpoint, not an equilibrium phase.

| ID | Public claim | Evidence class | Authoritative evidence | Warranted wording | Required limitation |
|---|---|---|---|---|---|
| C01 | The local rule is exactly invertible. | Exact audit | `rules/winner_d1fe99.json`; exhaustive 256-case permutation tests | The table is a permutation and has an exact inverse. | Reversibility is by construction when implemented correctly; not a novelty claim. |
| C02 | Signed charge is locally conserved. | Exact audit | rule audit; Eshkol/PyTorch parity records | Every local block transition preserves `Q = N_pos - N_neg`. | Principal runs start in global `Q = 0`; do not say the rule “conserves Q=0”. |
| C03 | Occupancy may change. | Exact audit + trajectory | local transition ledger; composition time series | The winner does not conserve `M = N_pos + N_neg + N_bound`. | `M` is distinct from signed charge and is not energy or physical mass. |
| C04 | A large localized-patch trajectory reversed exactly. | Exact audit | Large-run configuration, original confirmation preregistration and reversal audit | A 4096×4096 localized-patch state returned after 65,536 forward and 65,536 inverse phases with zero mismatches. | Distinct from the later homogeneous 1024² cohort; invertibility is by construction. |
| C05 | Large irregular structure develops. | Confirmation | four localized-patch 4096² winner confirmations; Figures 1–2 | Transient scale grew 72.66–136.11× and final recorded scale 15.77–16.33×. | Finite-size, finite-time, and nonmonotone; call it coarsening-like. |
| C06 | Local compositional disorder remains high. | Measurement | composition audit; Figure 2 | Dense late state normalized entropy was 0.946343 at occupancy 0.572989. | Single-site entropy does not rule out all microscopic order. |
| C07 | Waiting time depends strongly on starting density. | Preregistered follow-up | `FIRST_PASSAGE_RESULTS.csv`; `SURVIVAL_SUMMARY.csv`; Figure 3 | Event fractions and median first passage shift strongly from occupancy 0.16 to 0.26. | No equilibrium critical density or infinite-time threshold. |
| C08 | Transformation morphology is distributed under the frozen labels. | Mechanism constraint | v0.3.1 nucleation report | 67 events distributed, 24 unresolved, 0 localized. | Classifier/window dependent; does not exclude every nucleation description. |
| C09 | Prepared-shape and homogeneous runs differ under the recorded relative-growth endpoint. | Descriptive matched comparison | v0.3.1 seeded-basin results and frozen endpoint | 0/48 prepared-shape crossings versus 19/24 homogeneous at 262,144 phases; occupied shapes persisted or grew. | Each trajectory has its own initial finite-empty-component denominator. These counts are not an absolute seed-survival or organisation test. |
| C10 | The closest exact-`Q+M` sibling does not reproduce the winner regime. | Matched control | domain mechanism and v0.3 balanced summaries | Sibling 0/16, winner 10/16; sibling remained reactive. | One constructed sibling; no general impossibility theorem. |
| C11 | Transport-only rotation does not reproduce the regime. | Negative control | v0.3 balanced summary | Pure rotation 0/16 under the matched protocol. | Does not exclude other transport rules. |
| C12 | Several reaction pairs contribute. | Preregistered ablation | v0.3 summary; Figure 4 | Six single-pair removals gave 0/16 and others 2–7/16. | Single removals do not identify a unique minimal mechanism. |
| C13 | The regime is not unique within the sampled generator. | Bounded prevalence sample | 256-rule prevalence report | 19/256, 7.4219%, Wilson 95% 4.8026–11.3001%. | Not prevalence over all permutations or all charge-preserving rules. |
| C14 | The winner was search-selected. | Provenance | `selected_rule.json`; `metrics.py` | Discovery considered 14 unique rules / 16 replicate evaluations using a disclosed dynamics score. | Subsequent tests do not retroactively make discovery unselected. |
| C15 | Later measures were not in the discovery score. | Code/provenance audit | `metrics.py` and campaign chronology | Large reversal, first passage, morphology, sibling, ablation, prevalence, and v0.3.2 diagnostics were follow-ups. | Follow-up hypotheses vary between confirmatory and mechanism-constraining. |
| C16 | v0.3.2 shuffling erased measured organisation. | Intervention audit | topology-fixed branch JSON/CSVs; classifier review | Branch-A long-wave power fell by 99.211%; MI became approximately zero. | Does not itself identify a causal effect on later transformation. |
| C17 | The v0.3.2 endpoint failed under the shuffled null. | Diagnostic audit | `V032_CLASSIFIER_REVIEW_20260831.md`; erratum | 9/16 Branch-B primary shuffled states hit at elapsed zero; 14/16 by 4,096 phases despite very low long-wave power. | The intervention ran correctly; the endpoint was invalid for this contrast. |
| C18 | The history-inhibition/hidden-memory headline is withdrawn. | Correction | erratum; corrected report; amended stop decision | The campaign cannot establish preserved precursor memory, inhibition, acceleration, or causal necessity. | Historical files remain immutable and must be read with the erratum. |

## Prohibited public formulations

- “The rule conserves Q=0.”
- “Q and M are both zero.”
- “The model proves a phase transition.”
- “Density 0.18/0.20 is the critical density.”
- “Variable M is necessary for reversible organisation.”
- “The transformation nucleates by a local Poisson process.”
- “History erasure proves hidden multiscale memory.”
- “Shuffled worlds immediately rebuilt the same domains.”
- “The v0.3.2 first-passage result validates a causal mechanism.”

## Safe top-line formulation

One fully specified, exactly reversible four-state block automaton conserves signed charge while allowing occupancy to change, and it reproducibly develops delayed, irregular, large-scale finite-time organisation from random material. A 4096×4096 trajectory returned exactly after 65,536 phases. Density, geometry, a reactive exact-`Q+M` sibling, rotation, and reaction-pair ablations constrain the phenomenon; they do not establish equilibrium or a universal mechanism. The later history-erasure headline is withdrawn because its endpoint misclassified shuffled null geometry.

## v0.2.1 source-to-evidence corrections

The principal first-passage cohort is homogeneous 1024², not the localized 4096² reversal experiment. Its event predicate is finite-empty-component geometry plus a charge audit; spectra are separate diagnostics. Three-observation persistence is a separate endpoint from first hit. The corrected density table reports actual follow-up: 262,144 for 0.16 and 0.22–0.26; 524,288 for 0.18; mixed original/extended records for 0.20. No recorded event time or scientific result changed. Figure 2 is a separate untouched 1024² source, seed 6203 at phase 311,296, not the four-run localized composition statistic.

## v0.2.2 presentation and disclosure additions

The plain-language opening and conclusion rest on the existing exact-Q/inversion audits, irregular-domain observations, composition summaries and controlled density/reaction comparisons. They do not introduce an infinite-time activity claim or an entropy-decrease claim. The first figure uses the same four historical image panels in a 2×2 layout; `FIGURE_01_LAYOUT_V022.json` records the rendered-pixel source, exact extraction rectangles and inherited state provenance. No raw state hash was recomputed from a rendered image.

The AI-assistance statement discloses broad categories of actual assistance without importing model aliases or role assignments from the separate ZWS paper. Public code/data links are explicitly marked as pending; release plans are not assertions of existing public availability. Licence recommendations remain owner decisions.

## v0.2.3 figure and literature audit

The opening composite is an opaque RGB flattening of the v0.2.2 archived panels. The worked-rule diagram is generated from exact entries 1 -> 4 and 130 -> 154 of the frozen table and records the raw uint8 rule hash. The bibliography audit corrects three title/DOI pairings, adds the Morita-Imai DOI, and adds close two-species reversible and reactive lattice-gas neighbours. These are publication and positioning changes only; scientific evidence and results are unchanged.

## v0.2.4 Figure 1 source repair

| Claim | Evidence | Status | Wording ceiling |
|---|---|---|---|
| Figure 1 uses the same four archived scientific panels as the earlier layout, embedded directly rather than through the faulty composite-PDF path | `FIGURE_01_SOURCE_PANEL_AUDIT_V024.json`, `FIGURE_01_LAYOUT_V022.json`, source panel hashes, PDF QA | exact publication audit | Presentation/compatibility correction only; no new frame, crop, state, measurement or scientific inference |
