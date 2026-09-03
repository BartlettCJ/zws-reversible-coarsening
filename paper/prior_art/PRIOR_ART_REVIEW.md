# Focused prior-art review

## Scope and method

This review positions the ZWS reversible-null-phase study against a deliberately
bounded set of primary papers and verified bibliographic records. The search
covered reversible and partitioned cellular automata, reversible lattice gases,
Q2R/Ising automata, reversible aggregation, conservation laws, reaction-diffusion
automata, phase ordering, rare-event trajectories, and interventions that separate
configuration from dynamical randomness. The exact queries, portals, and stopping
rule are retained by the author and are not part of this release; DOI checks are recorded in `DOI_AUDIT.csv`.

This is a novelty-positioning review, not a claim of exhaustive priority. Terms
such as *reversible aggregation*, *conservative automaton*, *Q2R*, *partitioned
cellular automaton*, and *isoconfigurational ensemble* divide a broad literature
that does not use one common vocabulary.

## 1. Established broad precedent

Exact microscopic reversibility in cellular automata is longstanding. Fredkin and
Toffoli developed conservative collision-based logic, and Margolus made reversible
block updates a practical way to encode locality and invertibility
([Fredkin & Toffoli 1982](https://doi.org/10.1007/BF01857727);
[Margolus 1984](https://doi.org/10.1016/0167-2789(84)90252-5)). Vichniac emphasized
that reversible microscopic dynamics may nevertheless display an apparent
macroscopic arrow of time, nonergodicity, order parameters, and large-scale
correlations ([Vichniac 1984](https://doi.org/10.1016/0167-2789(84)90253-7)).
Those papers establish the conceptual and architectural precedent; the present
study does not claim to originate reversible block cellular automata or emergent
macroscopic irreversibility.

Deterministic lattice gases likewise established that exact discrete microscopic
dynamics can reproduce macroscopic transport while conserving particle-like
quantities ([Frisch, Hasslacher & Pomeau 1986](https://doi.org/10.1103/PhysRevLett.56.1505)).
Takesue connected additive conserved quantities in reversible automata to
statistical-mechanical behavior, while Hattori and Takesue gave a local-current
criterion for additive conservation laws
([Takesue 1987](https://doi.org/10.1103/PhysRevLett.59.2499);
[Hattori & Takesue 1991](https://doi.org/10.1016/0167-2789(91)90150-8)).
Particle- and number-conserving automata were developed independently as a large
formal class ([Kohyama 1989](https://doi.org/10.1143/PTP.81.47);
[Boccara & Fukś 2002](https://doi.org/10.3233/FUN-2002-521-302)). These are direct
precedents for exact discrete invariants, but a conserved nonnegative particle
number is not the same object as a signed charge constrained to zero while total
occupancy changes.

## 2. Nearest technical neighbours

### Reversible Ising and Q2R automata

Creutz introduced deterministic Ising dynamics with spin, parity, and an
energy-exchange degree of freedom, enabling nonequilibrium, heat-flow, mixing, and
time-correlation studies ([Creutz 1986](https://doi.org/10.1016/S0003-4916(86)80006-9)).
The Q2R line is still the closest family for conservative, reversible lattice
dynamics with macroscopic ordering. Later work describes long cycles,
coarse-grained irreversible behavior, equilibrium and nonequilibrium regimes, and
phase-space structure
([Urbina & Rica 2016](https://doi.org/10.1103/PhysRevE.94.062140);
[Urbina 2020](https://doi.org/10.1103/PhysRevE.101.012111);
[Montalva-Medel, Rica & Urbina 2020](https://doi.org/10.1016/j.chaos.2020.109618)).

The overlap is substantial: deterministic reversible local dynamics, conserved
quantities, long transient or cyclic behavior, and macroscopic order. The exact
difference is also substantial. Q2R is an Ising-derived spin/energy construction
with every lattice site occupied by a spin-like degree of freedom. ZWSLab uses four
cell states, an exact signed charge

\[
Q=N_{+}-N_{-}=0,
\]

and a distinct occupancy/material observable

\[
M=N_{+}+N_{-}+N_{\mathrm{bound}}
\]

that is *not* conserved by the winner. The ZWSLab programme also includes
exhaustive local invariant audits, reaction-pair ablations, density-dependent
first-passage measurements, exact reversal of a large trajectory, and a
preregistered state-matched spatial-history intervention. None of the inspected
Q2R records reports that complete combination.

### Reversible aggregation

D’Souza and Margolus introduced a local deterministic microscopically reversible
lattice gas in which gas particles aggregate into a crystal while latent heat is
explicitly carried by mobile heat tokens
([D’Souza & Margolus 1999](https://doi.org/10.1103/PhysRevE.60.264)). This is the
closest precedent for visually striking aggregation under exact microscopic
reversibility. It proves that reversible dynamics and cluster growth are fully
compatible.

The mechanism and observables differ. Reversible aggregation uses explicit gas,
crystal, and heat species and a crystal-boundary growth/evaporation process. The
ZWSLab winner instead has signed positive/negative excitations and a bound-neutral
state, preserves charge exactly, allows occupancy to change, and develops
irregular excitation-rich and near-vacuum domains without a simple crystal front.
The final experiment removes spatial organization while preserving every current
state count; the reversible-aggregation paper does not report a matched-inventory
global/tile reshuffling intervention or reaction-pair ablation programme.

### Reaction-diffusion cellular automata

Weimar and Boon constructed cellular automata from reaction-diffusion equations
using moving-average diffusion and probabilistic reaction lookup
([Weimar & Boon 1994](https://doi.org/10.1103/PhysRevE.49.1749)). This supplies
broad precedent for local reaction rules generating macroscopic spatial
organization. It differs in all three defining constraints here: it is
probabilistic rather than exactly invertible, it does not impose the ZWS signed
charge invariant, and it does not use same-inventory history erasure to test a
delayed transition.

## 3. Kinetic and causal-analysis neighbours

The continuum phase-ordering literature supplies the language of coarsening,
conserved versus nonconserved fields, scaling, and dynamic universality
([Hohenberg & Halperin 1977](https://doi.org/10.1103/RevModPhys.49.435);
[Bray 1994](https://doi.org/10.1080/00018739400101505)). In particular, the
conceptual neighborhood of a nonconserved order parameter coupled to a conserved
quantity is useful when discussing ZWSLab's variable occupancy alongside exact
charge. It is an analogy, not an identification: the automaton has no demonstrated
equilibrium free energy, temperature, or membership in a Hohenberg–Halperin
universality class.

Transition-path sampling provides a rigorous precedent for studying rare
transitions using ensembles of deterministic trajectories without assuming a
reaction coordinate in advance
([Bolhuis, Dellago & Chandler 1998](https://doi.org/10.1039/A801266K)). ZWSLab's
first-passage and event-centered measurements are methodologically adjacent, but
they are direct finite ensembles rather than transition-path Monte Carlo and do
not estimate a thermodynamic rate constant.

The isoconfigurational ensemble of Widmer-Cooper, Harrowell, and Fynewever holds a
particle configuration fixed while varying momenta to distinguish configurational
propensity from intermittent dynamical randomness
([Widmer-Cooper, Harrowell & Fynewever 2004](https://doi.org/10.1103/PhysRevLett.93.135701)).
This is the closest intervention logic found: hold selected state information
fixed and vary another part of the microstate to test causal predictiveness.
ZWSLab performs the complementary deterministic manipulation—hold the complete
single-site inventory and update phase fixed while deliberately erasing spatial
arrangement. It is not an isoconfigurational ensemble, and the analogy should not
be stretched beyond that experimental logic.

## 4. Narrow residual contribution after the v0.3.2 correction

Subject to the bounded search and the uncertainty below, the residual contribution
is the combination of:

1. an exactly reversible 2×2 block cellular automaton with four local states;
2. exhaustive verification of the exact signed invariant `Q` while the studied
   worlds occupy the global `Q=0` sector and occupancy `M` may change;
3. delayed, irregular, high-entropy macroscopic domain formation rather than a
   simple crystal or checkerboard endpoint;
4. exact cell-for-cell reversal of a 4096×4096 trajectory after 65,536 phases;
5. finite-ensemble density, waiting-time, event-morphology, and prepared-geometry
   measurements; and
6. a frozen programme of rotation, reactive exact-`Q+M` sibling, prevalence, and
   multi-reaction controls.

The v0.3.2 spatial-history intervention is not part of the residual scientific
claim. Its shuffles were valid, but its endpoint classified random connected-empty
geometry as a strong domain before long-wave organisation recovered. It survives
as a useful classifier-failure case study, not evidence of causal spatial memory.

## 5. What remains uncertain

- The search did not prove nonexistence of an exact signed-charge-zero analogue.
  Older partitioned-automaton and lattice-gas work is terminologically dispersed,
  and some proceedings or technical reports may not be well indexed.
- Some reversible lattice gases have multiple species and additive invariants.
  The inspected records did not expose the exact ZWS combination of zero signed
  charge, variable occupancy, delayed irregular coarsening, reaction-pair
  ablations, and matched-inventory history erasure, but a deeper specialist review
  could identify closer constructions.
- Q2R and reversible-aggregation papers are genuine technical neighbours, not
  merely background. Any paper should compare mechanisms explicitly rather than
  imply an empty field.
- Phase-ordering terminology is useful descriptively. Without equilibrium,
  infinite-time, or universality evidence, ZWSLab should say *finite-time
  coarsening-like organization* or *strong-domain transformation*, not an
  equilibrium phase transition.

## Positioning statement for the companion paper

Reversible cellular automata, conservative lattice gases, reversible Ising/Q2R
models, and reversible aggregation already show that invertible local rules can
support statistical-mechanical behavior and macroscopic structure. The present
work studies a narrower architecture: a four-state block automaton that exactly
conserves signed charge while allowing occupancy to change. Its residual
contribution is the audited combination of delayed distributed coarsening-like
organisation, exact large-lattice reversal, broad density-dependent waiting-time
ensembles, event morphology, prepared-geometry controls, a reactive exact-`Q+M`
sibling, and reaction-pair ablations. The result is finite-time and model-specific;
it is not an equilibrium, universality, or causal-history claim.

## v0.2.3 focused correction and extension

An entry-by-entry title/DOI recheck found three inherited metadata mismatches: HPP1973, Broadwell2001, and Klobas-Prosen2022 had correct DOIs paired with titles from other works. The bibliography and audit CSV now use the titles shown by the publisher or author records. The Morita-Imai publisher DOI is now recorded. Two close neighbours were added: a one-dimensional two-species hardcore reversible cellular automaton with vacancies and charge transport, and a reactive lattice-gas automaton that produces waves and Turing patterns. These records narrow the positioning; they do not change any ZWSLab result or imply exhaustive priority.
