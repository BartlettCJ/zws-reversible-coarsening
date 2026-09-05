# ZWS reversible coarsening: companion code and evidence

Repository: https://github.com/BartlettCJ/zws-reversible-coarsening

Public code-and-evidence release for the working paper

> Chris Bartlett (2026). *Delayed Distributed Coarsening in an Exactly Reversible
> Signed-Charge-Conserving Block Cellular Automaton.* Companion to *Zero Without Stasis*.
> Zenodo, DOI [10.5281/zenodo.22270456](https://doi.org/10.5281/zenodo.22270456).

The parent paper is *Zero Without Stasis* (DOI [10.5281/zenodo.21966374](https://doi.org/10.5281/zenodo.21966374)).

## What is here

| Directory | Contents |
|---|---|
| `paper/` | LaTeX sources, the figures the paper uses, built PDFs (`paper/build/`), the prior-art review, the claim-evidence ledger, and the correction record for the withdrawn v0.3.2 inference. |
| `src/zws_nullity_lab/` | The core simulator (four-state reversible Margolus block automaton with exact signed-charge conservation), the rule search, and the campaign/analysis code of the first campaigns. |
| `extensions/reversible_null_phase/` | The follow-up package: density and first-passage campaigns, controls and ablations, the history-erasure intervention, the frozen rule (`rules/winner_d1fe99.json`, raw-table SHA-256 `d1fe99b5...`), and the independent Eshkol implementation with its recorded parity. |
| `experiments/` | Per campaign: the preregistration, the frozen plan, the report, the compact result tables (CSV/JSON) behind every number in the paper, the analysis scripts the tests exercise, and the erratum that withdraws the history-erasure headline. |
| `tests/`, `extensions/.../tests/` | Unit and integration tests (exact invariants, exact inversion, deterministic resume, analysis). |
| `docs/` | Protocol, metrics, result schema, statistical plan, threats to validity. |
| `MANIFEST.csv` | Every file in this release with its size and SHA-256. `REDACTIONS.md` lists the text rewrites made for release (local paths and a private repository name replaced by placeholders). |

Not included, by design: raw trajectories and checkpoints (regenerable from the frozen
rule, seeds and configurations; sizes are noted in the paper), videos and interactive
explorers, and the private development history.

## Running it

```
python -m venv .venv && .venv/Scripts/activate      # or source .venv/bin/activate
pip install -e . -e extensions/reversible_null_phase
pytest tests extensions/reversible_null_phase/tests
python -m zws_reversible_null_phase.cli --help
```

A CPU-only smoke campaign is `extensions/reversible_null_phase/configs/smoke.yaml`;
the reported campaigns used the other profiles on an NVIDIA GPU. The Eshkol parity
files record an exact 16x16, 256-step agreement between the Python simulator and
the independent Eshkol program.

## Reproducibility notes

* Expected test result for this release: all core tests and all extension tests pass.
* Configuration hashes: package 0.4.0 of `zws_reversible_null_phase` introduced hash
  scheme 2, which hashes the rule path exactly as declared in the YAML file, so a
  configuration hashes identically at any checkout location. Plans recorded by earlier
  package versions (the v0.3.0 and v0.3.1 campaigns) carry scheme-1 hashes, which
  included the rule file's absolute path on the original machine; they are verified
  from the payload embedded in each plan (`StudyConfig.payload_sha256`), and the test
  suite does exactly that. The rule itself is identified by the raw-table SHA-256
  printed in the paper, which no scheme change touches.
* Three hash-pinned files (`experiments/*/FROZEN_PLAN.json` and
  `extensions/reversible_null_phase/configs/v032_history_erasure.yaml`) are shipped
  byte-for-byte as recorded and therefore still name the original machine's project
  folder, `C:\ZWSlab\...`; every other local path in this release was replaced by a
  placeholder (see `REDACTIONS.md`).
* Git tag names quoted in the campaign records (for example
  `v0.1.4-windows-discovery-freeze`) refer to the authors' development history, not
  to tags of this repository. Campaign working records, review notes and multi-gigabyte
  trajectory archives are retained by the author and are not part of this release.
* The campaign reports were written against the full working archive and occasionally
  cite per-run manifests, input audits, integrity records or figure manifests that are
  not included here. The result tables, summaries and rule files that the paper's
  numbers rest on are included; the omitted records are retained by the author.
* Line endings are stored byte-for-byte: `.gitattributes` disables all end-of-line
  conversion, because several files are hash-pinned by the tests, the release manifest
  and retained checksum records. Clone with any settings and the bytes match.

## Third-party software

The independent parity oracle is written in [Eshkol](https://github.com/tsotchke/eshkol),
a Scheme-based language with native automatic differentiation, compiled to native code
via LLVM, by **tsotchke** (MIT licence). This release pins Eshkol `v1.3.4-evolve`
(`eshkol/ESHKOL_VERSION.txt`) and uses it unmodified; nothing here is part of Eshkol
and the Eshkol project is not affiliated with this study. The paper cites it as
`Tsotchke2026Eshkol` (`paper/references.bib`) and `CITATION.cff` lists it under
`references`. If you use the oracle, cite Eshkol as its authors ask:

```bibtex
@software{eshkol2026,
  title = {Eshkol: A Programming Language for Mathematical Computing},
  author = {tsotchke},
  version = {1.3.4-evolve},
  year = {2026},
  url = {https://github.com/tsotchke/eshkol},
  note = {Scheme-based language with native automatic differentiation}
}
```

## Licence and citation

Code (`src/`, `extensions/`, `configs/`, `tests/`, `eshkol/`, `scripts/`): MIT, see `LICENSE`.
Manuscript, supplement, figures and evidence tables: copyright 2026 Chris Bartlett,
released under CC BY 4.0. Cite the paper via its DOI; `CITATION.cff` carries it as the
preferred citation and lists the parent paper and Eshkol as references.
