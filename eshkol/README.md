# Eshkol Independent Exact Oracle

The primary simulator and search engine are Python/PyTorch. For every selected rule, `zwslab export-eshkol` emits a self-contained Scheme/Eshkol program that checks:

1. all 256 outputs lie in `0..255`;
2. no output is repeated;
3. exact integer block charge is unchanged for every input.

Run a generated oracle with:

```bash
eshkol-run analysis/eshkol_exact_oracle.esk
```

Expected output:

```text
RULE <name>: PASS -- bijective and exactly charge-preserving on all 256 blocks
```

## Installing the pinned Windows release

```powershell
.\scripts\install_eshkol_windows.ps1
.\scripts\run_latest_eshkol_oracle.ps1
```

The installer selects the Windows x64 CUDA archive when present, verifies it against the release `SHA256SUMS.txt`, and records the runner path locally.

## Building the pinned compiler

From WSL/Linux:

```bash
bash scripts/build_eshkol_wsl.sh
```

This clones `tsotchke/eshkol` at tag `v1.3.4-evolve`, records the resolved commit, configures a release build, and installs no global files. The exact oracle does not require CUDA, although the build script can enable it when the local toolchain supports it.

The oracle is intentionally small. A compiler failure is reported separately from Python test results; it is not grounds to waive the local exhaustive audit.
