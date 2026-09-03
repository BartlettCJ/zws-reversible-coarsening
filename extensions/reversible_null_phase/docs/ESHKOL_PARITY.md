# Eshkol parity oracle

The command

```powershell
zws-null-phase export-eshkol --rule rules\winner_d1fe99.json --output eshkol\winner_parity.esk
```

generates a self-contained Eshkol program that:

1. exhaustively checks output range, permutation uniqueness and exact local charge for all 256 inputs;
2. runs the full Margolus update independently on a small exact-inventory torus;
3. prints every final cell plus exact Q and M.

Python stores its own complete expected final lattice beside the source. Running

```powershell
zws-null-phase run-eshkol eshkol\winner_parity.esk --eshkol-run C:\path\to\eshkol-run.exe
```

compares every cell. This is intentionally stronger than comparing one weak checksum. The large GPU study remains PyTorch-based; Eshkol serves as an independent compiler/runtime implementation of the exact semantics.

On Windows, prefer the repository wrapper:

```powershell
.\extensions\reversible_null_phase\scripts\run_eshkol_parity.ps1 `
  -EshkolRun C:\path\to\eshkol-run.exe
```

It discovers Visual Studio Build Tools through `vswhere`, imports the x64 MSVC/Windows SDK environment, discovers the complete local LLVM installation, verifies all required linker libraries, and invokes Eshkol in `--run` mode. Eshkol's `#(...)` form is tensor-backed, so the generated oracle deliberately uses `(vector ...)` to preserve exact integer values when rule outputs are reused as vector indices.
