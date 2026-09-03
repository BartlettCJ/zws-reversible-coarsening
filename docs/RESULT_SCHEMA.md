# Result Schema

## Run root

- `config.resolved.yaml` — exact merged configuration; resume is hash-locked to it.
- `manifest.json` — Python, PyTorch, device, GPU, seed, and configuration hash.
- `process_policy.json` — best-effort priority/CPU-affinity result.
- `status.json` — atomic heartbeat, stage, generation, candidate index, and checkpoint summary.
- `resource_state.json` — latest resource decision.
- `resource_log.csv` — resource samples and yield reasons.
- `generation_history.csv` — one row per completed generation.
- `search_history.png` — visual search-quality history.
- `search_result.json` — ranked paths and full generation history.
- `checkpoints/resumable/LATEST.json` — pointer and hash for the latest full search checkpoint.
- `checkpoints/resumable/checkpoint_*.pt[.sha256]` — retained verified fallback checkpoints.
- `checkpoints/generation_*` — legacy human-readable best-at-interval snapshots.
- `rules/` — ranked final rules.
- `interesting/<category>/` — successive champion rule JSON, exact state tensor, preview PNG, and metadata.
- `control/` — optional `PAUSE`, `RESUME`, `HIBERNATE`, and `STOP` request files.

## Analysis root

- `selected_rule.json`
- `analysis_config.yaml`
- `analysis_progress.pt` — stage-resume state, rule hash, and configuration hash.
- `analysis_status.json`
- `analysis.json`
- `REPORT.md`
- `REPORT.html`
- `eshkol_exact_oracle.esk`
- `assets/trajectory.pt` — exact uint8 PyTorch trajectory.
- `assets/final_state.png`
- `assets/trajectory.gif`
- `assets/controls.png`
- `assets/events/events.json`
- `assets/events/event_*.png`
- `assets/events/event_*_crop.png`
- `assets/events/event_*_state.npy`
- `assets/events/event_*.json`
- `assets/events/event_contact_sheet.png`
- `explorer/explorer.html`
- `explorer/trajectory.json`
- `explorer/trajectory.bin`
- `explorer/metric_series.json`

## `analysis.json`

Top-level keys:

- `schema` (`zws-nullity-analysis-v2`)
- `created_utc`
- `rule`
- `environment`
- `local_rule_audit`
- `trace_audit`
- `cross_device_audit`
- `holdout`
- `controls`
- `redescription`
- `coarse_graining`
- `finite_window_measure`
- `cycle`
- `events`
- `interpretation`
- `paths`

Downstream analysis should use this JSON rather than scraping the prose report.

## Trajectory bundle

`trajectory.json` declares:

- schema and rule SHA-256;
- frame/height/width dimensions;
- `uint8` layout;
- state names, charges, and palette;
- binary SHA-256;
- event metadata and metric-series path.

`trajectory.bin` is a contiguous `frames × height × width` uint8 array. The explorer verifies dimensions; the Python loader also verifies its SHA-256 before exact capture export.

## Capture bundle

A CLI capture contains:

- full-frame PNG;
- declared crop PNG;
- exact uint8 `.npy` state;
- surrounding GIF clip;
- JSON with rule hash, frame/tick, crop, clip interval, state checksum, and paths.
