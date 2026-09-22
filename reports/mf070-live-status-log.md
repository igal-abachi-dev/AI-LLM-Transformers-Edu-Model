# MF-070 live status log — raw data for a post-completion training curve

`mf070-live-status-log.csv` (same directory) is a real, append-only record of periodic
external "live status" checks against `MF-070`'s live 150M-Modern release run (started
2026-09-15 16:24) — not the formal `RunMetadata`/`run.json` the training script writes
itself, but an independent, manually-triggered log of the same run's progress plus real
GPU telemetry (temperature/utilization/VRAM/power) that `RunMetadata` doesn't capture.
Purpose: once the run completes, this gives a real start-to-finish curve (loss, throughput,
thermals) to look for patterns, rather than only ever seeing the single latest snapshot.

**Honesty note on the first four rows (2026-09-18 through 2026-09-21T02:55):** GPU
temperature/utilization/VRAM/power weren't preserved in enough detail across earlier
session turns to reconstruct with confidence for those specific checks, so those cells are
left blank rather than guessed. `update`/`progress_pct`/`elapsed_hours`/`tokens_per_sec`/
`loss_mean_last300` for every row are real, either read directly from `nvidia-smi`/the
training log at the time or carried over precisely from `mf070-live-run-status.md`'s own
recorded snapshots. From 2026-09-21T16:24 onward, every column is a real, directly-captured
reading.

## Columns

- `timestamp` — when the check was run (local time, no timezone conversion applied).
- `update` / `max_updates` / `progress_pct` — training progress.
- `elapsed_hours` — wall-clock time since the run started (from the log's own `elapsed=`
  field, not derived from `timestamp` deltas).
- `tokens_per_sec` — from the training log's own `tokens/s=` field.
- `loss_mean_last300` — mean of the raw per-update `loss=` value over the last 300 log
  lines at check time (smooths the real, noisy per-update swing — individual updates can
  range ~0.3-4.0 depending on which mixture source landed that step).
- `gpu_temp_c` / `gpu_util_pct` / `vram_used_mib` / `power_w` — from `nvidia-smi`, when
  captured.
- `checkpoint` — the latest periodic checkpoint directory name at check time.
- `notes` — anything worth flagging about that specific check (anomalies, data-gap
  disclosures).

## How to keep this current

Append one new row per "live status... record it" request, in the same format — don't
rewrite or trim earlier rows (that's what `mf070-live-run-status.md`'s own rolling
snapshot is for; this file is deliberately append-only). Once the run completes, this
becomes the real source data for whatever loss/throughput/thermal curve or statistics get
built afterward.
