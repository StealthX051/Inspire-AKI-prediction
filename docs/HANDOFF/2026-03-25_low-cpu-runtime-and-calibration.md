# Handoff: Low-CPU Runtime, Calibration, and Overnight Sequence Tuning

- Author: Codex
- Date: 2026-03-25
- Branch: `justin`
- Current `HEAD`: `2d3cc8e` `justin signoff 3/24/26 PM improved multithreading performance, fixed calibration data leakage`
- Current worktree: uncommitted follow-up fixes and runtime/default-config updates on top of `2d3cc8e`

## What Changed Today

- Landed the low-CPU tabular runtime patch and the calibration leakage fix in `HEAD` `2d3cc8e`.
- Added follow-up uncommitted fixes after `HEAD` while rerunning the real-data default config.
- Updated agent-facing status in [AGENTS.md](/home/exouser/Inspire-AKI-prediction/AGENTS.md) so the current continuation point and test status are no longer stale.

The main code changes already captured in `HEAD` are:

- grouped calibration CV on `op_id` in [src/inspire_aki/evaluation/calibration.py](/home/exouser/Inspire-AKI-prediction/src/inspire_aki/evaluation/calibration.py), which fixes repeated-row leakage during isotonic calibration
- durable per-study tabular HPO outputs and resume under [src/inspire_aki/pipelines/tune.py](/home/exouser/Inspire-AKI-prediction/src/inspire_aki/pipelines/tune.py) and `artifacts/tuning/tabular_studies/`
- narrow low-CPU tabular concurrency:
  - `svm` HPO fans out across `preop` / `intraop` / `combined`
  - `svm` training fans out across repeat tasks
  - `log_reg` remains serial with a moderate BLAS cap
- SVM training convergence aligned to legacy `tol=0.01` in [src/inspire_aki/models/tabular.py](/home/exouser/Inspire-AKI-prediction/src/inspire_aki/models/tabular.py)
- targeted benchmark filters in [src/inspire_aki/benchmarking.py](/home/exouser/Inspire-AKI-prediction/src/inspire_aki/benchmarking.py) and the CLI

The follow-up worktree changes after that commit are:

- safe Optuna `best_value` handling in [src/inspire_aki/models/hpo.py](/home/exouser/Inspire-AKI-prediction/src/inspire_aki/models/hpo.py), which fixes the sequence-HPO progress callback crash when early trials are only `PRUNED`
- sequence-HPO patience semantics fix in [src/inspire_aki/models/hpo.py](/home/exouser/Inspire-AKI-prediction/src/inspire_aki/models/hpo.py):
  - only true `trial.should_prune()` paths now mark a trial `PRUNED`
  - patience-based early stopping now returns `best_val_metric` and completes the trial
  - this prevents the “all trials pruned, no completed trials” failure mode seen in the overnight sequence tuner
- graceful `Ctrl-C` handling in [src/inspire_aki/cli.py](/home/exouser/Inspire-AKI-prediction/src/inspire_aki/cli.py), [src/inspire_aki/orchestration.py](/home/exouser/Inspire-AKI-prediction/src/inspire_aki/orchestration.py), and [src/inspire_aki/io/progress.py](/home/exouser/Inspire-AKI-prediction/src/inspire_aki/io/progress.py):
  - direct stage commands now exit cleanly with code `130`
  - `run all` now records interrupted runs as `aborted`
  - overlapped child stages are terminated before parent exit
- default sequence batch-size increase in [configs/aki/default.yaml](/home/exouser/Inspire-AKI-prediction/configs/aki/default.yaml):
  - HPO `models.hpo.sequence_batch_size: 4096`
  - final training `models.sequence_defaults.batch_size: 4096`
- test refresh for the above behavior in:
  - [tests/test_cli.py](/home/exouser/Inspire-AKI-prediction/tests/test_cli.py)
  - [tests/test_progress_logging.py](/home/exouser/Inspire-AKI-prediction/tests/test_progress_logging.py)
  - [tests/test_orchestration.py](/home/exouser/Inspire-AKI-prediction/tests/test_orchestration.py)
  - [tests/test_config_and_artifacts.py](/home/exouser/Inspire-AKI-prediction/tests/test_config_and_artifacts.py)
  - [tests/test_refactor_contracts.py](/home/exouser/Inspire-AKI-prediction/tests/test_refactor_contracts.py)
  - [tests/test_sequence_runtime.py](/home/exouser/Inspire-AKI-prediction/tests/test_sequence_runtime.py)

## Documentation Status

The canonical docs are updated for the important behavior changes:

- [README.md](/home/exouser/Inspire-AKI-prediction/README.md)
- [docs/current/pipeline.md](/home/exouser/Inspire-AKI-prediction/docs/current/pipeline.md)
- [docs/refactor/behavior_drift.md](/home/exouser/Inspire-AKI-prediction/docs/refactor/behavior_drift.md)
- [docs/legacy/07_manuscript_alignment.md](/home/exouser/Inspire-AKI-prediction/docs/legacy/07_manuscript_alignment.md)
- [docs/legacy/08_reproducibility_and_known_gaps.md](/home/exouser/Inspire-AKI-prediction/docs/legacy/08_reproducibility_and_known_gaps.md)

Those docs now explicitly cover:

- the grouped-calibration leakage fix
- the durable `tabular_studies` HPO resume behavior
- the intentionally narrow `svm`-only low-CPU concurrency policy
- clean interrupt handling for direct stages and `run all`
- the sequence-HPO fix that distinguishes true Optuna pruning from patience-based early stopping
- the main default `4096` sequence HPO and final-training batch sizes
- the fact that evaluation is still non-nested

## Verification

- Full test suite passed: `.venv/bin/pytest -q`
- Result: `89 passed`

Residual warnings during tests:

- `joblib` / `fork()` deprecation warnings in multithreaded test paths
- existing `torch.load(..., weights_only=False)` future warning in the sequence bundle test

## Current Overnight Run Status

- Active process at handoff time: PID `1456669`
- Current stage: `inspire-aki tune sequence --config configs/aki/default.yaml`
- Current config hash in progress logs: `f88c854b30ac4e11`
- `tune tabular` already completed successfully earlier in this rerun.
- The current `tune sequence` rerun started after:
  - fixing the Optuna callback crash
  - fixing the patience-vs-pruning semantics bug
  - raising default sequence HPO batch size to `4096`
- The previous PID `1446153` was stopped because it was running the pre-fix sequence-HPO logic.
- The current PID `1456669` is the restarted patched process and should now allow patience-stopped trials to complete normally.
- Latest sampled process state after restart:
  - roughly `258%` CPU
  - about `4.95 GiB` RSS
  - GPU sample around `3.7 GiB / 40 GiB` and `23%` utilization
- As of the latest log sample, the restarted run has written a new `stage_start` event for PID `1456669` at `2026-03-25T03:31:12.183862+00:00`; at that snapshot it had not yet logged its first completed or pruned trial after restart.

Useful monitor:

```bash
tail -f artifacts/logs/tune_sequence_progress.jsonl
```

Latest continuation facts:

- because sequence tuning outputs are still written at stage completion, if this overnight run is interrupted before `tune sequence` finishes, the correct resume point remains:

```bash
inspire-aki tune sequence --config configs/aki/default.yaml
```

- if `tune sequence` completes, the next continuation point is:

```bash
inspire-aki train tabular --config configs/aki/default.yaml
inspire-aki train sequence --config configs/aki/default.yaml
inspire-aki evaluate calibrate --config configs/aki/default.yaml
inspire-aki evaluate metrics --config configs/aki/default.yaml
inspire-aki evaluate delong --config configs/aki/default.yaml
inspire-aki evaluate dca --config configs/aki/default.yaml
inspire-aki report manuscript --config configs/aki/default.yaml
```

## Resume Commands

- The prior real-data run from earlier today should still be treated as stale because tuning/training policy changed during the workstream.
- Resume from `tune`, not `train`, unless the current overnight `tune sequence` run finishes successfully.

Recommended command chain:

```bash
source .venv/bin/activate
ts=$(date -u +%Y%m%dT%H%M%SZ)
mkdir -p logs
set -euo pipefail

inspire-aki tune tabular --config configs/aki/default.yaml 2>&1 | tee logs/${ts}_tune_tabular.log
inspire-aki tune sequence --config configs/aki/default.yaml 2>&1 | tee logs/${ts}_tune_sequence.log
inspire-aki train tabular --config configs/aki/default.yaml 2>&1 | tee logs/${ts}_train_tabular.log
inspire-aki train sequence --config configs/aki/default.yaml 2>&1 | tee logs/${ts}_train_sequence.log
inspire-aki evaluate calibrate --config configs/aki/default.yaml 2>&1 | tee logs/${ts}_evaluate_calibrate.log
inspire-aki evaluate metrics --config configs/aki/default.yaml 2>&1 | tee logs/${ts}_evaluate_metrics.log
inspire-aki evaluate delong --config configs/aki/default.yaml 2>&1 | tee logs/${ts}_evaluate_delong.log
inspire-aki evaluate dca --config configs/aki/default.yaml 2>&1 | tee logs/${ts}_evaluate_dca.log
inspire-aki report manuscript --config configs/aki/default.yaml 2>&1 | tee logs/${ts}_report_manuscript.log
```

Useful live monitors:

- `tail -f artifacts/logs/tune_tabular_progress.jsonl`
- `tail -f artifacts/logs/tune_sequence_progress.jsonl`
- `tail -f artifacts/logs/train_tabular_progress.jsonl`
- `tail -f artifacts/logs/train_sequence_progress.jsonl`
- `tail -f artifacts/logs/run_all_events.jsonl`

## Risks / Notes

- The calibration leakage fix is intentional drift from the legacy notebook behavior and should be treated as a correctness fix, not a bug.
- The current repeated-CV evaluation is still optimistic relative to nested CV; that limitation is documented but intentionally unchanged in this patch.
- `Ctrl-C` is now handled cleanly inside `inspire-aki`, but that does not by itself keep a `tmux` session alive; if the shell exits, the tmux session may still disappear unless it is launched with something like `; exec bash` or `remain-on-exit`.
- The current `4096` sequence batch-size change materially increased GPU memory usage, but the early HPO timing signal is mixed rather than clearly faster than `1024`; leave the overnight run as-is, then reassess from real trial timings before increasing batch size further.
