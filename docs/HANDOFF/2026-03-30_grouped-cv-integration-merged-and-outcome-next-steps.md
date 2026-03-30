# Handoff: Grouped CV Merged and Outcome Work Next

- Author: Codex
- Date: 2026-03-30
- Scope: current branch/worktree roles, grouped-CV merge status, modular outcome implementation state, validation evidence, and the clean next starting point for continuing the MACCE run or adding other outcomes

## Branch / Worktree Status

- `justin`
  - remote status: matches `origin/justin`
  - role: stable refactor line and current merge base
- `outcome-extension`
  - remote status: matches `origin/outcome-extension`
  - role: active branch for modular outcome extension work
  - note: now diverges from `justin` with active-outcome config resolution, generic label/report plumbing, grouped-holdout MACCE configs, and new outcome tests
- `cv-integration-aki`
  - current `HEAD`: `2eafd87`
  - remote status: matches `origin/cv-integration-aki`
  - role: preserved integration branch containing the grouped-CV reintegration commits as a reviewable series
- `eval-backend-refactor`
  - current `HEAD`: `231f39d`
  - role: historical donor/reference branch that produced the March 27 grouped-CV artifacts
  - note: useful for provenance and audit, but no longer the preferred development branch

## What Is Now On The Main Line

- the March 27 grouped-CV mechanics have been brought back onto the newer `justin` evaluation and reporting stack
- grouped split generation, grouped tuning, grouped training, and grouped report compatibility now live on the current refactor path
- `run all` now inserts `evaluate generate` automatically for `grouped_holdout` and `grouped_nested_cv`
- the newer report layer from the default AKI run remains intact, including manuscript-style tables, richer statistics, reclassification outputs, and modernized report rendering

Relevant implementation surfaces:

- [`src/inspire_aki/pipelines/evaluate_generate.py`](/home/exouser/Inspire-AKI-prediction/src/inspire_aki/pipelines/evaluate_generate.py)
- [`src/inspire_aki/pipelines/tune.py`](/home/exouser/Inspire-AKI-prediction/src/inspire_aki/pipelines/tune.py)
- [`src/inspire_aki/pipelines/train.py`](/home/exouser/Inspire-AKI-prediction/src/inspire_aki/pipelines/train.py)
- [`src/inspire_aki/datasets/splits.py`](/home/exouser/Inspire-AKI-prediction/src/inspire_aki/datasets/splits.py)
- [`src/inspire_aki/cli.py`](/home/exouser/Inspire-AKI-prediction/src/inspire_aki/cli.py)

## Validation Evidence

- default AKI baseline remains complete under [`/media/volume/ncs_inspire_data/ncs_aki/artifacts/default`](/media/volume/ncs_inspire_data/ncs_aki/artifacts/default)
- historical March 27 grouped-CV reference remains under [`/media/volume/ncs_inspire_data/ncs_aki/artifacts/full_gcv`](/media/volume/ncs_inspire_data/ncs_aki/artifacts/full_gcv)
- current reintegrated grouped-CV staged smoke succeeded under [`/media/volume/ncs_inspire_data/ncs_aki/artifacts/cv_integration_real_smoke_20260330T013549Z`](/media/volume/ncs_inspire_data/ncs_aki/artifacts/cv_integration_real_smoke_20260330T013549Z)
- focused grouped orchestration coverage passed on the integration branch, including grouped synthetic `run all` coverage and grouped tune/train contract tests
- current `outcome-extension` synthetic suite now passes with `131` tests
- a new synthetic MACCE grouped-holdout smoke now passes end to end through `tune tabular`, `train tabular`, calibration, evaluation, consort, and manuscript tables

## Current Real-Data MACCE Snapshot

- active real-data artifact root for the shipped MACCE config:
  - [`/media/volume/ncs_inspire_data/ncs_aki/artifacts/macce_default`](/media/volume/ncs_inspire_data/ncs_aki/artifacts/macce_default)
- `inspire-aki runtime inspect --config configs/macce/default.yaml` resolves cleanly on the current host class
- as of March 30, 2026, `report tables` has been rerun for the MACCE artifact root so:
  - [`performance_table.html`](/media/volume/ncs_inspire_data/ncs_aki/artifacts/macce_default/reports/tables/performance_table.html) now shows grouped-holdout bootstrap CIs instead of `N/A`
  - [`performance_table_calibrated.html`](/media/volume/ncs_inspire_data/ncs_aki/artifacts/macce_default/reports/tables/performance_table_calibrated.html) now shows grouped-holdout bootstrap CIs instead of `N/A`
  - the manuscript-table CI path now bootstraps directly from saved prediction artifacts using the same table-level metric definitions, rather than reusing the broader evaluation bootstrap artifact
  - [`cohort_characteristics.html`](/media/volume/ncs_inspire_data/ncs_aki/artifacts/macce_default/reports/tables/cohort_characteristics.html) now restores the legacy `False = female` encoding, removes duplicated merged department rows, and emits full department names
- current observed run-state snapshot:
  - [`preprocess_preop.json`](/media/volume/ncs_inspire_data/ncs_aki/artifacts/macce_default/manifests/preprocess_preop.json) is present under `manifests/`
  - `logs/` exists under the same artifact root and is where `run_all_events.jsonl` and the `tune_*` / `train_*` progress logs will appear once those stages execute
- practical recovery rule:
  - check the newest file under `manifests/` first to find the last completed stage
  - if `run all` was used, inspect `logs/run_all_events.jsonl`
  - once tuning/training starts, inspect `logs/tune_tabular_progress.jsonl`, `logs/tune_sequence_progress.jsonl`, `logs/train_tabular_progress.jsonl`, and `logs/train_sequence_progress.jsonl`

## Practical Note About `run all`

- a narrow real-data grouped `run all` smoke was launched only to validate orchestration on the merged line
- that path still pays the full raw-data preprocessing cost, so it spends a long time in `preprocess_intraop`
- do not block new outcome work on waiting for that real-data `run all` smoke; the better validation signal is the successful staged grouped smoke plus the grouped synthetic `run all` tests

## Current Outcome-Extension Surface

- active target selection now flows through `study.outcome_key`, `outcome.*`, and derived `models.target`
- shipped outcome catalog entries now cover:
  - `aki`
  - `macce`
  - `pna`
  - `pe`
  - `postop_icu_admission`
  - `postop_mortality_30d`
- active label artifacts now write `cohort/labels.csv`; AKI still also writes `cohort/aki_labels.csv` for compatibility
- report consort/tables and sequence training/prediction are now target-aware
- grouped holdout HPO manifests now split on `patient_id` instead of falling back to op-level HPO splits
- shipped MACCE configs now exist under:
  - [`configs/macce/default.yaml`](/home/exouser/Inspire-AKI-prediction/configs/macce/default.yaml)
  - [`configs/macce/smoke.yaml`](/home/exouser/Inspire-AKI-prediction/configs/macce/smoke.yaml)
  - [`configs/macce/smoke_hpo.yaml`](/home/exouser/Inspire-AKI-prediction/configs/macce/smoke_hpo.yaml)

## Recommended Next Starting Point

Use `outcome-extension` for the next work.

Start from:

```bash
cd /home/exouser/Inspire-AKI-prediction
git switch outcome-extension
source .venv/bin/activate
```

Then extend the remaining outcomes in small slices:

1. extend the outcome/config surface first
2. run a narrow grouped staged validation for the new outcome
3. widen the model set only after the narrow grouped path is clean

The current highest-value real-data continuation is the grouped-holdout MACCE path:

```bash
cd /home/exouser/Inspire-AKI-prediction
git switch outcome-extension
source .venv/bin/activate

inspire-aki runtime inspect --config configs/macce/default.yaml
inspire-aki preprocess preop --config configs/macce/default.yaml
inspire-aki preprocess intraop --config configs/macce/default.yaml
inspire-aki preprocess tabular --config configs/macce/default.yaml
inspire-aki preprocess labels --config configs/macce/default.yaml
inspire-aki preprocess timeseries --config configs/macce/default.yaml
inspire-aki preprocess sequence --config configs/macce/default.yaml
inspire-aki evaluate generate --config configs/macce/default.yaml
inspire-aki tune tabular --config configs/macce/default.yaml
inspire-aki tune sequence --config configs/macce/default.yaml
inspire-aki train tabular --config configs/macce/default.yaml
inspire-aki train sequence --config configs/macce/default.yaml
inspire-aki evaluate calibrate --config configs/macce/default.yaml
inspire-aki evaluate metrics --config configs/macce/default.yaml
inspire-aki evaluate delong --config configs/macce/default.yaml
inspire-aki evaluate dca --config configs/macce/default.yaml
inspire-aki evaluate reclassification --config configs/macce/default.yaml
inspire-aki report manuscript --config configs/macce/default.yaml
```

If only the manuscript tables need regeneration after a reporting-only fix, rerun:

```bash
cd /home/exouser/Inspire-AKI-prediction
git switch outcome-extension
source .venv/bin/activate
inspire-aki report tables --config configs/macce/default.yaml
```

## Quick Commands For The Next Coder

To recheck the current branch/worktree state:

```bash
cd /home/exouser/Inspire-AKI-prediction
git branch -vv
git worktree list
```

To inspect the preserved grouped-CV integration history:

```bash
cd /home/exouser/Inspire-AKI-prediction-cv-integration
git log --oneline -n 5
```
