# Handoff: Grouped CV Merged and Outcome Work Next

- Author: Codex
- Date: 2026-03-30
- Scope: current branch/worktree roles, grouped-CV merge status, validation evidence, and the clean next starting point for adding other outcomes

## Branch / Worktree Status

- `justin`
  - remote status: matches `origin/justin`
  - role: stable refactor line and current merge base
- `outcome-extension`
  - remote status: matches `origin/outcome-extension`
  - role: follow-on branch for additional outcomes
  - note: intended to remain content-identical to `justin` until outcome-specific work begins
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

## Practical Note About `run all`

- a narrow real-data grouped `run all` smoke was launched only to validate orchestration on the merged line
- that path still pays the full raw-data preprocessing cost, so it spends a long time in `preprocess_intraop`
- do not block new outcome work on waiting for that real-data `run all` smoke; the better validation signal is the successful staged grouped smoke plus the grouped synthetic `run all` tests

## Recommended Next Starting Point

Use `outcome-extension` for the next work.

Start from:

```bash
cd /home/exouser/Inspire-AKI-prediction
git switch outcome-extension
source .venv/bin/activate
```

Then add the next outcome in small slices:

1. extend the outcome/config surface first
2. run a narrow grouped staged validation for the new outcome
3. widen the model set only after the narrow grouped path is clean

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
