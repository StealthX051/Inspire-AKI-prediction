# AGENTS.md

This file is the short agent-facing operating contract for `Inspire-AKI-prediction`.

## Start Here

1. Read `README.md`.
2. Read `docs/README.md`.
3. Treat the refactored CLI and `src/inspire_aki/` as the current implementation; use `docs/legacy/` and the numbered scripts for audit/parity work.

## Trust Order

- Highest trust:
  - `data_preprocessing/01`-`06`
  - `create_results/07`-`10`
  - `create_results/bootstrap_metrics.py`
  - `create_results/decision_curve.py`
  - checked-in performance tables under `create_results/`
- Medium trust:
  - notebooks under `create_results/`
  - `data_preprocessing/consort_parity.py`
  - `data_preprocessing/outcomes_data_selection.py`
- Lower trust:
  - root notebooks
  - `notebooks/`
  - `preoperative_models/aki_experiments/`
  - legacy single-model scripts in `preoperative_models/`
  - checked-in model artifact directories

## What This Repo Actually Is

- A research codebase for postoperative AKI prediction using INSPIRE data.
- Not turnkey.
- Strongly tied to private data.
- The legacy numbered scripts still reflect the older `/home/server/Projects/data/...` server layout in places.
- The refactor defaults now target the mounted volume path `/media/volume/ncs_inspire_data/ncs_aki/data/inspire`.
- Mid-refactor: the new package path is `src/inspire_aki/` with CLI entrypoint `inspire-aki`.

## Current Handoff Status

As of March 25, 2026:

- the synthetic refactor suite passes with `90` tests
- the refactor now has:
  - live `run all` and stage progress logging
  - grouped calibration CV on `op_id`, which fixes repeated-row leakage during calibration
  - durable per-study tabular HPO outputs under `artifacts/tuning/tabular_studies/`
  - narrow low-CPU tabular concurrency: `svm` fans out across regimes in HPO and across repeats in training, while `log_reg` stays serial with a moderate BLAS cap
- the current repeated-CV evaluation remains non-nested by design: HPO runs once before repeated-CV training/evaluation
- if the model-selection policy changes, resume from `tune ...`, not `train ...`
- the next recommended real-data continuation point for the main default config is:
  - `inspire-aki tune tabular --config configs/aki/default.yaml`
  - `inspire-aki tune sequence --config configs/aki/default.yaml`
  - `inspire-aki train tabular --config configs/aki/default.yaml`
  - `inspire-aki train sequence --config configs/aki/default.yaml`
  - `inspire-aki evaluate calibrate --config configs/aki/default.yaml`
  - `inspire-aki evaluate metrics --config configs/aki/default.yaml`
  - `inspire-aki evaluate delong --config configs/aki/default.yaml`
  - `inspire-aki evaluate dca --config configs/aki/default.yaml`
  - `inspire-aki report manuscript --config configs/aki/default.yaml`

## Safe Working Rules

- Prefer `rg` and targeted reads before editing.
- For feature work, prefer the smallest correct change that fully satisfies the request.
- Extend existing code paths, helpers, configs, and CLIs before adding new ones.
- Do not redevelop, shadow, or parallel an existing feature when the current implementation can be adapted safely.
- Do not perform opportunistic refactors, file moves, renames, or architectural cleanup unless they are required to complete the requested change correctly.
- Keep diffs narrow:
  - touch the fewest files that can reasonably solve the task
  - preserve existing public interfaces, artifact paths, config keys, and command surfaces unless the request requires changing them
  - avoid adding new dependencies, abstractions, wrappers, toggles, or layers when existing patterns are sufficient
- If a new helper or abstraction seems useful, add it only when at least one existing call site can reuse it immediately or the current file would otherwise become materially worse.
- When a request appears to overlap existing behavior, inspect and reuse that behavior first; patch the current implementation rather than replacing it.
- Keep verification proportional to the change:
  - small localized edits should get focused tests/checks
  - broad reruns belong only to changes that actually affect those surfaces
- Do not casually edit notebooks when a `.py` source or checked-in markdown output already captures the same behavior.
- Do not treat checked-in model directories as source code.
- Do not promise reproducibility without private INSPIRE data.
- Keep handoff/TODO docs selective:
  - do not create a new handoff for a read-only review, status check, or restatement of repo state
  - create or update a handoff only when you materially changed code/docs/configs, ran a meaningful command whose result changes the next step, or captured a blocker/traceback/artifact path that would otherwise be lost
  - prefer updating an existing same-day handoff for the same workstream instead of adding another near-duplicate note
  - delete superseded handoff or TODO files when they no longer add unique value
- Keep agent-facing docs concise and non-repetitive:
  - prefer updating an existing canonical doc over adding a new standalone summary
  - avoid restating the same repo context across multiple files unless the duplication is necessary for navigation or manuscript alignment
  - keep notes focused on durable decisions, blockers, artifact paths, and the next concrete step
- Keep instruction files scoped:
  - use the repo-root `AGENTS.md` for stable repo-wide defaults
  - if one subtree needs materially different instructions, add a closer `AGENTS.md` or `AGENTS.override.md` there instead of bloating the root file
- If you change code behavior, update:
  - `README.md`
  - the relevant `docs/*.md`
  - `docs/legacy/07_manuscript_alignment.md` if manuscript-facing behavior changed

## Canonical Script Order

- Refactored CLI:
  - `inspire-aki run all --config configs/aki/default.yaml`
  - `inspire-aki preprocess ...`
  - `inspire-aki train ...`
  - `inspire-aki evaluate ...`
  - `inspire-aki report ...`
- Legacy audit/parity order:

- Tabular path:
  - `data_preprocessing/01_extract_preop.py`
  - `data_preprocessing/02_extract_intraop.py`
  - `data_preprocessing/03_create_base.py`
  - `data_preprocessing/04_AKI_data_selection.py`
  - `create_results/07_tabular_hpo.py`
  - `create_results/08_tabular_model_creation.py`
- Sequence path:
  - `data_preprocessing/05_time_series_cleaner.py`
  - `data_preprocessing/06_create_lstm_trainable.py`
  - `create_results/09_lstm_hpo.py`
  - `create_results/10_lstm_model_creation.py`
- Evaluation path:
  - `create_results/11_consort.ipynb`
  - `create_results/12_cohort_characteristics.ipynb`
  - `create_results/13_performance_metrics.ipynb`
  - `create_results/14_delong_table.ipynb`
  - `create_results/15_shap.ipynb`
  - `create_results/16_shap_batch.ipynb`

## Non-Obvious Repo Facts

- AKI staging currently includes dialysis via `ward_vitals.csv` `crrt`.
- Base preprocessing normalizes before imputation.
- Sequence preparation pads to `200` steps and drops longer cases.
- Current training toggles do not enable every model or every dataset.
- `asa_rule` only applies to datasets that still contain `asa`.
- The refactor writes manifests and stage outputs under `artifacts/`.
- The refactor resolves stage worker/thread budgets through `src/inspire_aki/runtime.py`; use `inspire-aki runtime inspect --config ...` before expensive runs on a new host class.
- Refactor raw predictions are stage-owned partitions under `artifacts/predictions/raw/`, plus a rebuilt combined `raw_predictions.parquet`.
- Parallel timeseries and sequence preprocessing now uses internal staging artifacts under `artifacts/staging/`.
- `inspire-aki report manuscript` now includes SHAP when configured in `reports.manuscript_sections`.
- Legacy alias export is explicit through `inspire-aki compat export-legacy`; `run all` does not export aliases automatically.
- Explicit team notes now live under `docs/TODO/` and `docs/HANDOFF/`.

## Done Criteria

- Any claim about repo behavior is tied to a real file, script, or checked-in output.
- Any portability limitation is stated plainly.
- Any manuscript drift is documented instead of silently ignored.
- New docs link back into the existing docs index rather than duplicating large sections.

## Deep Docs

- [docs/README.md](docs/README.md)
- [docs/current/README.md](docs/current/README.md)
- [docs/current/pipeline.md](docs/current/pipeline.md)
- [docs/legacy/02_data_pipeline.md](docs/legacy/02_data_pipeline.md)
- [docs/legacy/03_labels_and_features.md](docs/legacy/03_labels_and_features.md)
- [docs/legacy/04_modeling_and_evaluation.md](docs/legacy/04_modeling_and_evaluation.md)
- [docs/legacy/07_manuscript_alignment.md](docs/legacy/07_manuscript_alignment.md)
- [docs/legacy/08_reproducibility_and_known_gaps.md](docs/legacy/08_reproducibility_and_known_gaps.md)
- [docs/codex_workflow.md](docs/codex_workflow.md)
- [docs/refactor/behavior_drift.md](docs/refactor/behavior_drift.md)
