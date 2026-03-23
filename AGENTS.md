# AGENTS.md

This file is the short agent-facing operating contract for `Inspire-AKI-prediction`.

## Start Here

1. Read `README.md`.
2. Read `docs/README.md`.
3. Treat the numbered scripts as the main pipeline unless a task explicitly targets legacy code.

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

## Safe Working Rules

- Prefer `rg` and targeted reads before editing.
- Do not casually edit notebooks when a `.py` source or checked-in markdown output already captures the same behavior.
- Do not treat checked-in model directories as source code.
- Do not promise reproducibility without private INSPIRE data.
- If you change code behavior, update:
  - `README.md`
  - the relevant `docs/*.md`
  - `docs/07_manuscript_alignment.md` if manuscript-facing behavior changed

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
- Refactor raw predictions are stage-owned partitions under `artifacts/predictions/raw/`, plus a rebuilt combined `raw_predictions.parquet`.
- `inspire-aki report manuscript` now includes SHAP when configured in `reports.manuscript_sections`.
- Legacy alias export is explicit through `inspire-aki compat export-legacy`; `run all` does not export aliases automatically.

## Done Criteria

- Any claim about repo behavior is tied to a real file, script, or checked-in output.
- Any portability limitation is stated plainly.
- Any manuscript drift is documented instead of silently ignored.
- New docs link back into the existing docs index rather than duplicating large sections.

## Deep Docs

- [docs/README.md](docs/README.md)
- [docs/02_data_pipeline.md](docs/02_data_pipeline.md)
- [docs/03_labels_and_features.md](docs/03_labels_and_features.md)
- [docs/04_modeling_and_evaluation.md](docs/04_modeling_and_evaluation.md)
- [docs/07_manuscript_alignment.md](docs/07_manuscript_alignment.md)
- [docs/08_reproducibility_and_known_gaps.md](docs/08_reproducibility_and_known_gaps.md)
- [docs/09_codex_workflow.md](docs/09_codex_workflow.md)
- [docs/refactor/behavior_drift.md](docs/refactor/behavior_drift.md)
