# Inspire-AKI-prediction

Legacy names inside the repo still refer to `VitalDB-Dimensionality-Reduction`. This repository is best understood as a codebase for studying postoperative acute kidney injury (AKI) prediction in noncardiac surgery patients using the INSPIRE dataset.

## Current Refactor Status

- A refactored package-first execution path now exists under `src/inspire_aki/`.
- The new canonical entrypoint is the Typer CLI `inspire-aki`.
- The legacy numbered scripts and notebooks still matter for audit/parity, but they are no longer the only execution surface.
- The repo is still not turnkey: private INSPIRE data and explicit path configuration are still required.
- The refactored training defaults to `available CPU workers - 2` through a shared runtime policy.
- Raw refactor predictions are now written as stage partitions plus a deterministic combined `raw_predictions.parquet` view.
- `inspire-aki report manuscript` is now the report-level command that includes SHAP, rather than requiring a separate SHAP call.

## What This Repo Is

- A mixed Python + notebook pipeline for:
  - cohort construction from INSPIRE source tables
  - AKI label derivation from perioperative creatinine data
  - preoperative and intraoperative feature engineering
  - tabular model training and deep sequence model training
  - calibration, thresholding, bootstrap confidence intervals, DeLong testing, decision-curve analysis, and SHAP interpretation
- A messy research repo with both newer numbered scripts and older exploratory code paths.
- Not a turnkey package. Most execution paths assume private INSPIRE data and hard-coded absolute paths under `/home/server/Projects/data/...`.

## Current Code-First Takeaways

- The most canonical executable path is the numbered script sequence:
  - `data_preprocessing/01_extract_preop.py`
  - `data_preprocessing/02_extract_intraop.py`
  - `data_preprocessing/03_create_base.py`
  - `data_preprocessing/04_AKI_data_selection.py`
  - `data_preprocessing/05_time_series_cleaner.py`
  - `data_preprocessing/06_create_lstm_trainable.py`
  - `create_results/07_tabular_hpo.py`
  - `create_results/08_tabular_model_creation.py`
  - `create_results/09_lstm_hpo.py`
  - `create_results/10_lstm_model_creation.py`
- Evaluation and figure generation then continue in notebooks under `create_results/`.
- The repo strongly supports the study conclusion that tabular models outperform the sequence path on the checked-in results:
  - `create_results/performance_table.md` reports combined AutoGluon AUROC `0.932`
  - `create_results/performance_table.md` reports combined `MLP+LSTM` AUROC `0.825`
- Important confirmed implementation details:
  - `data_preprocessing/04_AKI_data_selection.py` includes dialysis via `ward_vitals.csv` `crrt` when defining stage 3 AKI.
  - `data_preprocessing/03_create_base.py` normalizes before missing-value imputation, uses `-99` for `>=10%` missingness, and uses `KNNImputer` for lower-missingness columns.
  - `data_preprocessing/05_time_series_cleaner.py` cleans and interpolates 24 regular intraoperative signals.
  - `data_preprocessing/06_create_lstm_trainable.py` pads sequences to length `200` and drops longer operations.
  - `create_results/08_tabular_model_creation.py` and `create_results/10_lstm_model_creation.py` use repeated fold-style bootstrap splitting and do not enable every model/dataset by default.

## Repo Status

Treat this repository as a research archive with a partially modernized pipeline, not as polished production code.

- Mixed trust level:
  - higher trust: numbered scripts, checked-in performance tables, `bootstrap_metrics.py`, `decision_curve.py`
  - medium trust: `create_results/*.ipynb`, `data_preprocessing/consort_parity.py`, `data_preprocessing/outcomes_data_selection.py`
  - lower trust: exploratory notebooks, one-off scripts in `preoperative_models/`, legacy cleaners, artifact directories
- Private data dependency:
  - raw INSPIRE tables are not checked in
  - many scripts assume a local server filesystem layout
- Cohort-count drift exists:
  - manuscript summary says `49,198` patients
  - `data_preprocessing/consort_diagram_data.ipynb` records a filtered count of `57,724`
  - comments in `create_results/13_performance_metrics.ipynb` refer to roughly `67k` tabular patients and `54k` LSTM/hybrid patients

## Canonical Pipeline Map

### Refactored CLI path

1. `inspire-aki preprocess preop`
2. `inspire-aki preprocess intraop`
3. `inspire-aki preprocess tabular`
4. `inspire-aki preprocess labels`
5. `inspire-aki preprocess timeseries`
6. `inspire-aki preprocess sequence`
7. `inspire-aki tune tabular|sequence`
8. `inspire-aki train tabular|sequence`
9. `inspire-aki evaluate calibrate|metrics|delong|dca`
10. `inspire-aki explain shap`
11. `inspire-aki report consort|tables|curves|manuscript`
12. `inspire-aki compat export-legacy`

The refactor writes stage outputs and manifests under `artifacts/` instead of relying on implicit handoffs through `/home/server/...`.
The training path is idempotent at the artifact level:

- `train tabular` refreshes `artifacts/predictions/raw/tabular.parquet`
- `train sequence` refreshes `artifacts/predictions/raw/sequence.parquet`
- both rebuild `artifacts/predictions/raw_predictions.parquet`

Legacy alias exports remain explicit through `inspire-aki compat export-legacy`; they are not emitted automatically during `run all`.

### 1. Preoperative extraction

`data_preprocessing/01_extract_preop.py`

- Reads `operations.csv`, `labs.csv`, `diagnosis.csv`, and `ward_vitals.csv`
- Derives:
  - `BMI`
  - `BSA`
  - `booking_case_length`
  - `num_card_events`
  - one-hot department indicators
  - latest preop lab values within 90 days
  - latest ward vitals within 90 days
- Writes `preop_data_test.csv` under `/home/server/Projects/data/AKI/`

### 2. Intraoperative tabular feature engineering

`data_preprocessing/02_extract_intraop.py`

- Reads `vitals.csv`
- Builds:
  - 24 regular signals x 8 summary statistics
  - mean-only sparse variables
  - weight-adjusted and time-adjusted summed variables
  - `fluids_agg`
  - `equiv_MAC_totals`
- Writes `feature_engineered.csv`

### 3. Base dataset creation

`data_preprocessing/03_create_base.py`

- Merges preop and intraop tables
- Removes IDs and obvious leakage columns
- Replaces outliers using percentile windows
- Standardizes numeric features
- Fills high-missingness columns with `-99`
- KNN-imputes lower-missingness columns
- Writes:
  - `tabular_combined.csv`
  - `tabular_preop.csv`
  - `tabular_intraop.csv`
  - `normalization_stats.csv`

### 4. AKI label derivation

`data_preprocessing/04_AKI_data_selection.py`

- Uses preop creatinine plus postoperative 2-day and 7-day windows
- Includes dialysis from `ward_vitals.csv`
- Produces labeled AKI versions of:
  - `tabular_combined.csv`
  - `tabular_preop.csv`
  - `tabular_intraop.csv`

### 5. Sequence preparation

`data_preprocessing/05_time_series_cleaner.py`

- Filters to 24 regular intraoperative signals
- Removes duplicate `(op_id, chart_time, item_name)` rows
- Replaces outliers, interpolates onto 5-minute grids, fills within-op missingness with per-op means
- Writes `time_series_cleaned.csv`

`data_preprocessing/06_create_lstm_trainable.py`

- Merges cleaned time series with labeled tabular data
- Keeps only operations with acceptable time-series presence
- Pads sequences to `200` time steps
- Drops operations longer than that cap
- Writes `lstm_trainable.pkl`

### 6. Modeling

- Tabular HPO: `create_results/07_tabular_hpo.py`
- Tabular training: `create_results/08_tabular_model_creation.py`
- Deep HPO: `create_results/09_lstm_hpo.py`
- Deep training: `create_results/10_lstm_model_creation.py`

### 7. Evaluation and manuscript outputs

- Cohort diagram: `create_results/11_consort.ipynb`
- Cohort characteristics: `create_results/12_cohort_characteristics.ipynb`
- Calibration, F2 thresholding, CI bootstraps, reclassification, DCA: `create_results/13_performance_metrics.ipynb`
- DeLong testing: `create_results/14_delong_table.ipynb`
- SHAP for XGBoost combined model: `create_results/15_shap.ipynb`
- Batch SHAP for multiple models/datasets: `create_results/16_shap_batch.ipynb`

## Data and Portability

### Private inputs expected by the code

- `/home/server/Projects/data/INSPIRE/physionet.org/files/inspire/1.3/operations.csv`
- `/home/server/Projects/data/INSPIRE/physionet.org/files/inspire/1.3/labs.csv`
- `/home/server/Projects/data/INSPIRE/physionet.org/files/inspire/1.3/vitals.csv`
- `/home/server/Projects/data/INSPIRE/physionet.org/files/inspire/1.3/diagnosis.csv`
- `/home/server/Projects/data/INSPIRE/physionet.org/files/inspire/1.3/ward_vitals.csv`

### What is actually checked in

- checked-in result summaries:
  - `create_results/performance_table.md`
  - `create_results/performance_table_calibrated.md`
  - `create_results/descriptive_table.html`
  - `create_results/fill_rate_table.html`
  - `create_results/reclassification_report.html`
- checked-in model/artifact directories:
  - `AutogluonModels/`
  - `notebooks/mljar_results_improved/`

### Practical portability status

| Task | Main entrypoint | Status |
| --- | --- | --- |
| Read repo structure and checked-in findings | this README + `docs/` | runnable from repo only |
| Rebuild cohort/features from raw INSPIRE data | `data_preprocessing/01`-`06` | requires private data and matching `/home/server/...` layout |
| Re-run model training | `create_results/07`-`10` | requires private data plus the pinned environment in `environment.yml` |
| Re-read checked-in performance results | `create_results/performance_table*.md` | runnable from repo only |
| Re-run figure notebooks | `create_results/*.ipynb` | depends on prior outputs and server path layout |

## Main Outputs and Findings

The checked-in outputs suggest the repo was used to create:

- Figure-style outputs:
  - cohort flow / consort
  - ROC and PR curves
  - calibration curves
  - decision-curve analysis
  - SHAP beeswarms, waterfalls, dependence/scatter plots
- Table-style outputs:
  - cohort characteristics
  - model performance
  - calibrated performance
  - fill-rate table
  - reclassification report
  - DeLong comparison tables

The checked-in performance tables support the manuscript-level story:

- preop structured data is already very strong
- low-frequency intraoperative data adds modest value
- the combined tabular models outperform the hybrid deep sequence path

## Documentation Set

Start with:

- [AGENTS.md](AGENTS.md)
- [docs/README.md](docs/README.md)

Deep dives:

- [docs/01_repo_map.md](docs/01_repo_map.md)
- [docs/02_data_pipeline.md](docs/02_data_pipeline.md)
- [docs/03_labels_and_features.md](docs/03_labels_and_features.md)
- [docs/04_modeling_and_evaluation.md](docs/04_modeling_and_evaluation.md)
- [docs/05_artifacts_and_outputs.md](docs/05_artifacts_and_outputs.md)
- [docs/06_notebook_index.md](docs/06_notebook_index.md)
- [docs/07_manuscript_alignment.md](docs/07_manuscript_alignment.md)
- [docs/08_reproducibility_and_known_gaps.md](docs/08_reproducibility_and_known_gaps.md)
- [docs/09_codex_workflow.md](docs/09_codex_workflow.md)
- [docs/refactor/behavior_drift.md](docs/refactor/behavior_drift.md)

## Important Caveats

- Do not assume the current repo can be executed end-to-end as-is on a fresh machine.
- Do not assume every notebook is canonical; many are exploratory or legacy.
- Do not assume manuscript counts and current code outputs are fully aligned.
- Use `environment.yml` as the main setup file; `requirements.txt` mirrors its pip-managed dependencies.
- Some portability fixes in the refactor intentionally differ from brittle legacy behavior; see [docs/refactor/behavior_drift.md](docs/refactor/behavior_drift.md).

## Recommended Reading Order

1. [README.md](README.md)
2. [AGENTS.md](AGENTS.md)
3. [docs/README.md](docs/README.md)
4. [docs/02_data_pipeline.md](docs/02_data_pipeline.md)
5. [docs/04_modeling_and_evaluation.md](docs/04_modeling_and_evaluation.md)
6. [docs/07_manuscript_alignment.md](docs/07_manuscript_alignment.md)
7. [docs/08_reproducibility_and_known_gaps.md](docs/08_reproducibility_and_known_gaps.md)
