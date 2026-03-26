# Artifacts and Outputs

This document separates:

- private raw inputs
- legacy numbered-script outputs
- refactor package artifacts under the configured artifact root
- checked-in outputs already present in the repo

The important distinction is that the legacy pipeline writes into `/home/server/...`, while the refactor writes into the configured artifact root and keeps stage ownership explicit.

## Current Validation Snapshot

As of March 24, 2026:

- `/media/volume/ncs_inspire_data/ncs_aki/artifacts/smoke_hpo/` contains real-data preprocessing outputs plus completed HPO tuning outputs
- the current tuning artifacts now use normalized trial states such as `COMPLETE`, not raw Optuna numeric codes
- the tabular tuning manifests now enumerate both:
  - the per-dataset HPO split parquet
  - the shared tuning outputs under `tuning/`
- the downstream real-data `train -> evaluate -> report` portion of `configs/aki/smoke_hpo.yaml` still needs one clean end-to-end validation run

## Refactor Artifact Contract

The current `src/inspire_aki/` package is designed around staged artifacts plus manifests.

### Refactor artifact root

- default root in the shipped main config: `/media/volume/ncs_inspire_data/ncs_aki/artifacts/default`
- smoke root in the shipped smoke config: `/media/volume/ncs_inspire_data/ncs_aki/artifacts/smoke`
- smoke-HPO root in the shipped smoke-HPO config: `/media/volume/ncs_inspire_data/ncs_aki/artifacts/smoke_hpo`
- the path fragments below are relative to the configured artifact root
- every stage may also write a manifest under:
  - `manifests/`
- tuning now writes:
  - per-dataset tabular manifests such as `manifests/tune_tabular_preop.json`
  - a top-level aggregate manifest `manifests/tune_tabular.json`
  - `manifests/tune_sequence.json`

### Refactor preprocessing artifacts

| Artifact | Typical path under `artifacts/` | Produced by | Notes |
| --- | --- | --- | --- |
| Preop features | `features/preop/preop_features.csv` | `inspire-aki preprocess preop` | Refactor-owned preop feature table |
| Intraop tabular features | `features/intraop/feature_engineered.csv` | `inspire-aki preprocess intraop` | Refactor-owned intraop feature matrix |
| Cleaned time series | `features/timeseries/time_series_cleaned.csv` | `inspire-aki preprocess timeseries` | Refactor-owned long-format cleaned sequence table |
| Combined tabular dataset | `datasets/tabular/tabular_combined.csv` | `inspire-aki preprocess tabular` | Unlabeled combined base |
| Preop tabular dataset | `datasets/tabular/tabular_preop.csv` | `inspire-aki preprocess tabular` | Unlabeled preop base |
| Intraop tabular dataset | `datasets/tabular/tabular_intraop.csv` | `inspire-aki preprocess tabular` | Unlabeled intraop base |
| Unnormalized combined table | `datasets/tabular/tabular_combined_unnormalized.csv` | `inspire-aki preprocess tabular` | Preserved for inspection/export |
| Normalization stats | `datasets/tabular/normalization_stats.csv` | `inspire-aki preprocess tabular` | Saved scaling statistics |
| Fill-rate table | `datasets/tabular/fill_rates.csv` | `inspire-aki preprocess tabular` | Missingness/imputation summary |
| Labeled combined table | `datasets/tabular/tabular_combined_labeled.csv` | `inspire-aki preprocess labels` | Main combined modeling dataset |
| Labeled preop table | `datasets/tabular/tabular_preop_labeled.csv` | `inspire-aki preprocess labels` | Main preop modeling dataset |
| Labeled intraop table | `datasets/tabular/tabular_intraop_labeled.csv` | `inspire-aki preprocess labels` | Main intraop modeling dataset |
| Sequence-ready dataset | `datasets/sequence/lstm_trainable.pkl` | `inspire-aki preprocess sequence` | Padded sequence plus static features |

Refactor invariant:

- `features/intraop/feature_engineered.csv` must not contain `inf` or `-inf`
- `preprocess intraop` now fails if infinite values remain after feature engineering

### Refactor staging artifacts

These are internal implementation details for the adaptive parallel sequence path. They are not manuscript-facing outputs, but they are now expected during runtime.

| Artifact | Typical path under `artifacts/` | Produced by | Notes |
| --- | --- | --- | --- |
| Filtered time-series partitions | `staging/timeseries_filtered/part-*.parquet` | `inspire-aki preprocess timeseries` | Hash-partitioned filtered vitals rows |
| Cleaned time-series partitions | `staging/timeseries_cleaned/part-*.parquet` | `inspire-aki preprocess timeseries` | Cleaned/interpolated per-partition outputs before final concatenation |
| Sequence partitions | `staging/sequence/part-*.pkl` | `inspire-aki preprocess sequence` | Partitioned padded sequence shards before final concatenation |

### Refactor split artifacts

| Artifact | Typical path under `artifacts/` | Produced by | Notes |
| --- | --- | --- | --- |
| Tabular bootstrap splits | `datasets/splits/bootstrap_preop.parquet`, `bootstrap_intraop.parquet`, `bootstrap_combined.parquet` | `inspire-aki train tabular` | Repeated fold-style train/test manifests |
| Sequence bootstrap splits | `datasets/splits/bootstrap_sequence.parquet` | `inspire-aki train sequence` | Sequence train/test manifest |
| Tabular HPO splits | `datasets/splits/hpo_preop.parquet`, `hpo_intraop.parquet`, `hpo_combined.parquet` | `inspire-aki tune tabular` | Single source of truth for tabular HPO splits |
| Sequence HPO split | `datasets/splits/hpo_sequence.parquet` | `inspire-aki tune sequence` | Single source of truth for sequence HPO |

### Refactor model artifacts

| Artifact | Typical path under `artifacts/` | Produced by | Notes |
| --- | --- | --- | --- |
| Tabular model bundle | `models/tabular/<dataset>/<model>/repeat_<r>/fold_<f>/bundle.joblib` | `inspire-aki train tabular` | Canonical saved sklearn/AutoGluon bundle |
| Sequence model bundle | `models/sequence/<model>/repeat_<r>/fold_<f>/bundle.pt` | `inspire-aki train sequence` | Canonical saved Torch bundle |

### Sequence checkpoint contract

Refactor sequence bundles now persist enough metadata to be reloaded through `load_sequence_bundle(...)`, including:

- `model_key`
- `feature_names`
- `time_input_size`
- `lstm_hidden_size`
- `lstm_num_layers`
- `dropout_rate`
- `mlp_dims`
- `mode`
- `scaler`
- `state_dict`
- `metadata`
- `format_version`

### Refactor prediction artifacts

The refactor now treats prediction artifacts as stage-owned partitions plus one combined evaluation view.

| Artifact | Typical path under `artifacts/` | Produced by | Notes |
| --- | --- | --- | --- |
| Tabular raw prediction partition | `predictions/raw/tabular.parquet` | `inspire-aki train tabular` | Replaced on rerun, not appended |
| Sequence raw prediction partition | `predictions/raw/sequence.parquet` | `inspire-aki train sequence` | Replaced on rerun, not appended |
| Combined raw predictions | `predictions/raw_predictions.parquet` | materialized after each training stage | Deterministic union of the raw partitions |
| Calibrated predictions | `predictions/calibrated_predictions.parquet` | `inspire-aki evaluate calibrate` | Derived from the combined raw prediction view |

The combined raw prediction view is deduplicated and sorted so stage reruns are idempotent.

### Refactor tuning, evaluation, and reporting artifacts

| Artifact | Typical path under `artifacts/` | Produced by | Notes |
| --- | --- | --- | --- |
| Tabular best params | `tuning/tabular_best_params.json` | `inspire-aki tune tabular` | Machine-readable best params |
| Sequence best params | `tuning/sequence_best_params.json` | `inspire-aki tune sequence` | Machine-readable best params |
| Tabular trials | `tuning/tabular_trials.parquet` | `inspire-aki tune tabular` | Optional if trials exist |
| Sequence trials | `tuning/sequence_trials.parquet` | `inspire-aki tune sequence` | Optional if trials exist |
| Calibration thresholds | `evaluation/thresholds.csv` | `inspire-aki evaluate calibrate` | One threshold per model/regime/population |
| Metrics by fold | `evaluation/metrics_by_fold.csv` | `inspire-aki evaluate metrics` | Fold-level metric rows |
| Metrics summary | `evaluation/metrics_summary.csv` | `inspire-aki evaluate metrics` | Aggregated metrics |
| Bootstrap CI metrics | `evaluation/metrics_bootstrap_ci.csv` | `inspire-aki evaluate metrics` | Bootstrap summaries when configured |
| DeLong matrix | `evaluation/delong_matrix.csv` | `inspire-aki evaluate delong` | Pairwise AUROC comparison matrix |
| DeLong long table | `evaluation/delong_long.csv` | `inspire-aki evaluate delong` | Long-form pairwise results |
| DeLong FDR-corrected matrix | `evaluation/delong_fdr_corrected.csv` | `inspire-aki evaluate delong` | Benjamini-Hochberg corrected pairwise AUROC comparison matrix |
| DeLong FDR-corrected long table | `evaluation/delong_fdr_corrected_long.csv` | `inspire-aki evaluate delong` | Long-form corrected pairwise results |
| DCA curves | `evaluation/dca_curves.csv` | `inspire-aki evaluate dca` | Decision-curve point-estimate rows |
| DCA bootstrap CI | `evaluation/dca_bootstrap_ci.csv` | `inspire-aki evaluate dca` | Long-form DCA rows with 95% CI bounds and `tau_star` |
| Reclassification summary | `evaluation/reclassification_summary.csv` | `inspire-aki evaluate reclassification` | Stage-owned patient reclassification summary |
| SHAP importance CSVs | `explainability/shap_importance_<dataset>_<model>.csv` | `inspire-aki explain shap` or `report manuscript` | Only for supported SHAP models |
| Report figures | `reports/figures/*` | `inspire-aki report ...` | ROC, PR, calibration, DCA, SHAP, consort outputs in `png` and `svg` |
| Report tables | `reports/tables/*` | `inspire-aki report ...` | Every manuscript-facing table is emitted as `html`, `md`, and `csv` |

Refactor reporting note:

- cohort-characteristics outputs are now built from `tabular_combined_unnormalized.csv` merged with `cohort/aki_labels.csv`, not from the normalized labeled modeling table

### Refactor report contract

- `inspire-aki explain shap` is the SHAP-only path
- `inspire-aki report manuscript` is the top-level manuscript report path
- manuscript section composition is controlled by:
  - `reports.manuscript_sections`
- manuscript supplemental section generation is controlled by:
  - `reports.generate_supplemental_outputs`
- report output formats are controlled by:
  - `reports.table_formats`
  - `reports.figure_formats`
  - `reports.figure_png_dpi`
- SHAP batch composition is controlled by:
  - `reports.shap_jobs`
- legacy alias exports are explicit through:
  - `inspire-aki compat export-legacy`

The refactor does **not** export legacy aliases automatically during `run all`.

Current canonical manuscript-facing outputs include:

- tables:
  - `performance_table.*`
  - `performance_table_calibrated.*`
  - `cohort_characteristics.*`
  - `fill_rate_table.*`
  - `consort_audit.*`
  - `metrics_ci.*`
  - `delong_raw.*`
  - `delong_fdr_corrected.*`
  - `reclassification_report.*`
- figures:
  - `consort.{png,svg}`
  - `roc_curves_<dataset>.{png,svg}`
  - `pr_curves_<dataset>.{png,svg}`
  - `calibration_curves_<dataset>.{png,svg}`
  - `dca_curve_<dataset>_<model>*.{png,svg}`
  - `dca_datasource_comparison_<model>.{png,svg}`

## Private raw inputs expected by the current refactor defaults

| Artifact | Typical path | Produced outside repo | Notes |
| --- | --- | --- | --- |
| INSPIRE operations table | `/media/volume/ncs_inspire_data/ncs_aki/data/inspire/operations.csv` | yes | Required for cohort construction |
| INSPIRE labs table | `/media/volume/ncs_inspire_data/ncs_aki/data/inspire/labs.csv` | yes | Required for creatinine and preop lab extraction |
| INSPIRE vitals table | `/media/volume/ncs_inspire_data/ncs_aki/data/inspire/vitals.csv` | yes | Required for intraop features and sequence path |
| INSPIRE diagnosis table | `/media/volume/ncs_inspire_data/ncs_aki/data/inspire/diagnosis.csv` | yes | Used for cardiovascular history and outcome derivation |
| INSPIRE ward vitals table | `/media/volume/ncs_inspire_data/ncs_aki/data/inspire/ward_vitals.csv` | yes | Used for ward vitals and dialysis flag |

Historical note:

- the original numbered-script workflow was developed against a different server under `/home/server/Projects/data/...`
- the refactor defaults now point at the mounted replacement volume on this instance

## Legacy numbered-script artifacts

These are the main outputs expected by the numbered legacy scripts.

### Intermediate tabular artifacts

| Artifact | Produced by | Expected path | Notes |
| --- | --- | --- | --- |
| Preop extracted table | `data_preprocessing/01_extract_preop.py` | `/home/server/Projects/data/AKI/preop_data_test.csv` | Filename drift exists vs later scripts |
| Intraop feature table | `data_preprocessing/02_extract_intraop.py` | `/home/server/Projects/data/AKI/feature_engineered.csv` | Main intraop tabular feature matrix |
| Base combined table | `data_preprocessing/03_create_base.py` | `/home/server/Projects/data/base/tabular_combined.csv` | Unlabeled normalized base |
| Base preop table | `data_preprocessing/03_create_base.py` | `/home/server/Projects/data/base/tabular_preop.csv` | Unlabeled normalized base |
| Base intraop table | `data_preprocessing/03_create_base.py` | `/home/server/Projects/data/base/tabular_intraop.csv` | Unlabeled normalized base |
| Normalization stats | `data_preprocessing/03_create_base.py` | `/home/server/Projects/data/base/normalization_stats.csv` | Mean/variance saved for denormalization |
| Labeled combined table | `data_preprocessing/04_AKI_data_selection.py` | `/home/server/Projects/data/AKI/tabular_combined.csv` | Main combined training table |
| Labeled preop table | `data_preprocessing/04_AKI_data_selection.py` | `/home/server/Projects/data/AKI/tabular_preop.csv` | Main preop training table |
| Labeled intraop table | `data_preprocessing/04_AKI_data_selection.py` | `/home/server/Projects/data/AKI/tabular_intraop.csv` | Main intraop training table |

### Legacy sequence artifacts

| Artifact | Produced by | Expected path | Notes |
| --- | --- | --- | --- |
| Cleaned intraop time series | `data_preprocessing/05_time_series_cleaner.py` | `/home/server/Projects/data/AKI/time_series_cleaned.csv` | 24-signal cleaned/interpolated sequence table |
| LSTM trainable dataset | `data_preprocessing/06_create_lstm_trainable.py` | `/home/server/Projects/data/AKI/lstm_trainable.pkl` | Merged padded sequence + static feature dataset |

### Legacy training outputs

| Artifact | Produced by | Expected path | Notes |
| --- | --- | --- | --- |
| Tabular HPO summary | `create_results/07_tabular_hpo.py` | `/home/server/Projects/data/AKI/results/tabular_hpo_results.txt` | Text intended for copy/paste of best params |
| Deep HPO summary | `create_results/09_lstm_hpo.py` | `/home/server/Projects/data/AKI/results/hybrid_hpo_results.txt` | Same pattern for LSTM/hybrid |
| Preop test results pickle | `create_results/08_tabular_model_creation.py` | `/home/server/Projects/data/AKI/results/tabular_preop_test.pkl` | Stores model rows plus `base` row |
| Intraop test results pickle | `create_results/08_tabular_model_creation.py` and `10_lstm_model_creation.py` | `/home/server/Projects/data/AKI/results/tabular_intraop_test.pkl` | May contain tabular and LSTM rows |
| Combined test results pickle | `create_results/08_tabular_model_creation.py` and `10_lstm_model_creation.py` | `/home/server/Projects/data/AKI/results/tabular_combined_test.pkl` | May contain tabular and hybrid rows |
| Consolidated LSTM/hybrid pickle | `create_results/10_lstm_model_creation.py` | `/home/server/Projects/data/AKI/results/lstm_hybrid_test_optimized.pkl` | Deep-model specific output |

## Checked-in outputs already in the repo

These are present in version control and can be read without private data.

| Artifact | Path | Type | Notes |
| --- | --- | --- | --- |
| Uncalibrated performance table | `create_results/performance_table.md` | checked-in output | Main quick metric summary |
| Calibrated performance table | `create_results/performance_table_calibrated.md` | checked-in output | Post-calibration metric summary |
| Descriptive table | `create_results/descriptive_table.html` | checked-in output | Cohort characteristic table |
| Fill-rate table | `create_results/fill_rate_table.html` | checked-in output | Missingness/fill output |
| Reclassification report | `create_results/reclassification_report.html` | checked-in output | Patient-level movement summary |
| AutoGluon model dirs | `AutogluonModels/` | checked-in artifact | Saved predictor folders |
| MLJAR output tree | `notebooks/mljar_results_improved/` | checked-in artifact | Saved AutoML runs and reports |

## Figure/table-producing notebooks

| Notebook | Main outputs |
| --- | --- |
| `create_results/11_consort.ipynb` | cohort diagram counts / dot source |
| `create_results/12_cohort_characteristics.ipynb` | descriptive cohort table, fill-rate table |
| `create_results/13_performance_metrics.ipynb` | calibrated pickles, ROC/PR/calibration figures, reclassification output, DCA output |
| `create_results/14_delong_table.ipynb` | DeLong CSVs and formatted tables |
| `create_results/15_shap.ipynb` | SHAP explanation pickles and figure files |
| `create_results/16_shap_batch.ipynb` | batch SHAP outputs across model/dataset combinations |

## Source vs artifact guidance

When reading the repo:

- treat `*.py` and curated notebooks as source
- treat `AutogluonModels/` and `notebooks/mljar_results_improved/` as artifacts
- treat checked-in markdown/html tables as evidence of prior runs, not guarantees of fresh reproducibility
- treat `artifacts/` as the refactor-owned runtime surface
- treat `/home/server/...` outputs as the legacy numbered-script runtime surface

## Notable output drift

- Some scripts write to `/home/server/Projects/data/base/` while later steps read from `/home/server/Projects/data/AKI/`.
- Some notebooks expect files like `tabular_combined_unnormalized.csv` that are not generated by the canonical numbered path.
- The current repo therefore exposes multiple output layers, not one perfectly linear artifact chain.
- The refactor is cleaner than the legacy chain, but it is still not proof of exact real-data parity with the checked-in historical outputs.
