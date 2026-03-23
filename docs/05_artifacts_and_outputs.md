# Artifacts and Outputs

This document separates source code, expected external artifacts, and checked-in outputs.

## Core Input Layers

## Private raw inputs expected by the code

| Artifact | Typical path | Produced outside repo | Notes |
| --- | --- | --- | --- |
| INSPIRE operations table | `/home/server/Projects/data/INSPIRE/physionet.org/files/inspire/1.3/operations.csv` | yes | Required for cohort construction |
| INSPIRE labs table | `/home/server/Projects/data/INSPIRE/physionet.org/files/inspire/1.3/labs.csv` | yes | Required for creatinine and preop lab extraction |
| INSPIRE vitals table | `/home/server/Projects/data/INSPIRE/physionet.org/files/inspire/1.3/vitals.csv` | yes | Required for intraop features and sequence path |
| INSPIRE diagnosis table | `/home/server/Projects/data/INSPIRE/physionet.org/files/inspire/1.3/diagnosis.csv` | yes | Used for cardiovascular history and outcome derivation |
| INSPIRE ward vitals table | `/home/server/Projects/data/INSPIRE/physionet.org/files/inspire/1.3/ward_vitals.csv` | yes | Used for ward vitals and dialysis flag |

## Intermediate tabular artifacts

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

## Sequence artifacts

| Artifact | Produced by | Expected path | Notes |
| --- | --- | --- | --- |
| Cleaned intraop time series | `data_preprocessing/05_time_series_cleaner.py` | `/home/server/Projects/data/AKI/time_series_cleaned.csv` | 24-signal cleaned/interpolated sequence table |
| LSTM trainable dataset | `data_preprocessing/06_create_lstm_trainable.py` | `/home/server/Projects/data/AKI/lstm_trainable.pkl` | Merged padded sequence + static feature dataset |

## Training outputs expected by the modeling scripts

| Artifact | Produced by | Expected path | Notes |
| --- | --- | --- | --- |
| Tabular HPO summary | `create_results/07_tabular_hpo.py` | `/home/server/Projects/data/AKI/results/tabular_hpo_results.txt` | Text intended for copy/paste of best params |
| Deep HPO summary | `create_results/09_lstm_hpo.py` | `/home/server/Projects/data/AKI/results/hybrid_hpo_results.txt` | Same pattern for LSTM/hybrid |
| Preop test results pickle | `create_results/08_tabular_model_creation.py` | `/home/server/Projects/data/AKI/results/tabular_preop_test.pkl` | Stores model rows plus `base` row |
| Intraop test results pickle | `create_results/08_tabular_model_creation.py` and `10_lstm_model_creation.py` | `/home/server/Projects/data/AKI/results/tabular_intraop_test.pkl` | May contain tabular and LSTM rows |
| Combined test results pickle | `create_results/08_tabular_model_creation.py` and `10_lstm_model_creation.py` | `/home/server/Projects/data/AKI/results/tabular_combined_test.pkl` | May contain tabular and hybrid rows |
| Consolidated LSTM/hybrid pickle | `create_results/10_lstm_model_creation.py` | `/home/server/Projects/data/AKI/results/lstm_hybrid_test_optimized.pkl` | Deep-model specific output |

## Checked-In outputs already in the repo

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

## Figure/Table-Producing Notebooks

| Notebook | Main outputs |
| --- | --- |
| `create_results/11_consort.ipynb` | cohort diagram counts / dot source |
| `create_results/12_cohort_characteristics.ipynb` | descriptive cohort table, fill-rate table |
| `create_results/13_performance_metrics.ipynb` | calibrated pickles, ROC/PR/calibration figures, reclassification output, DCA output |
| `create_results/14_delong_table.ipynb` | DeLong CSVs and formatted tables |
| `create_results/15_shap.ipynb` | SHAP explanation pickles and figure files |
| `create_results/16_shap_batch.ipynb` | batch SHAP outputs across model/dataset combinations |

## Source vs Artifact Guidance

When reading the repo:

- treat `*.py` and curated notebooks as source
- treat `AutogluonModels/` and `notebooks/mljar_results_improved/` as artifacts
- treat checked-in markdown/html tables as evidence of prior runs, not guarantees of fresh reproducibility

## Notable Output Drift

- Some scripts write to `/home/server/Projects/data/base/` while later steps read from `/home/server/Projects/data/AKI/`.
- Some notebooks expect files like `tabular_combined_unnormalized.csv` that are not generated by the canonical numbered path.
- The current repo therefore exposes multiple output layers, not one perfectly linear artifact chain.
