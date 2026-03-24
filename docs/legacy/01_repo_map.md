# Repo Map

Status tags used below:

- `canonical`: current primary source of repo behavior
- `supporting`: useful, but not the main execution path
- `exploratory`: research/prototyping material
- `obsolete/superseded`: older material replaced by newer files
- `artifact/output`: generated models, tables, or saved analysis outputs

## Top-Level Structure

| Path | Status | Purpose | Notes |
| --- | --- | --- | --- |
| `README.md` | canonical | Human-facing overview | Rewritten as the main entrypoint |
| `AGENTS.md` | canonical | Short agent operating contract | Points into `docs/` |
| `docs/` | canonical | Deeper knowledge base | This folder |
| `environment.yml` | canonical | Preferred reproducible setup file | Conda entrypoint that installs the pinned Python stack |
| `data_preprocessing/` | canonical | Main preprocessing pipeline | Numbered scripts `01`-`06` are the main path |
| `create_results/` | canonical | Main modeling and evaluation layer | Numbered scripts `07`-`10` plus notebooks |
| `data_postprocessing/` | supporting | Older/parallel postprocessing notebooks | Includes legacy DeLong and SHAP notebooks |
| `preoperative_models/` | exploratory | Older single-model scripts and experiments | Useful for history, not the main path |
| `notebooks/` | exploratory | Broader exploratory work | Includes adjacent outcomes and unrelated experiments |
| `AutogluonModels/` | artifact/output | Saved AutoGluon runs | Checked-in model artifacts |
| `notebooks/mljar_results_improved/` | artifact/output | Saved AutoML result trees | Checked-in artifact directory |
| `mlstatkit/` | supporting | Shared DeLong/stat utility copy | Duplicated elsewhere |
| `AKI_data_loader.py` | exploratory | Time-series dataset loader prototype | Not the numbered pipeline |
| `dataloader.py` | exploratory | VitalDB dataset helper | Not central to INSPIRE pipeline |
| `lstm.py` | obsolete/superseded | Older standalone LSTM experiment | Superseded by `09` and `10` |
| `AKI_data_cleaner.py` | obsolete/superseded | Older cleaner helper | Notebook-derived, limited scope |
| `INSPIRE_data_cleaner.py` | exploratory | One-off CSV restructuring script | Not in the main numbered flow |
| `test.py` | exploratory | Minimal scratch script | No formal test suite exists |
| `requirements.txt` | supporting | Pinned pip dependency list | Mirrored by `environment.yml` |
| `=1.4.0` | obsolete/superseded | Stray file | Likely debris from a package install or shell command |
| `\` | obsolete/superseded | Stray file | Likely accidental artifact |

## Canonical Execution Surface

### Preprocessing

| File | Status | Role |
| --- | --- | --- |
| `data_preprocessing/01_extract_preop.py` | canonical | Extract preoperative cohort variables and preop labs/ward vitals |
| `data_preprocessing/02_extract_intraop.py` | canonical | Summarize intraoperative signals into tabular features |
| `data_preprocessing/03_create_base.py` | canonical | Merge, normalize, outlier-handle, and impute base datasets |
| `data_preprocessing/04_AKI_data_selection.py` | canonical | Derive AKI labels and write labeled tabular datasets |
| `data_preprocessing/05_time_series_cleaner.py` | canonical | Prepare cleaned regular intraop time series |
| `data_preprocessing/06_create_lstm_trainable.py` | canonical | Build padded time-series training dataset |

### Modeling and evaluation

| File | Status | Role |
| --- | --- | --- |
| `create_results/07_tabular_hpo.py` | canonical | Optuna HPO for tabular models |
| `create_results/08_tabular_model_creation.py` | canonical | Bootstrap-style tabular model training |
| `create_results/09_lstm_hpo.py` | canonical | Optuna HPO for LSTM/hybrid models |
| `create_results/10_lstm_model_creation.py` | canonical | Bootstrap-style deep model training |
| `create_results/bootstrap_metrics.py` | canonical | Bootstrap CIs for AUROC/AUPRC/F2/Brier/calibration |
| `create_results/decision_curve.py` | canonical | Decision-curve analysis helper |
| `create_results/11_consort.ipynb` | supporting | Cohort diagram notebook with explicit filtering logic |
| `create_results/12_cohort_characteristics.ipynb` | supporting | Cohort table and fill-rate table notebook |
| `create_results/13_performance_metrics.ipynb` | supporting | Calibration, F2 thresholding, ROC/PR/calibration, reclassification, DCA |
| `create_results/14_delong_table.ipynb` | supporting | Pairwise DeLong testing, including FDR-corrected output |
| `create_results/15_shap.ipynb` | supporting | Main XGBoost SHAP workflow |
| `create_results/16_shap_batch.ipynb` | supporting | Batch SHAP jobs across model/dataset combinations |

## Supporting and Adjacent Code

| Path | Status | Notes |
| --- | --- | --- |
| `data_preprocessing/consort_parity.py` | supporting | Script-style descriptive-table and denormalization helper |
| `data_preprocessing/outcomes_data_selection.py` | supporting | Broader outcomes pipeline for MACCE/PNA/PE/PRF/LOS/ICU/mortality |
| `data_preprocessing/MACCE_data_selection.py` | supporting | Simpler MACCE labeling path for the base datasets |
| `data_preprocessing/AKI_data_selection.py` | obsolete/superseded | Zero-byte file |
| `data_postprocessing/delong_justin.ipynb` | supporting | Alternate DeLong implementation and checker notebook |
| `data_postprocessing/delong.ipynb` | obsolete/superseded | Older DeLong notebook |
| `data_postprocessing/performance_metrics_obselete.ipynb` | obsolete/superseded | Older performance notebook |
| `data_postprocessing/shap_analyze.ipynb` | supporting | Rich SHAP analysis notebook with waterfalls and scatter fitting |
| `data_postprocessing/shap_derive.ipynb` | exploratory | Older SHAP derivation notebook |
| `data_postprocessing/shap_derive_gpu.ipynb` | exploratory | GPU SHAP derivation experiments |

## Legacy Model Surface

The `preoperative_models/` directory is useful historical context, but it is not the main current pipeline.

| Path family | Status | Notes |
| --- | --- | --- |
| `preoperative_models/*.py` single-model scripts | exploratory | Older standalone model runners |
| `preoperative_models/bootstrap*.ipynb` | exploratory | Earlier bootstrap workflows |
| `preoperative_models/justin_lstm.ipynb` | supporting | Supplemental deep-model comparison notebook |
| `preoperative_models/aki_experiments/*.ipynb` | exploratory | Model-specific experiments and tests |

## Duplicated Utilities

The same DeLong/stat helper code exists in three places:

- `mlstatkit/toolkit.py`
- `data_postprocessing/mlstatkit/toolkit.py`
- `preoperative_models/mlstatkit/toolkit.py`

Treat these as duplicated copies, not distinct implementations.

## Checked-In Outputs Worth Noticing

| Path | Status | Notes |
| --- | --- | --- |
| `create_results/performance_table.md` | artifact/output | Main checked-in uncalibrated performance summary |
| `create_results/performance_table_calibrated.md` | artifact/output | Main checked-in calibrated performance summary |
| `create_results/descriptive_table.html` | artifact/output | Cohort characteristics output |
| `create_results/fill_rate_table.html` | artifact/output | Fill-rate output |
| `create_results/reclassification_report.html` | artifact/output | Reclassification output |
| `AutogluonModels/` | artifact/output | Saved AutoGluon model directories |
| `notebooks/mljar_results_improved/` | artifact/output | Large AutoML artifact tree |
