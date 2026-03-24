# Notebook Index

Status tags:

- `canonical`: current main notebook entrypoint for a documented repo task
- `supporting`: useful adjunct to the main flow
- `exploratory`: research/prototyping notebook
- `obsolete/superseded`: older notebook replaced by a newer path

## Notebook Inventory

| Notebook | Status | Purpose |
| --- | --- | --- |
| `AKI_data_cleaner.ipynb` | obsolete/superseded | Older preop-cleaning notebook whose logic was partly ported into `01_extract_preop.py` |
| `andrew_test.ipynb` | exploratory | Root-level scratch notebook with ad hoc analysis |
| `create_results/11_consort.ipynb` | canonical | Figure-oriented cohort filtering and consort generation |
| `create_results/12_cohort_characteristics.ipynb` | canonical | Cohort characteristic table and fill-rate output |
| `create_results/13_performance_metrics.ipynb` | canonical | Calibration, F2 thresholding, curves, CI bootstraps, reclassification, and DCA |
| `create_results/14_delong_table.ipynb` | canonical | Pairwise DeLong testing and formatted output tables |
| `create_results/15_shap.ipynb` | canonical | Main SHAP workflow for the combined XGBoost model |
| `create_results/16_shap_batch.ipynb` | supporting | Batch SHAP generation across multiple models and datasets |
| `data_postprocessing/delong.ipynb` | obsolete/superseded | Older DeLong notebook replaced by the create-results path |
| `data_postprocessing/delong_justin.ipynb` | supporting | Alternate DeLong implementation and validation notebook |
| `data_postprocessing/performance_metrics_obselete.ipynb` | obsolete/superseded | Older performance notebook explicitly marked obsolete |
| `data_postprocessing/shap_analyze.ipynb` | supporting | Rich SHAP postprocessing notebook with waterfalls and scatter fitting |
| `data_postprocessing/shap_derive.ipynb` | exploratory | Older SHAP derivation workflow |
| `data_postprocessing/shap_derive_gpu.ipynb` | exploratory | GPU-oriented SHAP derivation and experimentation notebook |
| `data_preprocessing/aki_definition_test.ipynb` | exploratory | Scratch space for AKI definition testing |
| `data_preprocessing/aki_experiments/AKI_calculator.ipynb` | exploratory | Experimental AKI calculation notebook |
| `data_preprocessing/aki_experiments/more_prep_vars.ipynb` | exploratory | Experimental feature-engineering notebook |
| `data_preprocessing/aki_experiments/normalization.ipynb` | exploratory | Normalization experiments |
| `data_preprocessing/aki_experiments/smote.ipynb` | exploratory | SMOTE experiment, contrary to the final class-weighting direction |
| `data_preprocessing/consort_diagram_data.ipynb` | supporting | Legacy notebook with recorded cohort counts and consort dot generation |
| `data_preprocessing/missing_handling_test.ipynb` | supporting | Prototyping notebook for outlier and missing-data handling |
| `data_preprocessing/troubleshooting.ipynb` | exploratory | General debugging notebook |
| `notebooks/INSPIRE_Vasc_AOA.ipynb` | exploratory | Adjacent/side analysis not part of the AKI numbered path |
| `notebooks/INSPIRE_exploratory.ipynb` | exploratory | General INSPIRE exploration notebook |
| `notebooks/INSPIRE_hospital_los.ipynb` | exploratory | Adjacent hospital length-of-stay work |
| `notebooks/INSPIRE_sample_model.ipynb` | exploratory | Sample modeling notebook |
| `notebooks/MOVER_exploratory.ipynb` | exploratory | Different dataset exploration notebook |
| `notebooks/inspire_data_notebook.ipynb` | exploratory | General INSPIRE data inspection |
| `notebooks/inspire_outcomes.ipynb` | exploratory | Broader outcomes analysis outside the main AKI path |
| `preoperative_models/aki_experiments/test_autogluon.ipynb` | exploratory | AutoGluon experiment notebook |
| `preoperative_models/aki_experiments/test_dense_nn.ipynb` | exploratory | Dense neural-net experiment notebook |
| `preoperative_models/aki_experiments/test_ensemble.ipynb` | exploratory | Ensemble experiment notebook |
| `preoperative_models/aki_experiments/test_k_neighbors.ipynb` | exploratory | KNN experiment notebook |
| `preoperative_models/aki_experiments/test_logistic_regression.ipynb` | exploratory | Logistic-regression experiment notebook |
| `preoperative_models/aki_experiments/test_logistic_regression_bs.ipynb` | exploratory | Logistic-regression bootstrap experiment notebook |
| `preoperative_models/aki_experiments/test_logistic_regression_shap.ipynb` | exploratory | Logistic-regression SHAP experiment notebook |
| `preoperative_models/aki_experiments/test_random_forest.ipynb` | exploratory | Random-forest experiment notebook |
| `preoperative_models/aki_experiments/test_smote.ipynb` | exploratory | SMOTE experiment notebook |
| `preoperative_models/aki_experiments/test_svm.ipynb` | exploratory | SVM experiment notebook |
| `preoperative_models/aki_experiments/test_tabpfn.ipynb` | exploratory | TabPFN experiment notebook |
| `preoperative_models/aki_experiments/test_xgboost.ipynb` | exploratory | XGBoost experiment notebook |
| `preoperative_models/andrew_test.ipynb` | exploratory | Scratch notebook in the preoperative model area |
| `preoperative_models/bootstrap.ipynb` | exploratory | Older bootstrap training workflow |
| `preoperative_models/bootstrap_intraop.ipynb` | exploratory | Intraop-specific bootstrap workflow |
| `preoperative_models/bootstrap_justin.ipynb` | supporting | More mature tabular bootstrap experimentation notebook |
| `preoperative_models/bootstrap_lstm.ipynb` | exploratory | LSTM bootstrap experimentation notebook |
| `preoperative_models/justin_lstm.ipynb` | supporting | Supplemental deep-model comparison notebook |

## How To Read This Index

- If you want the current manuscript-facing outputs, start with `create_results/11` through `create_results/16`.
- If you want preprocessing intent and history, check `data_preprocessing/` notebooks second.
- If you want old model experiments, use `preoperative_models/` notebooks only as historical context.
- If a notebook and a numbered `.py` script disagree, prefer the numbered `.py` script unless the notebook is the only place a downstream figure/table is generated.
