# Modeling and Evaluation

This document captures the current model-training and evaluation logic from code. It is code-first and intentionally calls out when the current implementation looks incomplete or drifted.

## Current Refactor Contract

The `src/inspire_aki/` refactor now has these package-level guarantees:

- raw training predictions are partitioned by stage under `artifacts/predictions/raw/`
- `artifacts/predictions/raw_predictions.parquet` is rebuilt deterministically from those partitions
- sequence checkpoints now carry enough metadata to be reloaded through `load_sequence_bundle(...)`
- SHAP jobs are validated against the currently supported explainers:
  - `xgb`
  - `rf`
  - `log_reg`
- `report manuscript` is the top-level report command and includes SHAP when `reports.manuscript_sections` contains `shap`
- HPO manifests are authored in pipeline code and passed into the model/HPO layer rather than being rebuilt there

## Main Training Surface

## Tabular HPO

`create_results/07_tabular_hpo.py`

### Models searched

- logistic regression
- XGBoost
- random forest
- linear SVM (`LinearSVC`)
- KNN
- PyTorch MLP

### Split strategy

- single fixed split per dataset during HPO:
  - `80%` train/validation pool
  - `20%` holdout ignored for HPO
  - train/validation pool then split into:
    - `60%` train
    - `20%` validation
    - `20%` holdout

### Target and scaling

- target: `aki_boolean`
- all HPO models use `StandardScaler` on the tabular features before fitting

### Imbalance handling

- logistic regression: `class_weight='balanced'`
- random forest: `class_weight='balanced'`
- linear SVM: `class_weight='balanced'`
- XGBoost: `scale_pos_weight = negatives / positives`
- PyTorch MLP: `BCEWithLogitsLoss(pos_weight=negatives / positives)`

### Search-space notes

- `N_TRIALS = 50`
- objective metric for all HPO paths is validation AUROC except the deep HPO path below
- the script explicitly omits AutoGluon HPO

## Tabular training

`create_results/08_tabular_model_creation.py`

### Supported model branches

- `log_reg`
- `autogluon`
- `xgb`
- `rf`
- `svm`
- `mlp`
- `knn`
- `asa_rule`

### Current default toggles in code

- datasets:
  - `preop = False`
  - `intraop = True`
  - `combined = True`
- models:
  - `autogluon = True`
  - all other branches are currently `False`

This means the checked-in current default script configuration does **not** run the full benchmark matrix.

### Split logic

The script uses a custom `BootstrapSplitter`:

- if bootstrapping is enabled:
  - split the full dataset into five stratified folds
  - evaluate each fold as test once
  - repeat that five-fold cycle with new seeds until `25` iterations total
- if bootstrapping is disabled:
  - use a single stratified `80/20` split

This is better described as repeated fold-style resampling than classical bootstrap sampling with replacement.

### Metrics stored per run

- `roc_auc`
- `balanced_accuracy`
- `f1`
- `recall`
- `precision`
- `specificity`
- raw arrays:
  - `y_true`
  - `y_pred_binary`
  - `y_prob`

### Model-specific notes

- Logistic regression, SVM, KNN, random forest, XGBoost, and MLP use scaled features.
- AutoGluon intentionally uses unscaled raw features.
- `svm` is a `LinearSVC`, not an RBF SVM.
- `asa_rule` is a hard-coded heuristic:
  - `y_prob = asa / 6`
  - `y_pred = asa >= 4`

### Important implementation caveat

The current AutoGluon branch sets `sample_weight='balance_weight'` on `TabularPredictor`, but the script does not add a `balance_weight` column to the training DataFrame before fitting. Treat this as a likely implementation issue in the current code, even if checked-in results came from an earlier working variant.

## Deep HPO

`create_results/09_lstm_hpo.py`

### Modes searched

- `lstm_only`
- `hybrid`

### Inputs

- `lstm_trainable.pkl`
- `tabular_preop.csv`

### Data assembly

- `lstm_only` uses the sequence dataset directly
- `hybrid` merges sequence rows with preop static features on `op_id`
- fixed HPO split:
  - `80%` train/validation pool
  - `20%` holdout unused during HPO
  - train/validation pool then split `75/25`

### Scaling

- tabular features are scaled only for the `hybrid` path
- sequence tensors are assumed preprocessed already

### Objective and stopping

- `N_TRIALS = 50`
- Optuna objective metric is validation `balanced_accuracy`
- class imbalance handled via `BCEWithLogitsLoss(pos_weight=negatives / positives)`
- trial pruning occurs when patience is exceeded or Optuna prunes the trial

## Deep training

`create_results/10_lstm_model_creation.py`

### Modes available

- `lstm_only`
- `mlp_only`
- `hybrid`

### Current default toggles in code

- `lstm_only = True`
- `mlp_only = False`
- `hybrid = True`

### Split logic

- same repeated fold-style splitter design as the tabular training script
- within each train split, create a stratified `15%` validation subset

### Scaling

- tabular features are standardized per run
- time tensors are taken from the saved padded dataset

### Training details

- `lstm_only` and `hybrid` use mini-batched `DataLoader`s
- `mlp_only` uses a full-batch training path
- imbalance handled via `BCEWithLogitsLoss(pos_weight=negatives / positives)`
- early stopping is based on validation loss for LSTM/hybrid and validation AUROC for the MLP-only path

### Output behavior

- main consolidated output:
  - `/home/server/Projects/data/AKI/results/lstm_hybrid_test_optimized.pkl`
- additional writes:
  - intraop results appended to `tabular_intraop_test.pkl` as `lstm`
  - combined results appended to `tabular_combined_test.pkl` as `hybrid`

## Evaluation and Postprocessing

## Calibration and F2 thresholding

`create_results/13_performance_metrics.ipynb`

Confirmed notebook behavior:

- loads original result pickles
- applies isotonic calibration via `IsotonicRegression(out_of_bounds='clip')`
- pools fold outputs into long vectors before calibration/evaluation
- finds a new threshold by maximizing F2 over a grid from `0.01` to `0.99`
- recomputes threshold-dependent metrics on calibrated probabilities
- saves calibrated result pickles
- plots:
  - ROC curves
  - PR curves
  - calibration curves

## Bootstrap confidence intervals

`create_results/bootstrap_metrics.py`

The helper computes bootstrap confidence intervals for:

- AUROC
- AUPRC
- F2
- Brier score
- calibration intercept
- calibration slope

The notebook `create_results/13_performance_metrics.ipynb` imports and uses this helper.

## Reclassification analysis

`create_results/13_performance_metrics.ipynb`

The notebook contains a dedicated reclassification section that:

- reloads per-model predictions
- compares patient-level positive/negative movement after calibration
- writes/uses the checked-in `create_results/reclassification_report.html`

## Decision-curve analysis

- helper: `create_results/decision_curve.py`
- notebook driver: `create_results/13_performance_metrics.ipynb`

The helper computes model net benefit across threshold probabilities and writes:

- DCA plot image
- DCA data table CSV

The plot marks the calibrated F2-optimal threshold `tau_star`.

## DeLong testing

Primary notebook:

- `create_results/14_delong_table.ipynb`

Related helpers/notebooks:

- `mlstatkit/toolkit.py`
- `data_postprocessing/delong_justin.ipynb`
- `data_postprocessing/delong.ipynb`

Confirmed behavior in the main create-results notebook:

- pairwise DeLong comparisons only for models sharing identical ground truth
- raw p-value tables
- Benjamini-Hochberg FDR-corrected p-value tables
- CSV export of the comparison matrix

## SHAP workflows

### Main XGBoost SHAP notebook

`create_results/15_shap.ipynb`

Confirmed behavior:

- trains or loads an XGBoost combined-data model using hard-coded optimized parameters
- computes SHAP values with `shap.TreeExplainer`
- saves SHAP explanation pickles
- creates denormalized SHAP explanation variants
- generates:
  - beeswarm plots
  - waterfall plots
  - scatter/threshold plots
  - cross-feature dependence plots

### Batch SHAP notebook

`create_results/16_shap_batch.ipynb`

Confirmed behavior:

- can run multiple SHAP jobs across datasets and models
- defaults:
  - datasets `preop`, `intraop`, `combined` enabled
  - `xgb`, `rf`, `log_reg`, `svm` enabled
  - `mlp` disabled

### Older SHAP notebooks

- `data_postprocessing/shap_analyze.ipynb`
- `data_postprocessing/shap_derive.ipynb`
- `data_postprocessing/shap_derive_gpu.ipynb`

These provide older and richer exploratory SHAP analysis paths but are not the main current execution entrypoint.

## Checked-In Performance Summary

The checked-in markdown outputs are the easiest stable summary of current findings:

- `create_results/performance_table.md`
- `create_results/performance_table_calibrated.md`

Examples from the uncalibrated table:

- combined AutoGluon AUROC `0.932`
- combined `MLP+LSTM` AUROC `0.825`
- intraop LSTM AUROC `0.739`

These checked-in outputs align with the repo-level conclusion that tabular models outperform the deep sequence branch in this project.
