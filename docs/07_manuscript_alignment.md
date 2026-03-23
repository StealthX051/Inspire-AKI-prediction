# Manuscript Alignment

This table reconciles manuscript expectations from the supplied brief with current repo behavior. It is intentionally code-first.

| Topic | Manuscript expectation | Current code evidence | Current repo behavior | Confidence | Divergence / risk |
| --- | --- | --- | --- | --- | --- |
| Study framing | End-to-end perioperative AKI prediction pipeline using preop, intraop, and combined data regimes | numbered scripts `01`-`10`, checked-in performance tables | The repo does implement those three regimes and both tabular and deep branches | high | None at the high-level framing |
| AKI label logic | Severe AKI from perioperative creatinine, with some ambiguity about 48h vs 7d wording | `data_preprocessing/04_AKI_data_selection.py` | Binary target is `aki_2 or aki_3`, using both 2-day delta and 7-day ratio/max rules | high | Current docs must describe the exact code formula, not only KDIGO prose |
| Dialysis handling | Brief mentions ambiguity: manuscript mentions dialysis, supplement may de-emphasize it | `data_preprocessing/04_AKI_data_selection.py` | Stage 3 currently includes dialysis via `ward_vitals.csv` `crrt` | high | This is an explicit manuscript/code divergence to document |
| Cohort count | Brief says `49,198` patients | `data_preprocessing/consort_diagram_data.ipynb`, comments in `create_results/13_performance_metrics.ipynb` | Repo surfaces multiple counts: `57,724`, about `67k`, and about `54k` depending on stage | high | No single stable cohort count is enforced in the checked-in repo |
| Preop-only, intraop-only, combined datasets | Core study design uses three data regimes | `03_create_base.py`, `04_AKI_data_selection.py`, `07_tabular_hpo.py`, `08_tabular_model_creation.py` | Three tabular regimes are explicit and used throughout the current scripts | high | None |
| Sequence path | Raw variable-length intraop sequences feed deep models | `05_time_series_cleaner.py`, `06_create_lstm_trainable.py`, `09_lstm_hpo.py`, `10_lstm_model_creation.py` | Sequence path exists, but it pads to `200` and drops longer cases | high | The length cap is an implementation choice not obvious from the manuscript summary |
| Feature summaries | Intraop tabular path should summarize regular signals with eight statistics | `02_extract_intraop.py` | Current code does exactly that for 24 regular signals | high | None |
| Missing-data handling | Brief expects `<30%` style rule discussion, `-99` sentinel, KNN imputation | `03_create_base.py` | Current code uses `>=10%` -> `-99`, `0-10%` -> `KNNImputer`, after normalization | high | Thresholding and ordering may differ from manuscript prose |
| Class imbalance | Brief expects class weighting rather than SMOTE | `07_tabular_hpo.py`, `08_tabular_model_creation.py`, `09_lstm_hpo.py`, `10_lstm_model_creation.py` | Current training paths use class weights / `pos_weight` | high | Experimental SMOTE notebooks exist but are not the canonical path |
| Split logic | Brief flags ambiguity between simple 80/20 and repeated CV | `07_tabular_hpo.py`, `08_tabular_model_creation.py`, `09_lstm_hpo.py`, `10_lstm_model_creation.py` | HPO uses fixed 60/20/20-style train/val/holdout splits; model creation uses repeated fold-style resampling over 25 iterations | high | Needs careful wording; the repo does not use one universal split strategy |
| Model naming drift | Brief notes possible “SVM Ensemble” vs “SVM (Linear)” drift | `08_tabular_model_creation.py`, checked-in performance tables | Current main tabular SVM path is `LinearSVC`; calibrated output table labels it `SVM (Linear)` | high | Any manuscript text implying nonlinear SVM or ensemble should be checked |
| AutoGluon | Brief expects strong tabular performance and no HPO | `07_tabular_hpo.py`, `08_tabular_model_creation.py`, `create_results/performance_table.md` | AutoGluon is excluded from HPO and is the strongest checked-in combined model | high | Current code branch likely has a sample-weight column bug, so “current code” and “historical results” are slightly out of sync |
| Deep model performance | Brief expects LSTM/Transformer/TCN exploration and weaker hybrid performance | `preoperative_models/justin_lstm.ipynb`, `09_lstm_hpo.py`, `10_lstm_model_creation.py`, `create_results/performance_table.md` | The main maintained deep path is LSTM-only and hybrid; checked-in combined hybrid AUROC is `0.825` | high | Transformer/TCN work exists mostly in supporting notebooks, not the main current scripts |
| Calibration and F2 thresholding | Brief expects isotonic calibration and fixed F2-optimal thresholds | `create_results/13_performance_metrics.ipynb`, `create_results/bootstrap_metrics.py`, `create_results/decision_curve.py` | Current notebook applies isotonic calibration, reselects F2-optimal thresholds, bootstraps metrics, and generates DCA | high | None |
| DeLong testing | Brief expects formal pairwise AUROC testing | `create_results/14_delong_table.ipynb` | Current notebook performs pairwise DeLong tests and FDR correction for models sharing the same labels | high | None |
| SHAP interpretation | Brief expects beeswarms, waterfalls, threshold scatters, and feature ranking work | `create_results/15_shap.ipynb`, `16_shap_batch.ipynb`, `data_postprocessing/shap_analyze.ipynb` | Current repo contains all of those analysis paths | high | Some SHAP work is spread across legacy notebooks as well as the main notebook |

## Bottom Line

The manuscript brief is directionally accurate, but the repo must be documented with the following code-first realities:

- dialysis is currently included in AKI stage 3
- cohort counts drift across scripts and notebooks
- the sequence path uses a hard length cap and loses some long cases
- the current maintained SVM path is linear
- the current checked-in combined tabular result is stronger than the hybrid deep result
