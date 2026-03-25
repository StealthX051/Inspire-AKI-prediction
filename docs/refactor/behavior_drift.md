# Refactor Behavior Drift

This file records intentional or observed differences between the legacy numbered-script path and the refactored `src/inspire_aki/` package path.

It is organized by the refactored CLI stages described in the canonical pipeline map:

- `preprocess ...`
- `tune ...`
- `train ...`
- `evaluate ...`
- `explain ...`
- `report ...`
- `compat ...`

## Preprocess

### `preprocess preop`

| Drift | Legacy behavior | Refactored behavior | Why |
| --- | --- | --- | --- |
| Pandas dtype handling in `antype` | Mutated string columns in place with integer values, which breaks on newer pandas/Arrow-backed strings | Uses explicit `map()` to numeric anesthesia codes | Makes preprocessing portable across newer pandas versions |
| `merge_asof` time columns | Mixed integer and float surgery/lab timestamps depending on CSV inference | Casts surgery timing columns to float before `merge_asof` | Avoids dtype mismatch failures |
| Non-positive operation length | Legacy scripts could allow `op_len <= 0` cases into downstream preprocessing and relied on later `inf -> NaN` cleanup | Refactor now excludes `op_len <= 0` during preop filtering and records the exclusion in audit/manifests | Prevents invalid duration-normalized intraop features at the source |
| ICD-10 prefix exclusion ordering | Legacy scripts removed excluded `icd10_pcs` prefixes only after the preop lab and ward feature merges | Refactor now removes excluded operations before preop feature extraction, after the anesthesia/department merge filters | Avoids wasted `merge_asof` work on rows that will be dropped anyway |
| Preop audit artifact | Legacy extractor printed progress and wrote only the preop feature CSV | Refactor also emits `cohort/preop_audit.csv` plus a stage manifest with runtime and exclusion metadata | Makes cohort-step counts and stage metadata explicit for downstream reporting and handoff |

### `preprocess intraop`

| Drift | Legacy behavior | Refactored behavior | Why |
| --- | --- | --- | --- |
| Summary-stat column names | Legacy output columns used `entropy_*`, `kurtosis_*`, `skew_*`, and `trend_*` prefixes because the raw SciPy/helper function names flowed through `pivot_table` | Refactor currently emits `safe_entropy_*`, `safe_kurtosis_*`, `safe_skew_*`, and `safe_trend_*` because the wrapper names now flow through `pivot_table` | Changes the raw intraop feature schema relative to legacy; this looks incidental rather than scientific |
| Degenerate summary-stat and division handling | Legacy path used raw `entropy`, `kurtosis`, `skew`, `trend`, and direct division by `op_len` or `weight * op_len`, which could yield legacy SciPy values for flat series and could also produce non-finite outputs | Refactor now clamps degenerate summary-stat cases and nonpositive-denominator divisions through safe wrappers, so flat-series features no longer follow raw legacy SciPy behavior, and `preprocess intraop` fails if any `inf` remains | Turns silent artifact-quality problems into explicit invariants, but it also changes some raw feature values for degenerate series |

Real-data note as of March 24, 2026:

- On the mounted default cohort (`configs/aki/default.yaml`), `122,508` filtered operations produced `1,791,002` regular-signal `(op_id, item_name)` groups during intraop tabular feature extraction.
- Those guards are materially exercised on that cohort:
  - `19,755` groups had fewer than `2` finite values
  - `40,167` groups had fewer than `3` finite values
  - `60,334` groups had fewer than `4` finite values
  - `72,484` groups were constant
  - `7,270` groups had near-zero finite sums
  - `9,981` groups contained negative values
- No audited groups had zero finite values after the current preop/cohort filtering, so the empty-series branch was not observed in that audit.

### `preprocess tabular`

| Drift | Legacy behavior | Refactored behavior | Why |
| --- | --- | --- | --- |
| Tabular merge fallback | Refactor initially preserved a risky `subject_id` to `op_id` merge fallback | Tabular dataset assembly now requires `op_id` in both upstream frames and fails clearly otherwise | Removes a silent wrong-join path |
| Outlier-replacement randomness | Legacy `03_create_base.py` used one global `np.random.seed(42)` stream while replacing outliers column by column in serial order | Refactor parallelizes outlier replacement and uses per-column deterministic generators, so replacement draws can differ from legacy even when the percentile-window logic is otherwise the same | Keeps refactor results deterministic under parallel execution, but exact normalized/imputed values can drift from legacy when outlier replacement is active |
| Tabular artifact contract | Legacy base preprocessing wrote only the normalized `tabular_combined.csv`, `tabular_preop.csv`, `tabular_intraop.csv`, and `normalization_stats.csv` outputs | Refactor also emits `tabular_combined_unnormalized.csv` and `fill_rates.csv` during `preprocess tabular` | Makes reporting and missingness inspection explicit without changing the main modeling datasets |

### `preprocess labels`

| Drift | Legacy behavior | Refactored behavior | Why |
| --- | --- | --- | --- |
| Label artifact contract | Legacy `04_AKI_data_selection.py` derived `aki_boolean` and wrote only the labeled combined, preop, and intraop tabular datasets | Refactor also writes standalone `cohort/aki_labels.csv` and `cohort/labels_audit.csv`, plus a stage manifest, before joining labels back onto the tabular datasets under `artifacts/` | Makes label reuse and cohort-step counts explicit for downstream timeseries/sequence filtering, reporting, and handoff |

### `preprocess timeseries` and `preprocess sequence`

| Drift | Legacy behavior | Refactored behavior | Why |
| --- | --- | --- | --- |
| Timeseries outlier replacement | Legacy `05_time_series_cleaner.py` replaced `<=` 1st-percentile and `>=` 99th-percentile values with serial `np.random.uniform(...)` draws during the regular-vitals cleaning pass | Refactor computes per-label quantile windows once, then replaces only values strictly outside those bounds using deterministic row-keyed pseudo-random fractions so sequential and partitioned cleaning agree | Makes the multicore path reproducible, but exact cleaned timeseries values can drift from legacy when replacement is triggered or when values sit exactly on the extreme quantile thresholds |
| Timeseries/sequence staging | Earlier refactor wrote only final `time_series_cleaned.csv` and `lstm_trainable.pkl` outputs | The refactor now also writes internal partitioned staging artifacts under `artifacts/staging/` | Supports safe multicore preprocessing without changing the top-level output contract |
| Sequence tensor representation | Legacy `06_create_lstm_trainable.py` materialized Torch tensors during preprocessing and pickled them directly in `lstm_trainable.pkl` | Refactor stores padded sequence matrices as NumPy arrays in `lstm_trainable.pkl` and converts them to Torch tensors later in the sequence-model code paths | Keeps preprocessing independent of Torch imports and defers framework-specific tensor materialization to training and inference |

## Tune

### `tune tabular` and `tune sequence`

| Drift | Legacy behavior | Refactored behavior | Why |
| --- | --- | --- | --- |
| XGBoost device selection | Legacy HPO and training assumed CUDA in some paths | Refactor only requests CUDA when it is available | Improves portability to CPU-only machines |
| HPO RNG seeding | Legacy tabular and sequence HPO scripts called `np.random.seed(42)` and `torch.manual_seed(42)` before running studies | Refactor passes `random_state` into estimator constructors where available, but does not set global NumPy/Torch seeds before PyTorch-based HPO trials | Keeps the tuning code lighter, but `mlp`, `lstm_only`, and `hybrid` HPO runs are less repeatable run to run; this looks like a reproducibility regression relative to legacy |
| HPO split materialization | Legacy tabular and sequence HPO scripts created one stratified train/validation/holdout split inline with `train_test_split(...)` and did not persist it as an artifact | Refactor builds `datasets/splits/hpo_*.parquet` manifests in the pipeline layer and passes them into model HPO functions | Makes split ownership auditable and resumable, and keeps partitioning outside model-specific code |
| Optimization metric policy | Legacy tabular HPO optimized validation AUROC while legacy sequence HPO optimized validation `balanced_accuracy` | Refactor now optimizes both tabular and sequence HPO on validation `balanced_accuracy` | Aligns the current refactor with the manuscript-facing metric policy across model families, but intentionally diverges from legacy tabular HPO |
| Optuna trial-state handling | Earlier refactor assumed completed trials stringify like `...COMPLETE` | Refactor now normalizes Optuna trial states across enum-style and numeric `4.x` representations before checking or writing them | Fixes false “no completed trials” failures on newer Optuna |
| HPO result artifact contract | Legacy HPO wrote copy-paste text summaries such as `tabular_hpo_results.txt` and `hybrid_hpo_results.txt`; detailed trial history was not persisted as a structured artifact | Refactor writes machine-readable `tabular_best_params.json`, `sequence_best_params.json`, `tabular_trials.parquet`, `sequence_trials.parquet`, and stage manifests under `artifacts/` | Makes tuning outputs auditable, diffable, and reusable by downstream stages without manual transcription |
| Durable tabular study outputs and resume | Legacy tabular HPO had no durable per-study checkpoints; an interruption during a later model family could waste already-finished earlier studies | Refactor now writes one durable study artifact trio per `(dataset_regime, model_key)` under `tuning/tabular_studies/` plus `tune_tabular_<dataset>__<model>.json`, and matching studies resume automatically | Keeps low-CPU HPO interruptions from wasting earlier completed work |
| Low-CPU tabular HPO scheduling | Legacy tabular HPO was serial across datasets and model families | Refactor keeps heavy tabular families serial but runs `svm` HPO concurrently across `preop`, `intraop`, and `combined` study keys | Uses idle CPUs during the low-thread SVM slice without adding a general scheduler |
| Hybrid static-feature source | Legacy `09_lstm_hpo.py` tuned the `hybrid` model by merging sequence rows with `tabular_preop.csv` | Refactor tunes `hybrid` against the static tabular columns already embedded in `lstm_trainable.pkl`, which are built from `tabular_combined_labeled.csv` | Keeps the refactored sequence dataset self-contained, but changes the hybrid HPO input relative to legacy |

## Train

### `train tabular`

| Drift | Legacy behavior | Refactored behavior | Why |
| --- | --- | --- | --- |
| Tabular tuned-parameter handoff | Legacy `08_tabular_model_creation.py` expected HPO results to be pasted back into in-script dictionaries like `hpo_params_preop` before training | Refactor loads `tuning/tabular_best_params.json` automatically when present, otherwise falls back to configured `tabular_hpo_params` or `tabular_defaults` | Removes the manual copy-paste step between tune and train |
| `asa_rule` model routing | Could be enabled even for intraop-only datasets that do not contain `asa` | Skips `asa_rule` where `asa` is absent | Makes model-to-dataset compatibility explicit |
| Imbalance weighting policy | Legacy training code mixed `class_weight='balanced'`, `scale_pos_weight`, `pos_weight`, and an inconsistent AutoGluon `balance_weight` reference | Refactor now materializes explicit inverse-frequency `balance_weight`-style weighting across trainable models, with `knn` applying the same weighting intent through deterministic weighted resampling because `sklearn` KNN lacks `sample_weight` on `fit()` | Makes imbalance handling explicit and consistent across model families |
| AutoGluon GPU budgeting | Legacy `08_tabular_model_creation.py` relied on AutoGluon defaults and did not pass an explicit GPU budget | Refactor passes `models.autogluon.num_gpus` into `TabularPredictor.fit(...)`, with the main default config set to `auto` | Makes GPU use explicit on capable hosts without changing the model family or adding AutoGluon to Optuna HPO |
| Tabular model persistence | Legacy tabular training primarily wrote aggregated result pickles like `results/tabular_{dataset}_test.pkl` and did not persist reusable per-split model bundles | Refactor writes per-split tabular bundles under `models/tabular/.../bundle.joblib` | Makes trained tabular artifacts reusable for inference and downstream analysis |
| Low-CPU tabular train scheduling | Legacy tabular training fit every model/fold serially inside one process | Refactor keeps most tabular models serial but runs `svm` training concurrently at the repeat level within each dataset regime, while leaving `log_reg` serial with a moderate BLAS cap | Reduces the repeated-CV SVM wall clock without widening concurrency across every model family |
| SVM training convergence | Legacy `08_tabular_model_creation.py` trained `LinearSVC(..., max_iter=5000, tol=0.01)` | Refactor now aligns `train tabular` SVM to `tol=0.01` while leaving SVM HPO tolerance unchanged | Restores the legacy faster training convergence without changing the HPO search contract |

### `train tabular` and `train sequence`

| Drift | Legacy behavior | Refactored behavior | Why |
| --- | --- | --- | --- |
| Training RNG seeding | Legacy tabular and sequence training scripts called `np.random.seed(42)` and `torch.manual_seed(42)` before running model loops | Refactor passes per-model `seed` or `random_state` into many estimators, but does not set global NumPy/Torch seeds before the PyTorch MLP/LSTM/hybrid training loops | Keeps the training code lighter, but PyTorch-backed training is less repeatable run to run; this looks like a reproducibility regression relative to legacy |
| Optimization monitor policy | Legacy tabular MLP training monitored validation AUROC, while legacy LSTM/hybrid training monitored validation loss | Refactor now uses validation `balanced_accuracy` for the PyTorch early-stopping and scheduler decisions in both tabular MLP and sequence training | Aligns optimization with the current manuscript-facing metric policy, but intentionally diverges from the legacy training monitors |
| Bootstrap split materialization | Legacy training scripts generated bootstrap or holdout splits inline inside `BootstrapSplitter` and did not persist them | Refactor writes `datasets/splits/bootstrap_{preop,intraop,combined,sequence}.parquet` manifests and uses them as the split authority for training | Makes split ownership auditable and resumable across stages |
| Prediction artifact contract | Legacy training scripts wrote aggregated result pickles such as `tabular_{dataset}_test.pkl` and `lstm_hybrid_test_optimized.pkl`, including synthetic `base` or `base_54k` rows in those files | Refactor writes stage-owned raw prediction partitions plus a deterministic combined `predictions/raw_predictions.parquet` view, and leaves baseline handling to downstream evaluation/reporting code | Separates training from evaluation/reporting and makes reruns idempotent |

### `train sequence`

| Drift | Legacy behavior | Refactored behavior | Why |
| --- | --- | --- | --- |
| Sequence tuned-parameter handoff | Legacy `10_lstm_model_creation.py` expected HPO results to be pasted into in-script dictionaries like `hpo_params_lstm_only` and `hpo_params_hybrid` before training | Refactor loads `tuning/sequence_best_params.json` automatically when present, otherwise falls back to configured `sequence_hpo_params` or `sequence_defaults` | Removes the manual copy-paste step between tune and train |
| Hybrid static-feature source | Legacy `10_lstm_model_creation.py` trained `hybrid` by merging sequence rows with `tabular_preop.csv` | Refactor trains `hybrid` against the static tabular columns already embedded in `lstm_trainable.pkl`, which are built from `tabular_combined_labeled.csv` | Keeps the refactored sequence dataset self-contained, but changes the hybrid model input relative to legacy |
| Sequence model persistence | Legacy sequence training primarily wrote evaluation result pickles and did not persist reusable sequence bundles with enough metadata to reconstruct the model | Refactor writes `models/sequence/.../bundle.pt` with architecture metadata, scaler state, and `state_dict`, and reloads it through `load_sequence_bundle(...)` | Makes saved sequence artifacts reusable for inference and downstream analysis |

## Evaluate

### `evaluate calibrate`, `evaluate metrics`, `evaluate delong`, and `evaluate dca`

| Drift | Legacy behavior | Refactored behavior | Why |
| --- | --- | --- | --- |
| Calibration scope and input contract | Legacy `13_performance_metrics.ipynb` calibrated per-dataset result pickles like `tabular_preop_test.pkl`, `tabular_intraop_test.pkl`, and `tabular_combined_test.pkl`, and recovered labels from embedded `base` or `base_54k` rows | Refactor calibrates every `(dataset_regime, population_id, model_key)` group from `predictions/raw_predictions.parquet`, using explicit row-level `y_true` values and including sequence groups when present | Decouples calibration from the legacy pickle schema and makes the stage generic across tabular, sequence, and multiple populations |
| Calibration output and metric staging | Legacy calibration rewrote `*_calibrated.pkl` files containing recalibrated probabilities, binary predictions, thresholds, and updated threshold-dependent metrics in one artifact | Refactor `evaluate calibrate` writes `predictions/calibrated_predictions.parquet` plus `evaluation/thresholds.csv`, and leaves threshold-dependent metric recomputation to `evaluate metrics` | Splits evaluation into stage-owned artifacts instead of coupling calibration and metric summarization |
| Calibration CV grouping | Legacy notebook calibration pooled repeated predictions and split them with plain row-wise CV, so the same `op_id` could land in both calibration-train and calibration-test across repeats | Refactor now calibrates with grouped CV on `op_id`, preferring `StratifiedGroupKFold` and falling back to `GroupKFold` when grouped stratification is infeasible | Removes repeated-row leakage during isotonic calibration while preserving one threshold per prediction group |
| Sparse-group fallback | Legacy calibration always attempted 5-fold isotonic calibration and F2 threshold search, with no explicit guard for single-class or too-small pooled predictions | Refactor falls back to identity calibration and threshold `0.5` when a prediction group has fewer than 2 classes or fewer than 2 rows for cross-validation | Keeps smoke runs and narrow prediction groups from failing during evaluation |
| Summary metric aggregation | Legacy performance tables in `13_performance_metrics.ipynb` summarized per-fold metric arrays from the result pickles and reported means plus t-based confidence intervals | Refactor `metrics_summary.csv` computes one pooled patient-level metric row per `(dataset_regime, population_id, model_key)`, and `reporting/tables.py` renders those pooled values directly | Simplifies stage-owned reporting, but the reported point estimates can drift from the legacy tables |
| AUPRC definition | Legacy calibrated performance-table paths stored `pr_auc` from `precision_recall_curve(...)` plus trapezoidal `auc(recall, precision)`, while the bootstrap helper used `average_precision_score(...)` | Refactor standardizes both summary and bootstrap AUPRC on `average_precision_score(...)` | Removes the legacy inconsistency, but changes AUPRC values relative to the legacy calibrated markdown tables |
| Bootstrap metric artifact contract | Legacy bootstrap analysis wrote `metrics_ci.csv` with AUROC, AUPRC, F2, Brier score, calibration intercept, and calibration slope only | Refactor writes `metrics_bootstrap_ci.csv`, adds balanced accuracy, accuracy, recall, specificity, precision, and F1, renames calibration metrics to `calib_intercept` and `calib_slope`, and uses `mean` / `ci_lower_95` / `ci_upper_95` columns | Makes the bootstrap output broader and stage-owned |
| Bootstrap resampling stream | Legacy bootstrap CI generation seeded each resample through a master RNG that generated one worker seed per iteration | Refactor bootstrap CI generation uses chunk-local RNG streams, with one sequential generator for single-process runs and per-batch seeds for parallel batches | Keeps the stage deterministic under the refactored joblib execution model, but exact CI values can drift slightly even with the same top-level seed |
| Bootstrap calibration-fit compatibility | Legacy `bootstrap_metrics.py` estimated calibration intercept/slope with `LogisticRegression(penalty="none", solver="lbfgs")` | Refactor uses a very-high-`C` logistic regression (`C=1e12`, `solver="lbfgs"`, `max_iter=1000`) for the same estimates | Preserves the intended near-unregularized fit under newer scikit-learn behavior |
| Single-class balanced accuracy | Legacy fold-level metric collection kept `balanced_accuracy_score(...)` even when a fold or pooled vector contained only one class | Refactor sets `balanced_accuracy` to `NaN` whenever a metric group has fewer than 2 classes | Avoids emitting a value for a degenerate group, but differs from legacy sparse-fold outputs |
| DeLong scope and exclusions | Legacy `14_delong_table.ipynb` loaded tabular result pickles for `preop`, `intraop`, and `combined`, translated model names, and manually excluded `preop_ASA Rule`, `combined_ASA Rule`, and `combined_Hybrid (MLP + LSTM)` before pairwise testing | Refactor runs DeLong on every calibrated `(dataset_regime, population_id, model_key)` prediction group with no built-in model translation or exclusion list | Makes the stage generic across tabular, sequence, and multiple populations, but changes which models appear in the comparison set |
| DeLong multiple-testing correction | Legacy `14_delong_table.ipynb` produced both raw p-value tables and a Benjamini-Hochberg FDR-corrected table / CSV | Refactor writes only raw `evaluation/delong_matrix.csv` and `evaluation/delong_long.csv` from `evaluate delong`, with no FDR-correction pass | Keeps the stage simpler and machine-readable, but drops the legacy corrected significance view |
| DeLong artifact shape | Legacy DeLong output was mainly matrix-style notebook tables and CSV exports, with abbreviated display names such as `p_`, `i_`, and `c_` | Refactor writes a matrix using raw stage model ids like `combined_log_reg` plus a long-form table containing `model_left`, `model_right`, `auc_left`, `auc_right`, and `p_value` | Makes downstream consumption easier without notebook-specific formatting |
| DeLong population-id collision | Legacy DeLong comparisons used dataset-prefixed model names and had no separate `population_id` dimension to preserve | Refactor prepares groups by `(dataset_regime, population_id, model_key)` but stores them under `dataset_regime_model_key`, so distinct populations with the same regime/model overwrite each other before comparison | Current table naming omits `population_id`, which can silently drop comparisons when multiple populations share the same regime/model |
| DCA scope | Legacy `decision_curve.py` was driven from `13_performance_metrics.ipynb` for tabular `preop`, `intraop`, and `combined` models only | Refactor runs DCA for every calibrated `(dataset_regime, population_id, model_key)` prediction group | Makes the stage generic across tabular, sequence, and multiple populations |
| DCA artifact and plotting contract | Legacy `decision_curve.py` generated one per-model DCA plot plus one per-model DCA CSV, bootstrapped 95% CI bands, and marked the calibrated F2-optimal threshold `tau_star` on the plot | Refactor `evaluate dca` writes a single combined `evaluation/dca_curves.csv` with point-estimate net-benefit rows only; downstream report plotting consumes those rows without bootstrap CI bands or a `tau_star` marker | Splits raw curve calculation from reporting and simplifies the stage-owned artifact surface, but drops the legacy uncertainty band and threshold-marker view |
| DCA threshold grid endpoint | Legacy `decision_curve.py` used `np.arange(0.01, 0.305, 0.005)`, which stopped at `0.30` | Refactor builds the threshold grid from config and includes the configured max endpoint, so the default grid runs through `0.305` | Makes the configured max inclusive, but adds one extra threshold row relative to the legacy helper |
| Non-nested evaluation design | Legacy numbered scripts tuned once on the cohort and later evaluated tuned models through repeated CV on that same cohort | Refactor keeps that same non-nested design: HPO runs once before repeated-CV training/evaluation instead of nesting tuning inside each outer fold | Preserves legacy runtime and workflow shape, but the resulting evaluation remains optimistic relative to a fully nested design |

## Explain

### `explain shap`

| Drift | Legacy behavior | Refactored behavior | Why |
| --- | --- | --- | --- |
| SHAP/report config | Refactor initially allowed `batch_shap_jobs` drift and silently skipped unsupported jobs like `svm` | Config now normalizes to `reports.shap_jobs`, validates supported models, and fails fast on unsupported jobs | Removes silent partial reporting |
| SHAP model coverage | Legacy SHAP workflows were notebook-based: `15_shap.ipynb` focused on combined-data XGBoost, while `16_shap_batch.ipynb` could run `xgb`, `rf`, `log_reg`, `svm`, and `mlp` jobs across `preop`, `intraop`, and `combined` datasets | Refactor supports only `xgb`, `rf`, and `log_reg` via `SUPPORTED_SHAP_MODELS` and rejects the other legacy notebook model families | Narrows the maintained SHAP surface to the model families with first-class refactor support |
| SHAP model and split source | Legacy SHAP notebooks trained or loaded notebook-local models on an explicit single `80/20` stratified split and then explained that test set | Refactor reuses the saved `repeat_0/fold_0` training bundle plus the matching split manifest from the train stage, so default SHAP explanations are tied to the first staged training split rather than a dedicated notebook split | Keeps SHAP tied to stage-owned artifacts and avoids retraining inside reporting |
| SHAP explained sample size | Legacy main and batch SHAP notebooks explained the full test split for each job | Refactor caps the background sample at `200` rows and the explained test sample at `200` rows before computing SHAP values | Keeps report-time SHAP jobs cheaper and more predictable on large tabular datasets, but feature rankings and beeswarm distributions can drift from the legacy full-test explanations |
| SHAP artifact and plot surface | Legacy SHAP notebooks saved pickled `shap.Explanation` objects, including denormalized and probability-space XGBoost variants, and generated beeswarm, waterfall, interaction-scatter, and cross-feature dependence plots | Refactor writes only `explainability/shap_importance_<dataset>_<model>.csv` plus one beeswarm-style PNG per SHAP job | Reduces the maintained SHAP surface to the manuscript-facing summary artifacts rather than the full exploratory notebook suite |

## Report

### `report consort`

| Drift | Legacy behavior | Refactored behavior | Why |
| --- | --- | --- | --- |
| Consort count source and step semantics | Legacy `11_consort.ipynb` recomputed cohort counts directly from raw `operations.csv`, `labs.csv`, `ward_vitals.csv`, and intraop feature availability, then emitted manuscript-oriented steps with explicit exclusion reasons such as missing intraop variables and missing post-op outcome | Refactor `report consort` concatenates stage-owned `cohort/preop_audit.csv` and `cohort/labels_audit.csv`, exposing lower-level audit steps like `asa_lt_6`, `has_opend_time`, and `preop_creatinine_lt_threshold` instead of the legacy manuscript-style step names and reasons | Reuses existing stage audits instead of re-running raw-data cohort logic during reporting |
| Consort intraop and final AKI split coverage | Legacy `11_consort.ipynb` included an explicit `Operations after excluding missing intraoperative variables` step and appended final negative/positive AKI case counts for the terminal split | Refactor consort output stops at the concatenated audit rows and does not currently add a dedicated missing-intraop step label or the final negative/positive AKI split rows | Keeps the report stage lightweight, but the resulting output is less faithful to the legacy manuscript-facing diagram |
| Consort DOT graph contract | Legacy `11_consort.ipynb` wrote a vertical DOT graph with exclusion annotation nodes and a final branch to negative / positive AKI cases | Refactor writes a simple left-to-right linear DOT chain, plus CSV / Markdown audit tables | Simplifies the graph source and adds machine-readable audit tables |

### `report tables`

| Drift | Legacy behavior | Refactored behavior | Why |
| --- | --- | --- | --- |
| Performance-table contract | Legacy `13_performance_metrics.ipynb` wrote manuscript-style performance tables with `preop` / `intraop` / `combined` sections and one point-estimate row plus one CI row per model, and the checked-in repo also preserves a separate calibrated Markdown table | Refactor `reporting/tables.py` writes `evaluation/metrics_summary.csv` directly to CSV / Markdown, retaining native keys like `dataset_regime`, `population_id`, `model_key`, `n_rows`, and `threshold`, and currently emits only a single performance table with no CI rows | Keeps the report stage thin over stage-owned metrics, but the output is less manuscript-ready and collapses the legacy calibrated / uncalibrated table split |
| Cohort-characteristics scope and granularity | Legacy `12_cohort_characteristics.ipynb` grouped to one row per `subject_id`, built a single overall Table 1, and included rows such as female sex, postoperative AKI prevalence, ASA class counts, and department counts | Refactor builds cohort characteristics from `tabular_combined_unnormalized.csv` merged with labels on `op_id` and summarizes only a short candidate feature list into `overall`, `aki_negative`, and `aki_positive` columns | Reuses refactor-owned tabular artifacts and provides a quick stratified descriptive summary instead of reproducing the manuscript-specific table layout |
| Cohort-characteristics row semantics | Legacy `12_cohort_characteristics.ipynb` formatted binary rows as labeled counts plus percentages, such as `Female sex, n (%)`, and used pandas default sample SD for continuous summaries | Refactor `_cohort_characteristics(...)` prints plain percentages for binary columns, formats numeric rows as `mean +/- population SD`, and the `sex` row currently reports the share of `sex == True` under the generic label `sex` | The helper is generic rather than domain-aware; the `sex` labeling drift looks accidental rather than intentional |
| Fill-rate table shaping | Legacy `12_cohort_characteristics.ipynb` built the fill-rate table from `_isna` columns, consolidated repeated regular-signal features to one row per physiologic label, stripped prefixes like `mean_` / `sum_` / `max_` / `min_`, and rendered a titled manuscript-style HTML table | Refactor writes raw `features/fill_rates.csv` through `DataFrame.to_html`, exposing every feature row, including identifiers and department dummies, with no consolidation or manuscript-specific formatting | Reuses the preprocess-stage missingness artifact instead of reproducing the notebook-specific presentation logic |

### `report curves`

| Drift | Legacy behavior | Refactored behavior | Why |
| --- | --- | --- | --- |
| ROC / PR curve construction | Legacy `13_performance_metrics.ipynb` built mean ROC and PR curves from per-fold prediction arrays, optionally shaded them with foldwise standard-deviation bands, sorted legends by mean AUC, and labeled each curve with `mean ± std` AUC text | Refactor `reporting/curves.py` plots one pooled patient-level ROC or PR curve per model from `predictions/calibrated_predictions.parquet`, with no foldwise interpolation, no shaded uncertainty bands, and no legend statistics | Reuses the row-level prediction artifact directly instead of rebuilding notebook-specific fold summaries during reporting |
| Calibration curve construction | Legacy `13_performance_metrics.ipynb` pooled per-model predictions from the result pickles and called `calibration_curve(..., n_bins=15, strategy='uniform')` on the stored model probabilities | Refactor plots calibration curves from row-level predictions using calibrated probabilities when present, falling back to raw probabilities otherwise, and calls `calibration_curve(..., n_bins=10, strategy='quantile')` | Keeps the report stage aligned with calibrated prediction artifacts and uses quantile bins to avoid sparse-bin shapes, but the plotted curves can drift from the legacy figures |
| Curve artifact scope and layout | Legacy curve generation wrote ROC / PR / calibration plots under `figures/curves/`, and the DCA workflow additionally wrote one per-model DCA figure plus cross-dataset DCA comparison figures under `figures/dca/` | Refactor writes one ROC, PR, and calibration figure per `dataset_regime`, and writes DCA figures per `(dataset_regime, population_id)` under `reports/figures/` | Keeps the report stage generic over the stage-owned prediction artifacts while avoiding mixed-population DCA overlays whose Treat-All reference lines depend on cohort prevalence |

### `report manuscript`

| Drift | Legacy behavior | Refactored behavior | Why |
| --- | --- | --- | --- |
| Manuscript reporting | Refactor initially required a separate SHAP stage even when the user asked for `report manuscript` | `report manuscript` now dispatches sections from `reports.manuscript_sections` and includes SHAP by default | Makes the report contract truthful |

## Compat

### `compat export-legacy`

| Drift | Legacy behavior | Refactored behavior | Why |
| --- | --- | --- | --- |
| Legacy export toggle | Refactor config initially carried an unused `compat.export_legacy_aliases` flag | Legacy alias export is now explicit through `compat export-legacy` only | Removes dead config surface |

## Cross-Cutting Runtime and Config

These changes affect multiple stages rather than only one CLI command.

| Drift | Legacy behavior | Refactored behavior | Why |
| --- | --- | --- | --- |
| Optional heavy deps | Importing the package could fail immediately if `torch`, `xgboost`, `optuna`, or `shap` were missing | Heavy dependencies are imported lazily in the code paths that actually need them | Keeps the CLI usable for lighter tabular-only/test scenarios |
| Default raw-data path | Earlier refactor config still pointed at the broken `/home/server/Projects/data/INSPIRE/...` layout | Refactor defaults now point at `/media/volume/ncs_inspire_data/ncs_aki/data/inspire` on the mounted replacement volume | Aligns the package defaults with the current instance layout |
| Worker/thread policy | Earlier refactor still relied on an ad hoc `cpu_count() - 2` helper in several places | The refactor now resolves stage-specific runtime plans from detected CPU, RAM, and GPU resources, defaults the main AKI config to `runtime.profile: throughput`, and records the resolved plan in manifests | Makes large-node execution faster while keeping one concurrency authority |
| `run all` observability and overlap | Legacy scripts printed progress continuously, while the earlier refactor `run all` emitted only a final JSON blob after every stage finished serially | `run all` now emits immediate stage start/end events to stdout and `artifacts/logs/run_all_events.jsonl`, and in `runtime.orchestration.mode: overlap` it overlaps `tune sequence` with `train tabular` after `tune tabular` | Restores live monitoring and improves whole-node utilization without adding a general scheduler |
| Sequence tensor loader policy | Earlier refactor sequence HPO/training used multiprocessing `DataLoader` workers on already-materialized tensors and hardcoded HPO batch size `512` | Refactor now defaults tensor-backed sequence HPO/training to `num_workers=0`, enables persistent workers only when explicitly overridden above zero, and makes sequence-HPO batch size configurable through `models.hpo.sequence_batch_size` | Reduces loader churn on the current host and makes the HPO runtime path easier to benchmark and tune |

## Preserved-on-Purpose Behaviors

These behaviors still match the legacy scientific or pipeline contract, even if they remain questionable.

### Label and cohort behavior

- AKI stage 3 still includes dialysis from `ward_vitals.csv` `crrt`.

### Tabular preprocessing behavior

- Base preprocessing still standardizes before imputation.
- High-missingness columns still get `-99`, with KNN imputation for lower missingness.

### Sequence behavior

- Sequence preparation still defaults to the legacy-equivalent capped path.

### Evaluation behavior

- Calibration still uses isotonic regression with flat cross-validated probabilities and a single F2-optimal threshold per model/data regime.

## Follow-Up Candidates

### Preprocessing

- Revisit whether normalization-before-imputation should remain the default.

### Sequence path

- Decide whether sequence preprocessing should keep dropping longer cases or move entirely to loader-side truncation/padding.
