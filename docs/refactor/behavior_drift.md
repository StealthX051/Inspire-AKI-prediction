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
| Raw intraop feature validity | Legacy and early refactor paths could emit `inf` values in `feature_engineered.csv` from degenerate entropy or duration-based division | Refactor now uses safe summary-stat wrappers, guards nonpositive denominators, and fails the stage if any `inf` remains | Turns a silent artifact-quality problem into an explicit invariant |

### `preprocess tabular`

| Drift | Legacy behavior | Refactored behavior | Why |
| --- | --- | --- | --- |
| Tabular merge fallback | Refactor initially preserved a risky `subject_id` to `op_id` merge fallback | Tabular dataset assembly now requires `op_id` in both upstream frames and fails clearly otherwise | Removes a silent wrong-join path |

### `preprocess timeseries` and `preprocess sequence`

| Drift | Legacy behavior | Refactored behavior | Why |
| --- | --- | --- | --- |
| Timeseries/sequence staging | Earlier refactor wrote only final `time_series_cleaned.csv` and `lstm_trainable.pkl` outputs | The refactor now also writes internal partitioned staging artifacts under `artifacts/staging/` | Supports safe multicore preprocessing without changing the top-level output contract |

## Tune

### `tune tabular` and `tune sequence`

| Drift | Legacy behavior | Refactored behavior | Why |
| --- | --- | --- | --- |
| XGBoost device selection | Legacy HPO and training assumed CUDA in some paths | Refactor only requests CUDA when it is available | Improves portability to CPU-only machines |
| HPO split ownership | Refactor initially wrote HPO manifests in pipeline code but rebuilt them again in `models/hpo.py` | HPO manifests are now written once in pipeline code and passed into the tuning functions | Restores one split authority in the refactor |
| Optuna trial-state handling | Earlier refactor assumed completed trials stringify like `...COMPLETE` | Refactor now normalizes Optuna trial states across enum-style and numeric `4.x` representations before checking or writing them | Fixes false “no completed trials” failures on newer Optuna |
| Tabular tuning manifests | Earlier refactor per-dataset HPO manifests listed only the split parquet in `outputs` | Refactor now records the split parquet plus shared `tabular_best_params.json` and `tabular_trials.parquet`, and writes a top-level `tune_tabular.json` manifest | Makes tuning artifacts auditable and complete for handoff/resume |

## Train

### `train tabular`

| Drift | Legacy behavior | Refactored behavior | Why |
| --- | --- | --- | --- |
| `asa_rule` model routing | Could be enabled even for intraop-only datasets that do not contain `asa` | Skips `asa_rule` where `asa` is absent | Makes model-to-dataset compatibility explicit |
| AutoGluon sample weighting | Legacy training code referenced `balance_weight` without always materializing it | Refactor explicitly creates `balance_weight` before AutoGluon fitting | Fixes a real training-path bug |

### `train tabular` and `train sequence`

| Drift | Legacy behavior | Refactored behavior | Why |
| --- | --- | --- | --- |
| Raw prediction handling | Refactor initially overwrote or appended into one `raw_predictions.parquet` file depending on stage order | Training now writes stage-owned prediction partitions and rebuilds one deterministic combined raw prediction view | Makes reruns idempotent and removes order-dependent corruption |

### `train sequence`

| Drift | Legacy behavior | Refactored behavior | Why |
| --- | --- | --- | --- |
| Sequence checkpoints | Refactor initially saved write-only `bundle.pt` files without enough metadata to reconstruct the model | Sequence bundles now persist architecture metadata and can be reloaded through `load_sequence_bundle(...)` | Makes saved sequence artifacts reusable for inference and downstream analysis |

## Evaluate

### `evaluate calibrate`, `evaluate metrics`, `evaluate delong`, and `evaluate dca`

No major intentional evaluation-stage behavior drift is currently called out here beyond the preserved-on-purpose behaviors listed below.

## Explain

### `explain shap`

| Drift | Legacy behavior | Refactored behavior | Why |
| --- | --- | --- | --- |
| SHAP/report config | Refactor initially allowed `batch_shap_jobs` drift and silently skipped unsupported jobs like `svm` | Config now normalizes to `reports.shap_jobs`, validates supported models, and fails fast on unsupported jobs | Removes silent partial reporting |

## Report

### `report manuscript`

| Drift | Legacy behavior | Refactored behavior | Why |
| --- | --- | --- | --- |
| Manuscript reporting | Refactor initially required a separate SHAP stage even when the user asked for `report manuscript` | `report manuscript` now dispatches sections from `reports.manuscript_sections` and includes SHAP by default | Makes the report contract truthful |
| Cohort-characteristics source | Earlier refactor built cohort tables from the normalized labeled modeling dataset | Refactor now builds cohort characteristics from `tabular_combined_unnormalized.csv` merged with labels | Restores manuscript-facing clinical units in descriptive reporting |

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
| Worker/thread policy | Earlier refactor still relied on an ad hoc `cpu_count() - 2` helper in several places | The refactor now resolves stage-specific runtime plans from detected CPU, RAM, and GPU resources and records them in manifests | Makes large-node execution faster while keeping one concurrency authority |

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
