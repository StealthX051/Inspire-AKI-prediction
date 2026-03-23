# Refactor Behavior Drift

This file records intentional or observed differences between the legacy numbered-script path and the new `src/inspire_aki/` package path.

## Refactor-first fixes already applied

| Area | Legacy behavior | Refactored behavior | Why |
| --- | --- | --- | --- |
| Pandas dtype handling in `antype` | Mutated string columns in place with integer values, which breaks on newer pandas/Arrow-backed strings | Uses explicit `map()` to numeric anesthesia codes | Makes preprocessing portable across newer pandas versions |
| `merge_asof` time columns | Mixed integer and float surgery/lab timestamps depending on CSV inference | Casts surgery timing columns to float before `merge_asof` | Avoids dtype mismatch failures |
| `asa_rule` model routing | Could be enabled even for intraop-only datasets that do not contain `asa` | Skips `asa_rule` where `asa` is absent | Makes model-to-dataset compatibility explicit |
| Optional heavy deps | Importing the package could fail immediately if `torch`, `xgboost`, `optuna`, or `shap` were missing | Heavy dependencies are imported lazily in the code paths that actually need them | Keeps the CLI usable for lighter tabular-only/test scenarios |
| AutoGluon sample weighting | Legacy training code referenced `balance_weight` without always materializing it | Refactor explicitly creates `balance_weight` before AutoGluon fitting | Fixes a real training-path bug |
| XGBoost device selection | Legacy HPO/training assumed CUDA in some paths | Refactor only requests CUDA when it is available | Improves portability to CPU-only machines |
| Raw prediction handling | Refactor initially overwrote or appended into one `raw_predictions.parquet` file depending on stage order | Training now writes stage-owned prediction partitions and rebuilds one deterministic combined raw prediction view | Makes reruns idempotent and removes order-dependent corruption |
| Sequence checkpoints | Refactor initially saved write-only `bundle.pt` files without enough metadata to reconstruct the model | Sequence bundles now persist architecture metadata and can be reloaded through `load_sequence_bundle(...)` | Makes saved sequence artifacts reusable for inference and downstream analysis |
| SHAP/report config | Refactor initially allowed `batch_shap_jobs` drift and silently skipped unsupported jobs like `svm` | Config now normalizes to `reports.shap_jobs`, validates supported models, and fails fast on unsupported jobs | Removes silent partial reporting |
| Manuscript reporting | Refactor initially required a separate SHAP stage even when the user asked for `report manuscript` | `report manuscript` now dispatches sections from `reports.manuscript_sections` and includes SHAP by default | Makes the report contract truthful |
| HPO split ownership | Refactor initially wrote HPO manifests in pipeline code but rebuilt them again in `models/hpo.py` | HPO manifests are now written once in pipeline code and passed into the tuning functions | Restores one split authority in the refactor |
| Tabular merge fallback | Refactor initially preserved a risky `subject_id` to `op_id` merge fallback | Tabular dataset assembly now requires `op_id` in both upstream frames and fails clearly otherwise | Removes a silent wrong-join path |
| Legacy export toggle | Refactor config initially carried an unused `compat.export_legacy_aliases` flag | Legacy alias export is now explicit through `compat export-legacy` only | Removes dead config surface |

## Preserved-on-purpose behaviors

- AKI stage 3 still includes dialysis from `ward_vitals.csv` `crrt`.
- Base preprocessing still standardizes before imputation.
- High-missingness columns still get `-99`, with KNN imputation for lower missingness.
- Sequence preparation still defaults to the legacy-equivalent capped path.
- Calibration still uses isotonic regression with flat cross-validated probabilities and a single F2-optimal threshold per model/data regime.

## Follow-up candidates

- Revisit whether normalization-before-imputation should remain the default.
- Decide whether sequence preprocessing should keep dropping longer cases or move entirely to loader-side truncation/padding.
