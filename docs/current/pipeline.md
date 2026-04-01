# CLI Pipeline

These notes describe the maintained `inspire-aki` pipeline under `src/inspire_aki/`.

For archived scripts and notebooks, use [../../legacy/README.md](../../legacy/README.md).

## Top-Level Contract

- CLI surface: `src/inspire_aki/cli.py`
- Package code: `src/inspire_aki/`
- Default AKI config: `configs/aki/default.yaml`
- Additional shipped configs: `configs/aki/smoke.yaml`, `configs/aki/smoke_hpo.yaml`, `configs/macce/default.yaml`, `configs/macce/five_fold.yaml`, `configs/macce/smoke.yaml`, `configs/macce/smoke_hpo.yaml`
- Raw INSPIRE root: `paths.raw_inspire_dir`
- Artifact root: `paths.artifacts_dir`
- Runtime inspection: `inspire-aki runtime inspect --config ...`
- Full orchestration: `inspire-aki run all --config ...`

Current shipped evaluation modes:

- `configs/aki/default.yaml` -> `grouped_nested_cv`
- `configs/aki/smoke*.yaml` -> `grouped_holdout`
- `configs/macce/default.yaml` -> `grouped_holdout`
- `configs/macce/five_fold.yaml` -> `grouped_nested_cv`

These maintained grouped modes build manifests on `patient_id`, not isolated operation rows. The same patient should not appear on both sides of an outer train/test split or an inner train/validation split.

`legacy_repeated_cv` remains available only for historical/audit work and is not the default shipped path or the recommended manuscript-facing validation design.

## Leakage Guardrails

- `evaluate generate` resolves `patient_id` when needed and writes grouped split manifests plus split-audit tables under the artifact root.
- Grouped holdout and grouped nested-CV modes enforce patient-disjoint outer and inner splits rather than rebuilding inline operation-level random splits inside model code.
- The current grouped split logic stratifies at the patient level before assigning operations, which is the maintained path for reducing patient-overlap leakage.
- `evaluate calibrate` uses grouped calibration CV on `op_id`, so repeated prediction rows for the same operation stay together during isotonic fitting.
- The archived operation-level repeated-CV workflow is preserved only for audit/reference questions and should not be described as the primary way to run the repo today.

## Stage Map

### Preprocess

| Command | Main implementation | Primary outputs | Notes |
| --- | --- | --- | --- |
| `preprocess preop` | `pipelines/preprocess.py:run_preop` | `features/preop/preop_features.csv`, `cohort/preop_audit.csv` | Builds the study cohort and preop features |
| `preprocess intraop` | `pipelines/preprocess.py:run_intraop` | `features/intraop/feature_engineered.csv` | Produces tabular intraop summaries and fails if non-finite values remain |
| `preprocess tabular` | `pipelines/preprocess.py:run_tabular` | `datasets/tabular/tabular_{combined,preop,intraop}.csv`, `tabular_combined_unnormalized.csv`, `normalization_stats.csv`, `fill_rates.csv` | Assembles the tabular modeling datasets |
| `preprocess labels` | `pipelines/preprocess.py:run_labels` | `cohort/labels.csv`, `cohort/labels_audit.csv`, labeled tabular datasets | Derives the active outcome and joins labels back onto the datasets |
| `preprocess timeseries` | `pipelines/preprocess.py:run_timeseries` | `features/timeseries/time_series_cleaned.csv`, staging partitions | Cleans and interpolates labeled intraop time series |
| `preprocess sequence` | `pipelines/preprocess.py:run_sequence` | `datasets/sequence/lstm_trainable.pkl`, staging partitions | Builds the padded sequence-ready dataset |

### Tune

| Command | Main implementation | Primary outputs | Notes |
| --- | --- | --- | --- |
| `tune tabular` | `pipelines/tune.py:run_tune_tabular` | `tuning/tabular_best_params.json`, `tuning/tabular_trials.parquet`, `tuning/tabular_studies/`, HPO split manifests | Searches only enabled tabular HPO models |
| `tune sequence` | `pipelines/tune.py:run_tune_sequence` | `tuning/sequence_best_params.json`, `tuning/sequence_trials.parquet`, HPO split manifests | Searches only enabled sequence HPO models |

For grouped evaluation modes, `evaluate generate` must run before tuning. `run all` inserts that automatically so the downstream stages consume patient-grouped manifests instead of ad hoc row-wise splits.

### Train

| Command | Main implementation | Primary outputs | Notes |
| --- | --- | --- | --- |
| `train tabular` | `pipelines/train.py:run_train_tabular` | model bundles, `predictions/raw/tabular.parquet`, `predictions/raw_predictions.parquet` | Trains the enabled tabular models across supported dataset regimes |
| `train sequence` | `pipelines/train.py:run_train_sequence` | model bundles, `predictions/raw/sequence.parquet`, `predictions/raw_predictions.parquet` | Trains the enabled sequence models |

The package treats `predictions/raw/*.parquet` as stage-owned partitions and rebuilds the combined `raw_predictions.parquet` view from them.

### Evaluate

| Command | Main implementation | Primary outputs | Notes |
| --- | --- | --- | --- |
| `evaluate generate` | `pipelines/evaluate_generate.py:run_evaluate_generate` | grouped split manifests, split audits | Required for grouped holdout / grouped nested-CV execution; manifests are grouped on `patient_id` |
| `evaluate calibrate` | `pipelines/evaluate.py:run_calibration` | `predictions/calibrated_predictions.parquet`, `evaluation/thresholds.csv` | Uses grouped calibration CV on `op_id` so repeated rows for the same operation stay together |
| `evaluate metrics` | `pipelines/evaluate.py:run_metrics` | `evaluation/metrics_by_fold.csv`, `evaluation/metrics_summary.csv`, optional bootstrap CI outputs | Summary metrics and confidence intervals |
| `evaluate delong` | `pipelines/evaluate.py:run_delong` | raw and FDR-corrected DeLong tables | Pairwise AUROC testing |
| `evaluate dca` | `pipelines/evaluate.py:run_dca` | `evaluation/dca_curves.csv`, optional CI bands | Decision-curve analysis outputs |
| `evaluate reclassification` | `pipelines/evaluate.py:run_reclassification` | `evaluation/reclassification_summary.csv` | Downstream manuscript reporting input |

### Explain

| Command | Main implementation | Primary outputs | Notes |
| --- | --- | --- | --- |
| `explain shap` | `pipelines/report.py:run_shap` and `reporting/shap.py` | `explainability/shap_importance_*.csv`, SHAP figures | Supported model families are enforced in code |

### Report

| Command | Main implementation | Primary outputs | Notes |
| --- | --- | --- | --- |
| `report consort` | `pipelines/report.py:run_consort` | `reports/tables/consort_audit.*`, `reports/figures/consort.{png,svg}`, DOT source | Standalone consort generation |
| `report tables` | `pipelines/report.py:run_tables` | manuscript-facing tables in `html`, `md`, and `csv` | Performance, cohort, CI, DeLong, and reclassification tables |
| `report curves` | `pipelines/report.py:run_curves` | ROC, PR, calibration, and DCA figures in `png` and `svg` | Fold/run aggregation happens here |
| `report manuscript` | `pipelines/report.py:run_manuscript` | combined outputs under `reports/` | Dispatches the configured manuscript sections, including SHAP when enabled |

### Compat And Runtime

| Command | Main implementation | Primary outputs | Notes |
| --- | --- | --- | --- |
| `compat export-legacy` | `io/compat.py:export_legacy_datasets` | copied compatibility files under configured compat roots | Explicit export only; AKI-only |
| `runtime inspect` | `runtime.py` | JSON runtime plan | Use before expensive runs on a new host class |
| `runtime benchmark` | `benchmarking.py` | benchmark summaries under the configured artifact root | Target specific stages, models, or regimes when needed |

## Typical Runs

### Full default AKI run

```bash
source .venv/bin/activate
inspire-aki runtime inspect --config configs/aki/default.yaml
inspire-aki run all --config configs/aki/default.yaml
```

### Full MACCE run

```bash
source .venv/bin/activate
inspire-aki runtime inspect --config configs/macce/default.yaml
inspire-aki run all --config configs/macce/default.yaml
```

### Resume stage-by-stage

```bash
source .venv/bin/activate
inspire-aki preprocess ...
inspire-aki evaluate generate --config configs/aki/default.yaml
inspire-aki tune ...
inspire-aki train ...
inspire-aki evaluate ...
inspire-aki report ...
```

For the shipped grouped configs, do not skip `evaluate generate` when resuming manually. That stage materializes the patient-grouped manifests that the maintained tuning and training paths expect.

### Fast manuscript iteration

```bash
source .venv/bin/activate
inspire-aki report consort --config configs/aki/default.yaml
inspire-aki report tables --config configs/aki/default.yaml
inspire-aki report manuscript --config configs/aki/default.yaml
```

## Reporting And Artifact Notes

- The current report layer targets manuscript-ready exports directly.
- Tables are emitted in `html`, `md`, and `csv`.
- Figures are emitted in `png` and `svg`.
- Rerunning report stages replaces the canonical filenames under `reports/`.
- If model-selection policy changes, resume from `tune ...`, not `train ...`.
- When documenting validation design, describe the maintained pipeline as patient-grouped evaluation plus `op_id`-grouped calibration, not the archived operation-level repeated-CV workflow.
