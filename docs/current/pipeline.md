# Refactored Pipeline

These notes describe the current implementation under `src/inspire_aki/`.
They are meant to be readable, not exhaustive; for exact behavior, read the stage functions in `src/inspire_aki/pipelines/` and the supporting modules they call.

For the legacy numbered-script path, use [../legacy/README.md](../legacy/README.md).

## Top-Level Contract

- CLI surface: `src/inspire_aki/cli.py`
- Default config: `configs/aki/default.yaml`
- Artifact root: `paths.artifacts_dir` in config, `artifacts/` by default
- Raw INSPIRE root: `paths.raw_inspire_dir` in config
- Runtime planning: `inspire-aki runtime inspect --config ...`
- Orchestration: `inspire-aki run all --config ...`

Current behavior to keep in mind:

- `run all` executes preprocessing, tuning, training, evaluation, and `report manuscript`
- `run all` now emits immediate stage start/end lines plus `artifacts/logs/run_all_events.jsonl`
- long-running `tune_*` and `train_*` stages now append JSONL progress logs under `artifacts/logs/`
- interrupting a direct stage command or `run all` with `Ctrl-C` now exits cleanly with code `130`; overlapped child stages are terminated before the parent exits
- in `runtime.orchestration.mode: overlap`, `run all` overlaps `tune sequence` with `train tabular` after `tune tabular`
- `run all` does not call `compat export-legacy`
- SHAP can be run explicitly with `explain shap`, but `report manuscript` also includes SHAP when configured
- the current refactor optimization policy uses validation `balanced_accuracy` for HPO and early-stopping monitors
- trainable models use explicit inverse-frequency `balance_weight`-style weighting; `knn` applies the same weighting intent through deterministic weighted resampling because `sklearn` KNN does not accept `sample_weight`
- the current evaluation remains non-nested: HPO runs once on the cohort and later repeated-CV evaluation reuses that tuned parameter set

## Stage Map

### Preprocess

| Command | Main implementation | Primary inputs | Primary outputs | Notes |
| --- | --- | --- | --- | --- |
| `preprocess preop` | `pipelines/preprocess.py:run_preop` | raw `operations.csv`, `labs.csv`, `diagnosis.csv`, `ward_vitals.csv` | `features/preop/preop_features.csv`, `cohort/preop_audit.csv` | Builds the preop cohort/features and records audit metadata |
| `preprocess intraop` | `pipelines/preprocess.py:run_intraop` | raw `vitals.csv`, preop artifact | `features/intraop/feature_engineered.csv` | Builds tabular intraop features and fails if `inf` values remain |
| `preprocess tabular` | `pipelines/preprocess.py:run_tabular` | preop and intraop artifacts | `datasets/tabular/tabular_{combined,preop,intraop}.csv`, `tabular_combined_unnormalized.csv`, `normalization_stats.csv`, `features/fill_rates.csv` | Assembles the refactored tabular modeling datasets |
| `preprocess labels` | `pipelines/preprocess.py:run_labels` | preop artifact, combined tabular artifact, raw labs and ward vitals | `cohort/aki_labels.csv`, labeled tabular datasets | Derives AKI labels and joins them back onto each tabular regime |
| `preprocess timeseries` | `pipelines/preprocess.py:run_timeseries` | raw `vitals.csv`, `cohort/aki_labels.csv` | `features/timeseries/time_series_cleaned.csv` | Filters to labeled ops, cleans/interpolates timeseries, and writes staging partitions |
| `preprocess sequence` | `pipelines/preprocess.py:run_sequence` | `tabular_combined_labeled.csv`, cleaned timeseries artifact | `datasets/sequence/lstm_trainable.pkl` | Builds the sequence-ready dataset and writes sequence staging partitions |

### Tune

| Command | Main implementation | Primary inputs | Primary outputs | Notes |
| --- | --- | --- | --- | --- |
| `tune tabular` | `pipelines/tune.py:run_tune_tabular` | labeled preop, intraop, and combined tabular datasets | `datasets/splits/hpo_{preop,intraop,combined}.parquet`, `tuning/tabular_best_params.json`, `tuning/tabular_trials.parquet`, `tuning/tabular_studies/*` | Searches only the models enabled in `models.tabular_hpo_enabled`; current HPO objective is validation `balanced_accuracy`; matching completed per-study outputs resume automatically |
| `tune sequence` | `pipelines/tune.py:run_tune_sequence` | `datasets/sequence/lstm_trainable.pkl` | `datasets/splits/hpo_sequence.parquet`, `tuning/sequence_best_params.json`, `tuning/sequence_trials.parquet` | Searches only the models enabled in `models.sequence_hpo_enabled`; current HPO objective is validation `balanced_accuracy`; patience-based early stopping completes a trial, while only true Optuna pruning marks it `PRUNED` |

### Train

| Command | Main implementation | Primary inputs | Primary outputs | Notes |
| --- | --- | --- | --- | --- |
| `train tabular` | `pipelines/train.py:run_train_tabular` | labeled tabular datasets | `datasets/splits/bootstrap_{preop,intraop,combined}.parquet`, `models/tabular/...`, `predictions/raw/tabular.parquet`, `predictions/raw_predictions.parquet` | Trains every model in `models.tabular_enabled` across each tabular regime; tuned params override configured defaults when present; trainable models use explicit inverse-frequency balance weights; in the default optimized policy only `svm` fans out across repeats |
| `train sequence` | `pipelines/train.py:run_train_sequence` | `datasets/sequence/lstm_trainable.pkl` | `datasets/splits/bootstrap_sequence.parquet`, `models/sequence/...`, `predictions/raw/sequence.parquet`, `predictions/raw_predictions.parquet` | Trains every model in `models.sequence_enabled`; tuned params override configured defaults when present; sequence early stopping now follows validation `balanced_accuracy` |

### Evaluate

| Command | Main implementation | Primary inputs | Primary outputs | Notes |
| --- | --- | --- | --- | --- |
| `evaluate calibrate` | `pipelines/evaluate.py:run_calibration` | `predictions/raw_predictions.parquet` | `predictions/calibrated_predictions.parquet`, `evaluation/thresholds.csv` | Calibrates prediction groups and stores decision thresholds; repeated rows for the same `op_id` stay together through grouped calibration CV |
| `evaluate metrics` | `pipelines/evaluate.py:run_metrics` | calibrated predictions | `evaluation/metrics_by_fold.csv`, `evaluation/metrics_summary.csv`, `evaluation/metrics_bootstrap_ci.csv` | Bootstrap CI output is conditional on config and may be omitted |
| `evaluate delong` | `pipelines/evaluate.py:run_delong` | calibrated predictions | `evaluation/delong_matrix.csv`, `evaluation/delong_long.csv` | Pairwise AUROC comparison tables |
| `evaluate dca` | `pipelines/evaluate.py:run_dca` | calibrated predictions | `evaluation/dca_curves.csv` | Decision-curve rows for downstream plots/reporting |

### Explain

| Command | Main implementation | Primary inputs | Primary outputs | Notes |
| --- | --- | --- | --- | --- |
| `explain shap` | `pipelines/report.py:run_shap` and `reporting/shap.py` | trained bundles, datasets, report config | `explainability/shap_importance_*.csv`, SHAP figures | Uses `reports.shap_jobs`; only supported SHAP model families are allowed |

### Report

| Command | Main implementation | Primary inputs | Primary outputs | Notes |
| --- | --- | --- | --- | --- |
| `report consort` | `pipelines/report.py:run_consort` | cohort and audit artifacts | consort tables and graph source | Standalone consort output stage |
| `report tables` | `pipelines/report.py:run_tables` | tabular, label, and evaluation artifacts | manuscript-facing CSV, HTML, and Markdown tables | Includes cohort characteristics and performance tables |
| `report curves` | `pipelines/report.py:run_curves` | evaluation artifacts | ROC, PR, calibration, and DCA figures | Pure report-generation stage |
| `report manuscript` | `pipelines/report.py:run_manuscript` | report config and all upstream artifacts | combined report outputs under `reports/` | Dispatches sections from `reports.manuscript_sections` and includes SHAP when configured |

### Compat and Runtime

| Command | Main implementation | Primary inputs | Primary outputs | Notes |
| --- | --- | --- | --- | --- |
| `compat export-legacy` | `io/compat.py:export_legacy_datasets` | selected refactor artifacts | copied files under `compat_aki_dir`, `compat_base_dir`, and `compat_results_dir` | Explicit compatibility export only; not part of `run all` |
| `runtime inspect` | `runtime.py` | config plus detected host resources | JSON runtime summary | Use this before expensive runs on a new host class |
| `runtime benchmark` | `benchmarking.py` | config, selected profiles, selected targets | `artifacts/benchmarks/summary.{json,csv}` plus per-run logs | Compare runtime profiles or heavy stages without adding tracked benchmark artifacts; supports `--model-keys`, `--dataset-regimes`, and `--execution-policy` for targeted low-CPU benchmarks |

## Typical Current Runs

### Full default run

```bash
source .venv/bin/activate
inspire-aki runtime inspect --config configs/aki/default.yaml
inspire-aki run all --config configs/aki/default.yaml
inspire-aki runtime benchmark --config configs/aki/default.yaml --profiles throughput,balanced --targets tune_tabular,tune_sequence
```

### Resume stage-by-stage

```bash
source .venv/bin/activate
inspire-aki preprocess ...
inspire-aki tune ...
inspire-aki train ...
inspire-aki evaluate ...
inspire-aki report ...
```

## Runtime Notes

- `configs/aki/default.yaml` now defaults to `runtime.profile: throughput`
- `configs/aki/default.yaml` now defaults to `runtime.orchestration.mode: overlap`
- `configs/aki/smoke.yaml` and `configs/aki/smoke_hpo.yaml` pin `runtime.orchestration.mode: serial`
- `models.hpo.sequence_batch_size` controls the sequence-HPO batch size; the main default is `4096`
- `models.sequence_defaults.batch_size` controls final sequence training batch size; the main default is `4096`
- if the model-selection policy changes, resume the pipeline from `tune ...` rather than `train ...`
- the default low-CPU execution policy is intentionally narrow: `svm` gets outer concurrency, while `log_reg` stays serial with a moderate BLAS cap
