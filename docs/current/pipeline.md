# Refactored Pipeline

These notes describe the current implementation under `src/inspire_aki/`.
They are meant to be readable, not exhaustive; for exact behavior, read the stage functions in `src/inspire_aki/pipelines/` and the supporting modules they call.

For the legacy numbered-script path, use [../legacy/README.md](../legacy/README.md).

## Top-Level Contract

- CLI surface: `src/inspire_aki/cli.py`
- Default config: `configs/aki/default.yaml`
- Additional shipped outcome configs: `configs/macce/{default,smoke,smoke_hpo}.yaml`
- Artifact root: `paths.artifacts_dir` in config, `/media/volume/ncs_inspire_data/ncs_aki/artifacts/default` in the shipped default config
- Raw INSPIRE root: `paths.raw_inspire_dir` in config
- Active study target: `study.outcome_key`, normalized to `outcome.*` and `models.target`
- Runtime planning: `inspire-aki runtime inspect --config ...`
- Orchestration: `inspire-aki run all --config ...`

Current behavior to keep in mind:

- `run all` executes preprocessing, grouped split generation when needed, tuning, training, evaluation, and `report manuscript`
- as of March 26, 2026, the main default artifact root `/media/volume/ncs_inspire_data/ncs_aki/artifacts/default` has completed end to end through `report manuscript`
- `run all` now emits immediate stage start/end lines plus `<artifacts_dir>/logs/run_all_events.jsonl`
- long-running `tune_*` and `train_*` stages now append JSONL progress logs under `<artifacts_dir>/logs/`
- interrupting a direct stage command or `run all` with `Ctrl-C` now exits cleanly with code `130`; overlapped child stages are terminated before the parent exits
- in `runtime.orchestration.mode: overlap`, `run all` overlaps `tune sequence` with `train tabular` after `tune tabular`
- `run all` does not call `compat export-legacy`
- for `evaluation_mode: grouped_holdout` or `grouped_nested_cv`, `run all` inserts `evaluate generate` automatically before tuning
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
| `preprocess labels` | `pipelines/preprocess.py:run_labels` | preop artifact, combined tabular artifact, and the raw source tables required by the active outcome | `cohort/labels.csv`, `cohort/labels_audit.csv`, labeled tabular datasets | Derives the active outcome labels and joins `subject_id`, `patient_id`, and the active target back onto each tabular regime; when the active outcome is AKI it also writes the legacy compat alias `cohort/aki_labels.csv` |
| `preprocess timeseries` | `pipelines/preprocess.py:run_timeseries` | raw `vitals.csv`, `cohort/labels.csv` | `features/timeseries/time_series_cleaned.csv` | Filters to labeled ops, cleans/interpolates timeseries, and writes staging partitions |
| `preprocess sequence` | `pipelines/preprocess.py:run_sequence` | `tabular_combined_labeled.csv`, cleaned timeseries artifact | `datasets/sequence/lstm_trainable.pkl` | Builds the sequence-ready dataset and writes sequence staging partitions |

### Tune

| Command | Main implementation | Primary inputs | Primary outputs | Notes |
| --- | --- | --- | --- | --- |
| `tune tabular` | `pipelines/tune.py:run_tune_tabular` | labeled preop, intraop, and combined tabular datasets | `datasets/splits/hpo_{preop,intraop,combined}.parquet` in legacy mode, `datasets/splits/hpo_{preop,intraop,combined}_run_<id>.parquet` in grouped modes, `tuning/tabular_best_params.json`, `tuning/tabular_trials.parquet`, `tuning/tabular_studies/*` | Searches only the models enabled in `models.tabular_hpo_enabled`; current HPO objective is validation `balanced_accuracy`; matching completed per-study outputs resume automatically; for grouped evaluation modes it requires `evaluate generate` first and writes one HPO manifest plus run-scoped best params per generated outer run |
| `tune sequence` | `pipelines/tune.py:run_tune_sequence` | `datasets/sequence/lstm_trainable.pkl` | `datasets/splits/hpo_sequence.parquet` in legacy mode, `datasets/splits/hpo_sequence_run_<id>.parquet` in grouped modes, `tuning/sequence_best_params.json`, `tuning/sequence_trials.parquet` | Searches only the models enabled in `models.sequence_hpo_enabled`; current HPO objective is validation `balanced_accuracy`; patience-based early stopping completes a trial, while only true Optuna pruning marks it `PRUNED`; for grouped evaluation modes it requires `evaluate generate` first and writes one HPO manifest plus run-scoped best params per generated outer run |

### Train

| Command | Main implementation | Primary inputs | Primary outputs | Notes |
| --- | --- | --- | --- | --- |
| `train tabular` | `pipelines/train.py:run_train_tabular` | labeled tabular datasets | `datasets/splits/bootstrap_{preop,intraop,combined}.parquet`, `models/tabular/...`, `predictions/raw/tabular.parquet`, `predictions/raw_predictions.parquet` | Trains every model in `models.tabular_enabled` across each tabular regime; tuned params override configured defaults when present; trainable models use explicit inverse-frequency balance weights; in the default optimized policy only `svm` fans out across repeats; AutoGluon disables DyStack and skips optional model families when AutoGluon's compatibility checks fail; for grouped evaluation modes it requires `evaluate generate` first and trains from the grouped outer train/test folds instead of writing new bootstrap manifests |
| `train sequence` | `pipelines/train.py:run_train_sequence` | `datasets/sequence/lstm_trainable.pkl` | `datasets/splits/bootstrap_sequence.parquet`, `models/sequence/...`, `predictions/raw/sequence.parquet`, `predictions/raw_predictions.parquet` | Trains every model in `models.sequence_enabled`; tuned params override configured defaults when present; sequence early stopping now follows validation `balanced_accuracy`; for grouped evaluation modes it requires `evaluate generate` first and trains from the grouped outer train/test folds instead of writing a new bootstrap manifest |

### Evaluate

| Command | Main implementation | Primary inputs | Primary outputs | Notes |
| --- | --- | --- | --- | --- |
| `evaluate generate` | `pipelines/evaluate_generate.py:run_evaluate_generate` | labeled tabular datasets, optional sequence dataset, preop features for patient lookup | grouped split manifests under `datasets/splits/` plus `evaluation/split_audit*.csv` | Only needed for `evaluation_mode: grouped_holdout` or `grouped_nested_cv`; resolves `patient_id` from preop `subject_id` when the modeling datasets do not carry it directly |
| `evaluate calibrate` | `pipelines/evaluate.py:run_calibration` | `predictions/raw_predictions.parquet` | `predictions/calibrated_predictions.parquet`, `evaluation/thresholds.csv` | Calibrates prediction groups and stores decision thresholds; repeated rows for the same `op_id` stay together through grouped calibration CV |
| `evaluate metrics` | `pipelines/evaluate.py:run_metrics` | calibrated predictions | `evaluation/metrics_by_fold.csv`, `evaluation/metrics_summary.csv`, `evaluation/metrics_bootstrap_ci.csv` | Bootstrap CI output is conditional on config and may be omitted |
| `evaluate delong` | `pipelines/evaluate.py:run_delong` | calibrated predictions | `evaluation/delong_matrix.csv`, `evaluation/delong_long.csv`, `evaluation/delong_fdr_corrected.csv`, `evaluation/delong_fdr_corrected_long.csv` | Pairwise AUROC comparison tables, including FDR-corrected artifacts |
| `evaluate dca` | `pipelines/evaluate.py:run_dca` | calibrated predictions | `evaluation/dca_curves.csv`, `evaluation/dca_bootstrap_ci.csv` | Decision-curve rows plus long-form bootstrap CI bands for downstream manuscript plots |
| `evaluate reclassification` | `pipelines/evaluate.py:run_reclassification` | calibrated predictions | `evaluation/reclassification_summary.csv` | Stage-owned reclassification summary used by manuscript reporting; if the summary is empty, reporting skips the reclassification table instead of failing |

### Explain

| Command | Main implementation | Primary inputs | Primary outputs | Notes |
| --- | --- | --- | --- | --- |
| `explain shap` | `pipelines/report.py:run_shap` and `reporting/shap.py` | trained bundles, datasets, report config | `explainability/shap_importance_*.csv`, SHAP figures | Uses `reports.shap_jobs`; only supported SHAP model families are allowed |

### Report

| Command | Main implementation | Primary inputs | Primary outputs | Notes |
| --- | --- | --- | --- | --- |
| `report consort` | `pipelines/report.py:run_consort` | cohort and audit artifacts | `consort_audit.{html,md,csv}`, `consort.dot`, `consort.{png,svg}` | Standalone consort output stage; renders a top-down branched manuscript-style Graphviz flow with explicit exclusion summaries and final active-outcome negative / positive terminal nodes |
| `report tables` | `pipelines/report.py:run_tables` | tabular, label, and evaluation artifacts | manuscript-facing core tables in `html`, `md`, and `csv` | Includes legacy-style uncalibrated and calibrated performance tables plus descriptive tables; cohort summaries now use the active outcome display label rather than AKI-only wording, prefer the combined unnormalized cohort artifact when available, restore the legacy `False = female` sex encoding, and emit deduplicated full-name department rows; HTML performance tables keep a fixed manuscript order, restrict `ASA Rule` to preop, and use gentle monochrome column-wise gradients; grouped-holdout performance tables derive bootstrap CIs directly from the saved prediction artifacts using the same manuscript metric definitions shown in the table |
| `report curves` | `pipelines/report.py:run_curves` | evaluation artifacts | ROC, PR, calibration, DCA, and comparison figures in `png` and `svg` | Uses fold/run aggregation for ROC and PR uncertainty bands |
| `report manuscript` | `pipelines/report.py:run_manuscript` | report config and all upstream artifacts | combined report outputs under `reports/` | Dispatches `consort`, `tables`, `curves`, `statistics`, `reclassification`, and `shap` from `reports.manuscript_sections` |

### Compat and Runtime

| Command | Main implementation | Primary inputs | Primary outputs | Notes |
| --- | --- | --- | --- | --- |
| `compat export-legacy` | `io/compat.py:export_legacy_datasets` | selected refactor artifacts | copied files under `compat_aki_dir`, `compat_base_dir`, and `compat_results_dir` | Explicit compatibility export only; not part of `run all`; only supported when `study.outcome_key: aki` |
| `runtime inspect` | `runtime.py` | config plus detected host resources | JSON runtime summary | Use this before expensive runs on a new host class |
| `runtime benchmark` | `benchmarking.py` | config, selected profiles, selected targets | `<artifacts_dir>/benchmarks/summary.{json,csv}` plus per-run logs | Compare runtime profiles or heavy stages without adding tracked benchmark artifacts; supports `--model-keys`, `--dataset-regimes`, and `--execution-policy` for targeted low-CPU benchmarks |

## Typical Current Runs

### Full default run

```bash
source .venv/bin/activate
inspire-aki runtime inspect --config configs/aki/default.yaml
inspire-aki run all --config configs/aki/default.yaml
inspire-aki runtime benchmark --config configs/aki/default.yaml --profiles throughput,balanced --targets tune_tabular,tune_sequence
```

### Full MACCE grouped-holdout run

```bash
source .venv/bin/activate
inspire-aki runtime inspect --config configs/macce/default.yaml
inspire-aki run all --config configs/macce/default.yaml
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

### Rapid Manuscript Iteration

```bash
source .venv/bin/activate
inspire-aki report consort --config configs/aki/default.yaml
inspire-aki report tables --config configs/aki/default.yaml
inspire-aki report manuscript --config configs/aki/default.yaml
```

Notes:

- `report consort` is the fastest loop for Graphviz consort-layout iteration
- `report tables` is the tightest loop for manuscript table styling and ordering, but it still recomputes fold/run performance summaries from the saved prediction artifacts
- in grouped-holdout mode, rerunning `report tables` also recomputes the manuscript-table bootstrap intervals from those saved predictions

## Runtime Notes

- `configs/aki/default.yaml` now defaults to `runtime.profile: throughput`
- `configs/aki/default.yaml` now defaults to `runtime.orchestration.mode: overlap`
- `configs/aki/smoke.yaml` and `configs/aki/smoke_hpo.yaml` pin `runtime.orchestration.mode: serial`
- the shipped default config now raises the heavy-stage worker caps above the planner's generic throughput defaults:
  - `csv_read_threads: 24`
  - `tabular_column_workers: 24`
  - `timeseries_workers: 24`
  - `sequence_workers: 24`
  - `evaluation_workers: 24`
  - `bootstrap_workers: 24`
  - `report_workers: 16`
  - `shap_workers: 6`
- report defaults now target manuscript export directly:
  - `reports.table_formats: [html, md, csv]`
  - `reports.figure_formats: [png, svg]`
  - `reports.figure_png_dpi: 600`
  - `reports.manuscript_sections: [consort, tables, curves, statistics, reclassification, shap]`
- rerunning `report ...` stages writes to the same canonical filenames under `reports/`; existing files with the same stems are replaced automatically
- `models.hpo.sequence_batch_size` controls the sequence-HPO batch size; the main default is `4096`
- `models.sequence_defaults.batch_size` controls final sequence training batch size; the main default is `4096`
- `models.autogluon.num_cpus` is pinned to `32` in `configs/aki/default.yaml` on the current host so AutoGluon can use the full machine CPU count
- if the model-selection policy changes, resume the pipeline from `tune ...` rather than `train ...`
- the default low-CPU execution policy is intentionally narrow: `svm` gets outer concurrency, while `log_reg` stays serial with a moderate BLAS cap
