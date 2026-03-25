# Inspire-AKI-prediction

Legacy names inside the repo still refer to `VitalDB-Dimensionality-Reduction`. This repository is best understood as a codebase for studying postoperative acute kidney injury (AKI) prediction in noncardiac surgery patients using the INSPIRE dataset.

## Current Refactor Status

- A refactored package-first execution path now exists under `src/inspire_aki/`.
- The new canonical entrypoint is the Typer CLI `inspire-aki`.
- The legacy numbered scripts and notebooks still matter for audit/parity, but they are no longer the only execution surface.
- The repo is still not turnkey: private INSPIRE data and explicit path configuration are still required.
- The refactor now uses a centralized runtime planner in `src/inspire_aki/runtime.py` instead of ad hoc worker counts.
- Raw refactor predictions are now written as stage partitions plus a deterministic combined `raw_predictions.parquet` view.
- `inspire-aki report manuscript` is now the report-level command that includes SHAP, rather than requiring a separate SHAP call.
- The refactor defaults now point at the mounted volume path `/media/volume/ncs_inspire_data/ncs_aki/data/inspire` for raw INSPIRE inputs.
- The shipped configs now also place refactor artifacts on the mounted volume under `/media/volume/ncs_inspire_data/ncs_aki/artifacts/`.
- On the current 32-CPU / A100 node, the default `throughput` runtime profile now targets roughly `30` usable CPUs for CPU-bound stages, keeps tensor-backed sequence loaders at `0` workers by default, and leaves GPU-native sequence work on the GPU.
- On the current 32-CPU / A100 node, the main default config now pins AutoGluon `num_cpus` to `32`, so AutoGluon can use the full host CPU count instead of the runtime-capped generic training-worker budget.
- On the current 32-CPU / A100 node, the main default config now uses `4096` for both sequence HPO batch size and final sequence training batch size.
- Sequence HPO now distinguishes true Optuna pruning from patience-based early stopping, so early-stopped trials still complete and contribute best params.
- `inspire-aki run all` now emits live stage progress to stdout and `<artifacts_dir>/logs/run_all_events.jsonl`, with dedicated JSONL progress logs for `tune_*` and `train_*`.
- Interrupting a direct stage command or `run all` with `Ctrl-C` now exits cleanly with code `130`; overlapped child stages are terminated before the parent exits.
- In `throughput` mode, `run all` now overlaps `tune sequence` with `train tabular` after `tune tabular` completes.
- `inspire-aki runtime benchmark` now writes machine-readable summaries under `<artifacts_dir>/benchmarks/`.
- `tune tabular` now commits durable per-study artifacts under `<artifacts_dir>/tuning/tabular_studies/` and resumes matching completed studies automatically.
- The low-CPU tabular optimization is intentionally narrow: only `svm` gets new outer concurrency, with regime-level HPO fanout and repeat-level train fanout; `log_reg` stays serial but uses a moderate BLAS cap.
- AutoGluon tabular training now disables DyStack explicitly to avoid the Ray-subprocess failure mode seen on this host, and skips optional model families when AutoGluon's own compatibility checks fail.
- `evaluate calibrate` now uses grouped cross-validation on `op_id`, so repeated prediction rows for the same case are kept together during isotonic calibration.
- The refactor now excludes operations with `op_len <= 0` upstream, which is an intentional cleanup relative to the legacy scripts.
- The refactor now treats infinite intraop feature values as invalid and fails the stage instead of silently carrying them forward.

## Current Validation Status

As of March 24, 2026, the refactor is in a strong but not fully validated state.

- The synthetic refactor test suite is green:
  - `pytest -q` currently passes with `91` tests.
- The real-data refactor preprocessing path has been exercised on the mounted INSPIRE volume through:
  - `preprocess preop`
  - `preprocess intraop`
  - `preprocess tabular`
  - `preprocess labels`
  - `preprocess timeseries`
  - `preprocess sequence`
- The real-data HPO smoke path has also now completed both tuning stages after fixing Optuna `4.x` trial-state handling:
  - `tune tabular`
  - `tune sequence`
- The full real-data `configs/aki/smoke_hpo.yaml` run has **not** yet been validated end to end in one uninterrupted pass through:
  - training
  - calibration
  - metrics
  - DeLong
  - DCA
  - manuscript reporting
- Treat the current refactor as:
  - contract-tested on synthetic data
  - partially validated on real INSPIRE data
  - still awaiting one clean end-to-end HPO smoke run

If another coder is resuming this work, the current recommended continuation point is:

```bash
source .venv/bin/activate

inspire-aki train tabular --config configs/aki/smoke_hpo.yaml
inspire-aki train sequence --config configs/aki/smoke_hpo.yaml
inspire-aki evaluate calibrate --config configs/aki/smoke_hpo.yaml
inspire-aki evaluate metrics --config configs/aki/smoke_hpo.yaml
inspire-aki evaluate delong --config configs/aki/smoke_hpo.yaml
inspire-aki evaluate dca --config configs/aki/smoke_hpo.yaml
inspire-aki report manuscript --config configs/aki/smoke_hpo.yaml
```

## Environment Setup

Preferred reproducible path:

```bash
conda env create -f environment.yml
conda activate inspire-aki
```

`environment.yml` now installs the pinned dependencies from `requirements.txt` and installs the local package in editable mode, so the `inspire-aki` CLI is available after activation.

Current-machine note:

- this Ubuntu environment does not currently have `conda`, `mamba`, or `micromamba` on `PATH`
- if you want the fastest setup here without installing Conda first, use:

```bash
sudo apt-get update
sudo apt-get install -y python3-venv graphviz
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip 'setuptools<81' wheel
pip install -r requirements.txt
pip install -e .
```

## End-to-End Smoke Test

The fastest real-data smoke test for the refactored CLI is a lightweight profile that:

- uses the mounted INSPIRE source tables at `/media/volume/ncs_inspire_data/ncs_aki/data/inspire`
- runs the full preprocessing, training, evaluation, and report path
- disables HPO entirely
- trains only one cheap tabular model (`log_reg`) and one cheap sequence model (`lstm_only`)
- writes outputs to `/media/volume/ncs_inspire_data/ncs_aki/artifacts/smoke/`

Run it with:

```bash
bash scripts/run_smoke_test.sh
```

Or, if you want to invoke the CLI directly:

```bash
inspire-aki preprocess preop --config configs/aki/smoke.yaml
inspire-aki preprocess intraop --config configs/aki/smoke.yaml
inspire-aki preprocess tabular --config configs/aki/smoke.yaml
inspire-aki preprocess labels --config configs/aki/smoke.yaml
inspire-aki preprocess timeseries --config configs/aki/smoke.yaml
inspire-aki preprocess sequence --config configs/aki/smoke.yaml
inspire-aki train tabular --config configs/aki/smoke.yaml
inspire-aki train sequence --config configs/aki/smoke.yaml
inspire-aki evaluate calibrate --config configs/aki/smoke.yaml
inspire-aki evaluate metrics --config configs/aki/smoke.yaml
inspire-aki evaluate delong --config configs/aki/smoke.yaml
inspire-aki evaluate dca --config configs/aki/smoke.yaml
inspire-aki report manuscript --config configs/aki/smoke.yaml
```

To inspect the resolved runtime plan before launching a run:

```bash
inspire-aki runtime inspect --config configs/aki/smoke.yaml
```

To benchmark runtime profiles or specific heavy stages:

```bash
inspire-aki runtime benchmark --config configs/aki/smoke.yaml --profiles balanced,throughput --targets run_all
```

To benchmark only the low-CPU tabular slice:

```bash
inspire-aki runtime benchmark --config configs/aki/default.yaml --profiles throughput --targets tune_tabular,train_tabular --model-keys svm --dataset-regimes preop,intraop,combined --execution-policy optimized_low_cpu
```

The wrapper still exists if you want a shell shortcut:

```bash
bash scripts/benchmark_runtime_profiles.sh
```

Benchmark summaries are written under the resolved `<artifacts_dir>/benchmarks/` directory for the selected config.

Important:

- the wrapper now always calls `tune tabular` and `tune sequence`, but under `configs/aki/smoke.yaml` those stages are lightweight because the HPO model lists are empty
- the smoke profile still exercises SHAP and manuscript report rendering, so it will fail if those dependencies or upstream artifacts are missing
- cohort counts may differ slightly from earlier smoke runs because zero-duration operations are now excluded during preop filtering
- configured ICD-10 procedure-prefix exclusions now happen before preop feature extraction, so excluded operations no longer incur unnecessary `merge_asof` work

If you also want to smoke-test HPO itself, use the dedicated HPO profile:

```bash
bash scripts/run_smoke_test.sh configs/aki/smoke_hpo.yaml
```

That profile:

- runs one Optuna trial per implemented HPO model
- narrows search spaces to cheap ranges
- shortens tabular-MLP and sequence HPO training loops
- writes outputs to `/media/volume/ncs_inspire_data/ncs_aki/artifacts/smoke_hpo/`

Current implemented HPO models are:

- tabular: `log_reg`, `xgb`, `rf`, `svm`, `mlp`, `knn`
- sequence: `lstm_only`, `hybrid`

`autogluon` and `asa_rule` are not part of the Optuna HPO path.
When `autogluon` is trained, the refactor keeps the legacy training intent separate from Optuna:

- `TabularPredictor(eval_metric="balanced_accuracy")`
- class-balance sample weights are materialized explicitly before fitting
- `num_gpus` is now passed through from config, with `configs/aki/default.yaml` defaulting to `auto`

Current refactor optimization policy:

- trainable models use explicit inverse-frequency `balance_weight`-style weighting during fitting
- `knn` consumes those weights through deterministic weighted resampling because `sklearn` KNN does not expose `sample_weight` on `fit()`
- HPO and early-stopping monitors now optimize validation `balanced_accuracy`
- the current repeated-CV evaluation remains non-nested: HPO is run once on the cohort and the later bootstrap CV reuses that tuned parameter set

Current HPO smoke note:

- the preprocessing stages for `configs/aki/smoke_hpo.yaml` have already run successfully on the current INSPIRE mount
- the tuning stages have also now completed after patching Optuna state detection
- the remaining unvalidated portion is the downstream `train -> evaluate -> report` chain on the HPO smoke profile

## What This Repo Is

- A mixed Python + notebook pipeline for:
  - cohort construction from INSPIRE source tables
  - AKI label derivation from perioperative creatinine data
  - preoperative and intraoperative feature engineering
  - tabular model training and deep sequence model training
  - calibration, thresholding, bootstrap confidence intervals, DeLong testing, decision-curve analysis, and SHAP interpretation
- A messy research repo with both newer numbered scripts and older exploratory code paths.
- Not a turnkey package. Most execution paths assume private INSPIRE data and hard-coded absolute paths under `/home/server/Projects/data/...`.

## Current Code-First Takeaways

- The most canonical executable path is the numbered script sequence:
  - `data_preprocessing/01_extract_preop.py`
  - `data_preprocessing/02_extract_intraop.py`
  - `data_preprocessing/03_create_base.py`
  - `data_preprocessing/04_AKI_data_selection.py`
  - `data_preprocessing/05_time_series_cleaner.py`
  - `data_preprocessing/06_create_lstm_trainable.py`
  - `create_results/07_tabular_hpo.py`
  - `create_results/08_tabular_model_creation.py`
  - `create_results/09_lstm_hpo.py`
  - `create_results/10_lstm_model_creation.py`
- Evaluation and figure generation then continue in notebooks under `create_results/`.
- The repo strongly supports the study conclusion that tabular models outperform the sequence path on the checked-in results:
  - `create_results/performance_table.md` reports combined AutoGluon AUROC `0.932`
  - `create_results/performance_table.md` reports combined `MLP+LSTM` AUROC `0.825`
- Important confirmed implementation details:
  - `data_preprocessing/04_AKI_data_selection.py` includes dialysis via `ward_vitals.csv` `crrt` when defining stage 3 AKI.
  - `data_preprocessing/03_create_base.py` normalizes before missing-value imputation, uses `-99` for `>=10%` missingness, and uses `KNNImputer` for lower-missingness columns.
  - `data_preprocessing/05_time_series_cleaner.py` cleans and interpolates 24 regular intraoperative signals.
  - `data_preprocessing/06_create_lstm_trainable.py` pads sequences to length `200` and drops longer operations.
  - `create_results/08_tabular_model_creation.py` and `create_results/10_lstm_model_creation.py` use repeated fold-style bootstrap splitting and do not enable every model/dataset by default.

## Repo Status

Treat this repository as a research archive with a partially modernized pipeline, not as polished production code.

- Mixed trust level:
  - higher trust: numbered scripts, checked-in performance tables, `bootstrap_metrics.py`, `decision_curve.py`
  - medium trust: `create_results/*.ipynb`, `data_preprocessing/consort_parity.py`, `data_preprocessing/outcomes_data_selection.py`
  - lower trust: exploratory notebooks, one-off scripts in `preoperative_models/`, legacy cleaners, artifact directories
- Private data dependency:
  - raw INSPIRE tables are not checked in
  - many scripts assume a local server filesystem layout
- Cohort-count drift exists:
  - manuscript summary says `49,198` patients
  - `data_preprocessing/consort_diagram_data.ipynb` records a filtered count of `57,724`
  - comments in `create_results/13_performance_metrics.ipynb` refer to roughly `67k` tabular patients and `54k` LSTM/hybrid patients

## Canonical Pipeline Map

### Refactored CLI path

1. `inspire-aki preprocess preop`
2. `inspire-aki preprocess intraop`
3. `inspire-aki preprocess tabular`
4. `inspire-aki preprocess labels`
5. `inspire-aki preprocess timeseries`
6. `inspire-aki preprocess sequence`
7. `inspire-aki tune tabular|sequence`
8. `inspire-aki train tabular|sequence`
9. `inspire-aki evaluate calibrate|metrics|delong|dca`
10. `inspire-aki explain shap`
11. `inspire-aki report consort|tables|curves|manuscript`
12. `inspire-aki compat export-legacy`

The refactor writes stage outputs and manifests under the configured artifact root instead of relying on implicit handoffs through `/home/server/...`.
The training path is idempotent at the artifact level:

- `train tabular` refreshes `<artifacts_dir>/predictions/raw/tabular.parquet`
- `train sequence` refreshes `<artifacts_dir>/predictions/raw/sequence.parquet`
- both rebuild `<artifacts_dir>/predictions/raw_predictions.parquet`
- if weighting, HPO objective, or other model-selection policy changes, resume from `tune`, not `train`

The refactor also uses internal staging artifacts for the parallel sequence path:

- `<artifacts_dir>/staging/timeseries_filtered/part-*.parquet`
- `<artifacts_dir>/staging/timeseries_cleaned/part-*.parquet`
- `<artifacts_dir>/staging/sequence/part-*.pkl`

Legacy alias exports remain explicit through `inspire-aki compat export-legacy`; they are not emitted automatically during `run all`.

### 1. Preoperative extraction

`data_preprocessing/01_extract_preop.py`

- Reads `operations.csv`, `labs.csv`, `diagnosis.csv`, and `ward_vitals.csv`
- Derives:
  - `BMI`
  - `BSA`
  - `booking_case_length`
  - `num_card_events`
  - one-hot department indicators
  - latest preop lab values within 90 days
  - latest ward vitals within 90 days
- Writes `preop_data_test.csv` under `/home/server/Projects/data/AKI/`

### 2. Intraoperative tabular feature engineering

`data_preprocessing/02_extract_intraop.py`

- Reads `vitals.csv`
- Builds:
  - 24 regular signals x 8 summary statistics
  - mean-only sparse variables
  - weight-adjusted and time-adjusted summed variables
  - `fluids_agg`
  - `equiv_MAC_totals`
- Writes `feature_engineered.csv`

### 3. Base dataset creation

`data_preprocessing/03_create_base.py`

- Merges preop and intraop tables
- Removes IDs and obvious leakage columns
- Replaces outliers using percentile windows
- Standardizes numeric features
- Fills high-missingness columns with `-99`
- KNN-imputes lower-missingness columns
- Writes:
  - `tabular_combined.csv`
  - `tabular_preop.csv`
  - `tabular_intraop.csv`
  - `normalization_stats.csv`

### 4. AKI label derivation

`data_preprocessing/04_AKI_data_selection.py`

- Uses preop creatinine plus postoperative 2-day and 7-day windows
- Includes dialysis from `ward_vitals.csv`
- Produces labeled AKI versions of:
  - `tabular_combined.csv`
  - `tabular_preop.csv`
  - `tabular_intraop.csv`

### 5. Sequence preparation

`data_preprocessing/05_time_series_cleaner.py`

- Filters to 24 regular intraoperative signals
- Removes duplicate `(op_id, chart_time, item_name)` rows
- Replaces outliers, interpolates onto 5-minute grids, fills within-op missingness with per-op means
- Writes `time_series_cleaned.csv`

`data_preprocessing/06_create_lstm_trainable.py`

- Merges cleaned time series with labeled tabular data
- Keeps only operations with acceptable time-series presence
- Pads sequences to `200` time steps
- Drops operations longer than that cap
- Writes `lstm_trainable.pkl`

### 6. Modeling

- Tabular HPO: `create_results/07_tabular_hpo.py`
- Tabular training: `create_results/08_tabular_model_creation.py`
- Deep HPO: `create_results/09_lstm_hpo.py`
- Deep training: `create_results/10_lstm_model_creation.py`

### 7. Evaluation and manuscript outputs

- Cohort diagram: `create_results/11_consort.ipynb`
- Cohort characteristics: `create_results/12_cohort_characteristics.ipynb`
- Calibration, F2 thresholding, CI bootstraps, reclassification, DCA: `create_results/13_performance_metrics.ipynb`
- DeLong testing: `create_results/14_delong_table.ipynb`
- SHAP for XGBoost combined model: `create_results/15_shap.ipynb`
- Batch SHAP for multiple models/datasets: `create_results/16_shap_batch.ipynb`

## Data and Portability

### Private inputs expected by the refactored path

- `/media/volume/ncs_inspire_data/ncs_aki/data/inspire/operations.csv`
- `/media/volume/ncs_inspire_data/ncs_aki/data/inspire/labs.csv`
- `/media/volume/ncs_inspire_data/ncs_aki/data/inspire/vitals.csv`
- `/media/volume/ncs_inspire_data/ncs_aki/data/inspire/diagnosis.csv`
- `/media/volume/ncs_inspire_data/ncs_aki/data/inspire/ward_vitals.csv`

Historical note:

- the legacy numbered scripts and notebooks were originally developed against a different server layout under `/home/server/Projects/data/...`
- the refactor config now defaults to the mounted replacement volume on this instance

### What is actually checked in

- checked-in result summaries:
  - `create_results/performance_table.md`
  - `create_results/performance_table_calibrated.md`
  - `create_results/descriptive_table.html`
  - `create_results/fill_rate_table.html`
  - `create_results/reclassification_report.html`
- checked-in model/artifact directories:
  - `AutogluonModels/`
  - `notebooks/mljar_results_improved/`

### Practical portability status

| Task | Main entrypoint | Status |
| --- | --- | --- |
| Read repo structure and checked-in findings | this README + `docs/` | runnable from repo only |
| Rebuild cohort/features from raw INSPIRE data | `inspire-aki preprocess ...` | requires private data; refactor defaults now target `/media/volume/ncs_inspire_data/ncs_aki/data/inspire` |
| Re-run model training | `create_results/07`-`10` | requires private data plus the pinned environment in `environment.yml` |
| Re-read checked-in performance results | `create_results/performance_table*.md` | runnable from repo only |
| Re-run figure notebooks | `create_results/*.ipynb` | depends on prior outputs and still reflects the older server-era workflow |

## Main Outputs and Findings

The checked-in outputs suggest the repo was used to create:

- Figure-style outputs:
  - cohort flow / consort
  - ROC and PR curves
  - calibration curves
  - decision-curve analysis
  - SHAP beeswarms, waterfalls, dependence/scatter plots
- Table-style outputs:
  - cohort characteristics
  - model performance
  - calibrated performance
  - fill-rate table
  - reclassification report
  - DeLong comparison tables

The checked-in performance tables support the manuscript-level story:

- preop structured data is already very strong
- low-frequency intraoperative data adds modest value
- the combined tabular models outperform the hybrid deep sequence path

## Documentation Set

Start with:

- [AGENTS.md](AGENTS.md)
- [docs/README.md](docs/README.md)

Deep dives:

- [docs/current/README.md](docs/current/README.md)
- [docs/current/pipeline.md](docs/current/pipeline.md)
- [docs/codex_workflow.md](docs/codex_workflow.md)
- [docs/refactor/behavior_drift.md](docs/refactor/behavior_drift.md)
- [docs/legacy/README.md](docs/legacy/README.md)
- [docs/legacy/01_repo_map.md](docs/legacy/01_repo_map.md)
- [docs/legacy/02_data_pipeline.md](docs/legacy/02_data_pipeline.md)
- [docs/legacy/03_labels_and_features.md](docs/legacy/03_labels_and_features.md)
- [docs/legacy/04_modeling_and_evaluation.md](docs/legacy/04_modeling_and_evaluation.md)
- [docs/legacy/05_artifacts_and_outputs.md](docs/legacy/05_artifacts_and_outputs.md)
- [docs/legacy/06_notebook_index.md](docs/legacy/06_notebook_index.md)
- [docs/legacy/07_manuscript_alignment.md](docs/legacy/07_manuscript_alignment.md)
- [docs/legacy/08_reproducibility_and_known_gaps.md](docs/legacy/08_reproducibility_and_known_gaps.md)

Project coordination:

- [docs/TODO/README.md](docs/TODO/README.md)
- [docs/HANDOFF/README.md](docs/HANDOFF/README.md)

## Important Caveats

- Do not assume the current repo can be executed end-to-end as-is on a fresh machine.
- Do not assume every notebook is canonical; many are exploratory or legacy.
- Do not assume manuscript counts and current code outputs are fully aligned.
- Use `environment.yml` as the main setup file; `requirements.txt` mirrors its pip-managed dependencies.
- Some portability fixes in the refactor intentionally differ from brittle legacy behavior; see [docs/refactor/behavior_drift.md](docs/refactor/behavior_drift.md).

## Recommended Reading Order

1. [README.md](README.md)
2. [AGENTS.md](AGENTS.md)
3. [docs/README.md](docs/README.md)
4. [docs/current/README.md](docs/current/README.md)
5. [docs/current/pipeline.md](docs/current/pipeline.md)
6. [docs/codex_workflow.md](docs/codex_workflow.md)
7. [docs/legacy/07_manuscript_alignment.md](docs/legacy/07_manuscript_alignment.md)
8. [docs/legacy/08_reproducibility_and_known_gaps.md](docs/legacy/08_reproducibility_and_known_gaps.md)
