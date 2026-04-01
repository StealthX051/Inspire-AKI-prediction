# Legacy Archive

This directory preserves the historical notebook-era and script-era research workflow. It is kept for audit, parity checks, manuscript-reference work, and recovery of old implementation details.

It is not the supported execution surface for the repo. The maintained interface is the CLI pipeline described in [`../docs/current/README.md`](../docs/current/README.md).

## Structure

- `code/`
  - archived scripts, notebooks, helper modules, and exploratory research code
- `root_experiments/`
  - loose root-level experiments moved out of the primary repo surface
- `reference_outputs/`
  - small historical manuscript-facing outputs still kept in-repo
- `docs/`
  - archive-only notes for the historical workflow

## Trust Levels

- Highest archive trust:
  - `code/data_preprocessing/01`-`06`
  - `code/create_results/07`-`10`
  - `code/create_results/bootstrap_metrics.py`
  - `code/create_results/decision_curve.py`
  - `reference_outputs/create_results/`
- Medium archive trust:
  - `code/create_results/*.ipynb`
  - `code/data_preprocessing/consort_parity.py`
  - `code/data_preprocessing/outcomes_data_selection.py`
- Lower archive trust:
  - exploratory notebooks
  - `code/preoperative_models/`
  - loose experiments under `root_experiments/`
  - historical model/output directories that were removed from the main repo surface

## Archive Docs

- [docs/02_data_pipeline.md](docs/02_data_pipeline.md)
- [docs/03_labels_and_features.md](docs/03_labels_and_features.md)
- [docs/04_modeling_and_evaluation.md](docs/04_modeling_and_evaluation.md)
- [docs/05_artifacts_and_outputs.md](docs/05_artifacts_and_outputs.md)
- [docs/06_notebook_index.md](docs/06_notebook_index.md)

## Externalized Artifacts

Large generated model and AutoML trees were removed from the primary repo surface during the CLI-first cleanup. See [externalized_artifacts.md](externalized_artifacts.md) for the manifest.

## How To Use The Archive Safely

- inspect it when historical behavior matters
- do not describe it as the current primary interface
- patch `src/inspire_aki/` for forward-looking work unless the task explicitly targets the archive
