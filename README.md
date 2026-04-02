# Inspire-AKI-prediction

`inspire-aki` is the maintained interface for this repository. The package under `src/inspire_aki/` is the current pipeline for INSPIRE postoperative outcome modeling, with AKI as the default shipped target and additional outcome configs available for adjacent studies.

Legacy scripts, notebooks, and historical outputs are still kept in-repo for audit and manuscript-reference work, but they now live under [`legacy/`](legacy/README.md) and are archive-only.

## Quick Start

Preferred reproducible setup:

```bash
conda env create -f environment.yml
conda activate inspire-aki
```

Venv fallback:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip 'setuptools<81' wheel
pip install -r requirements.txt
pip install -e .
```

Then confirm the CLI is installed:

```bash
inspire-aki --help
```

## Primary Interface

- Package code: `src/inspire_aki/`
- CLI entrypoint: `inspire-aki`
- Shipped configs: `configs/aki/` and `configs/macce/`
- Helper wrappers: `scripts/run_smoke_test.sh`, `scripts/benchmark_runtime_profiles.sh`
- Tests: `tests/`

## Typical Runs

Inspect the resolved runtime plan before a new host-class run:

```bash
inspire-aki runtime inspect --config configs/aki/default.yaml
```

Run the full default AKI pipeline:

```bash
inspire-aki run all --config configs/aki/default.yaml
```

The default AKI config now includes two preoperative clinical baselines alongside the learned models:
`asa_rule` and `gs_aki_rule` (`Adapted GS-AKI`). `gs_aki_rule` is a maintained reviewer-response baseline that reuses the same patient-grouped manifests, calibration, metrics, and report flow as the rest of the current pipeline.

Resume stage-by-stage when needed:

```bash
inspire-aki evaluate generate --config configs/aki/default.yaml
inspire-aki tune tabular --config configs/aki/default.yaml
inspire-aki tune sequence --config configs/aki/default.yaml
inspire-aki train tabular --config configs/aki/default.yaml
inspire-aki train sequence --config configs/aki/default.yaml
inspire-aki evaluate calibrate --config configs/aki/default.yaml
inspire-aki evaluate metrics --config configs/aki/default.yaml
inspire-aki evaluate delong --config configs/aki/default.yaml
inspire-aki evaluate dca --config configs/aki/default.yaml
inspire-aki evaluate reclassification --config configs/aki/default.yaml
inspire-aki report manuscript --config configs/aki/default.yaml
```

Iterate on manuscript-facing outputs without retraining:

```bash
inspire-aki report consort --config configs/aki/default.yaml
inspire-aki report tables --config configs/aki/default.yaml
inspire-aki report manuscript --config configs/aki/default.yaml
```

## Data And Scope

- This repo is not turnkey. End-to-end execution still requires private INSPIRE data.
- Raw data location is configured through `paths.raw_inspire_dir`; the shipped configs target the mounted volume path `/media/volume/ncs_inspire_data/ncs_aki/data/inspire`.
- Stage outputs, manifests, predictions, and reports are written under the configured `paths.artifacts_dir`.
- The maintained shipped configs use patient-grouped evaluation modes. `evaluate generate` materializes manifests on `patient_id` so the same patient does not cross train/test or train/validation boundaries in grouped runs.
- Calibration is also guarded against repeated-row leakage: the maintained pipeline fits isotonic calibration with grouped CV on `op_id`, keeping repeated prediction rows for the same operation together.
- The default AKI run evaluates both maintained clinical baselines (`asa_rule` and the proxy-based `gs_aki_rule`) on that same grouped leakage-safe path; `gs_aki_rule` is restricted to the AKI outcome and the preop dataset regime.
- `inspire-aki compat export-legacy` remains explicit and AKI-only; it is not part of `run all`.

## Repo Map

- Current pipeline docs: [`docs/current/README.md`](docs/current/README.md)
- Reviewer-facing context: [`docs/reviewer/README.md`](docs/reviewer/README.md)
- Legacy archive: [`legacy/README.md`](legacy/README.md)
- Contributor notes: [`AGENTS.md`](AGENTS.md), `docs/HANDOFF/`, `docs/TODO/`

For the detailed CLI stage map and artifact contracts, start at [`docs/README.md`](docs/README.md).
