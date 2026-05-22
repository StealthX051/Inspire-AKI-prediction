# INSPIRE AKI Prediction

Analysis code for the JAMIA Open article:

**Integration of Intraoperative Data in Interpretable Machine Learning Models to Predict Postoperative AKI in Noncardiac Surgery Patients**

This repository provides code, configuration files, tests, and environment files for reproducing the manuscript analyses after obtaining authorized access to the source data.

## Data access

The source data analyzed in the study are INSPIRE version 1.3, available from PhysioNet:

- INSPIRE version 1.3 DOI: https://doi.org/10.13026/46m4-f655
- PhysioNet record: https://physionet.org/content/inspire/1.3/

INSPIRE source files and row-level derived datasets are not included in this repository or in the associated Dryad record. Users must obtain authorized access through PhysioNet and agree to the applicable Data Use Agreement before running the full analysis.

Although later INSPIRE versions may be available, this study used INSPIRE version 1.3.

## Quick start

Analyses were performed in Python 3.10. The preferred reproducible setup is:

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

## Local paths

The shipped configs contain site-specific defaults for the original analysis environment. Before running the pipeline, update local paths in the selected config, for example:

```yaml
paths:
  raw_inspire_dir: /path/to/authorized/INSPIRE_v1.3
  artifacts_dir: /path/to/local/artifacts
```

Generated artifacts, reports, trained models, predictions, train/test splits, patient-level SHAP values, and row-level derived datasets should remain local and are excluded from the publication release snapshot.

## Primary interface

- Package code: `src/inspire_aki/`
- CLI entrypoint: `inspire-aki`
- Shipped configs: `configs/aki/` and `configs/macce/`
- Helper wrappers: `scripts/run_smoke_test.sh`, `scripts/benchmark_runtime_profiles.sh`
- Supplemental sensitivity and audit utilities: `scripts/department_os_audit.py`, `scripts/department_ot_reviewer_report.py`, `scripts/combined_xgb_missingness_sensitivity.py`, `scripts/run_reviewer_missingness_sensitivity.sh`
- Tests: `tests/`

## Typical runs

Inspect the resolved runtime plan before a new host-class run:

```bash
inspire-aki runtime inspect --config configs/aki/default.yaml
```

Run the full default AKI pipeline:

```bash
inspire-aki run all --config configs/aki/default.yaml
```

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

Run supplemental sensitivity or audit utilities only when needed:

```bash
bash scripts/run_reviewer_missingness_sensitivity.sh
.venv/bin/python scripts/department_os_audit.py --config configs/aki/default.yaml
.venv/bin/python scripts/department_ot_reviewer_report.py --config configs/aki/default.yaml
```

## Data and scope

- End-to-end execution requires authorized INSPIRE data.
- Raw data location is configured through `paths.raw_inspire_dir`.
- Stage outputs, manifests, predictions, and reports are written under `paths.artifacts_dir`.
- The maintained shipped configs use patient-grouped evaluation modes. `evaluate generate` materializes manifests on `patient_id` so the same patient does not cross train/test or train/validation boundaries in grouped runs.
- Calibration and threshold selection are guarded against leakage in the maintained grouped modes: learned models generate outer-train OOF calibration predictions from patient-grouped inner folds, fit isotonic calibration on those outer-train rows only, choose thresholds on the same calibrated outer-train rows, and then apply both to untouched outer-test operations.
- The maintained rule baselines keep prespecified thresholds instead of data-chosen cutoffs. `asa_rule` stays a binary preop rule, while `gs_aki_rule` is evaluated primarily by ordinal count/class with a prespecified Class III+ high-risk threshold only when binary metrics are needed.
- The default AKI run evaluates both maintained clinical baselines (`asa_rule` and the proxy-based `gs_aki_rule`) on that same grouped leakage-safe path; `gs_aki_rule` is restricted to the AKI outcome and the preop dataset regime.

## Documentation

- Current pipeline docs: `docs/current/README.md`
- CLI stage map and artifact contracts: `docs/current/pipeline.md`
- Dryad upload materials: `dryad/`

## Citation

If you use this software, please cite the associated JAMIA Open article, the archived software release, and the INSPIRE source dataset. See `CITATION.cff` for software citation metadata.

## License

This repository is released under the MIT License. See `LICENSE`.

This license applies to the code and repository documentation. INSPIRE source data are not included in this repository and must be obtained separately through PhysioNet under the applicable access terms.
