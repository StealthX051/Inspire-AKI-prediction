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
- Focused reviewer-response audit utilities: `scripts/department_os_audit.py`, `scripts/department_ot_reviewer_report.py`
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
`asa_rule` and `gs_aki_rule` (`Adapted GS-AKI`). `gs_aki_rule` is a maintained reviewer-response baseline that reuses the same patient-grouped manifests, metrics, and report flow as the rest of the current pipeline, but it is kept as a deterministic ordinal count/class score rather than isotonic-calibrated or threshold-optimized.

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

The default full-study configs now emit maintained SHAP beeswarms plus raw-value SHAP scatter plots for enabled SHAP jobs. Scatter and manual dependence plots explain the full held-out split, while the explainer background rows remain sampled for reproducible runtime. The SHAP stage now also balances its worker budget across active jobs instead of fanning out equally across every configured job, which shortens wall-clock time on larger hosts. The current featured manuscript scatter copies for `BSA`, `max_hr`, `max_inibg_mbp`, `op_len`, `preop_hb`, and `ward_rr` are mirrored under `reports/figures/shap_scatter_featured/` when those features are available in the selected SHAP job output.

Run the focused department reviewer-response audits when needed:

```bash
.venv/bin/python scripts/department_os_audit.py --config configs/aki/default.yaml
.venv/bin/python scripts/department_ot_reviewer_report.py --config configs/aki/default.yaml
```

These narrow audit scripts are maintained repo utilities under `scripts/`, not CLI stages. They default to writing reviewer-facing markdown/CSV outputs under repo-local `reports/`.

Run the cardiothoracic procedure audit explicitly when needed:

```bash
inspire-aki report procedure-audit --config configs/aki/default.yaml
```

The shipped default AKI config now points to the strict operation-level adjudicated noncardiac cohort used for the paper. The explicit procedure-audit report remains available as the transparency companion to that default filtering rule, but it is still not part of `report manuscript` or `run all`.

If you want to rerun the same strict cohort into a separate artifact root for isolated review work, the compatibility alias remains available:

```bash
inspire-aki run all --config configs/aki/strict_noncardiac_review.yaml
```

The currently mounted `artifacts/default` tree may predate this default-cohort promotion. Regenerate it before citing updated cohort counts, event counts, or procedure-audit totals from the new default path.

## Data And Scope

- This repo is not turnkey. End-to-end execution still requires private INSPIRE data.
- Raw data location is configured through `paths.raw_inspire_dir`; the shipped configs target the mounted volume path `/media/volume/ncs_inspire_data/ncs_aki/data/inspire`.
- Stage outputs, manifests, predictions, and reports are written under the configured `paths.artifacts_dir`.
- The maintained shipped configs use patient-grouped evaluation modes. `evaluate generate` materializes manifests on `patient_id` so the same patient does not cross train/test or train/validation boundaries in grouped runs.
- Calibration is guarded against repeated-row leakage: learned models use grouped isotonic calibration with CV on `op_id`, keeping repeated prediction rows for the same operation together.
- The maintained rule baselines keep prespecified thresholds instead of data-chosen cutoffs. `asa_rule` stays a binary preop rule, while `gs_aki_rule` is evaluated primarily by ordinal count/class with a prespecified Class III+ high-risk threshold only when binary metrics are needed.
- The default AKI run evaluates both maintained clinical baselines (`asa_rule` and the proxy-based `gs_aki_rule`) on that same grouped leakage-safe path; `gs_aki_rule` is restricted to the AKI outcome and the preop dataset regime.
- `inspire-aki compat export-legacy` remains explicit and AKI-only; it is not part of `run all`.

## Repo Map

- Current pipeline docs: [`docs/current/README.md`](docs/current/README.md)
- Reviewer-facing context: [`docs/reviewer/README.md`](docs/reviewer/README.md)
- Legacy archive: [`legacy/README.md`](legacy/README.md)
- Contributor notes: [`AGENTS.md`](AGENTS.md), `docs/HANDOFF/`, `docs/TODO/`

For the detailed CLI stage map and artifact contracts, start at [`docs/README.md`](docs/README.md).
