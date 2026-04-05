# Missingness Sensitivity Workflow

This note documents the maintained reviewer-response workflow added for the missing-data concern in the combined `xgb` model.

## Scope

- keep the default manuscript preprocessing path unchanged
- rerun only the combined `xgb` model for this reviewer question
- compare the default sentinel-based handling with a reviewer-specific sensitivity analysis that uses median imputation plus explicit missingness indicators for features with `>10%` missingness
- preserve the existing KNN handling for features with `0-10%` missingness
- derive the converted `>10%` feature list from the baseline `fill_rates.csv` artifact so the reviewer sensitivity matches the manuscript-era missingness rule rather than redefining the feature set fold by fold

## Why This Stays Separate

This workflow is intentionally separate from the main CLI preprocessing stages.

The maintained tabular preprocess currently applies outlier handling, normalization, and missingness handling before grouped train/test manifests are materialized. A public CLI-wide `missing_strategy` switch would therefore need a broader refactor so the alternative preprocessing could be fit inside each outer training fold without leaking grouped holdout information. That is a larger architecture change than this reviewer response requires.

For this reason, the reviewer workflow:

- leaves [../../src/inspire_aki/features/missingness.py](../../src/inspire_aki/features/missingness.py) unchanged
- leaves the default manuscript CLI path unchanged
- reuses the maintained grouped split, training, calibration, metrics, and SHAP helpers where possible
- fits the alternative missingness handling only inside the reviewer analysis module on the grouped outer folds

## Maintained Code Surface

- [../../src/inspire_aki/reporting/missingness_sensitivity.py](../../src/inspire_aki/reporting/missingness_sensitivity.py)
  - reviewer-specific grouped sensitivity workflow
- [../../configs/aki/reviewer_combined_xgb_baseline.yaml](../../configs/aki/reviewer_combined_xgb_baseline.yaml)
  - narrow baseline rerun config for a fresh isolated combined `xgb` comparison
- [../../scripts/combined_xgb_missingness_sensitivity.py](../../scripts/combined_xgb_missingness_sensitivity.py)
  - direct Python entrypoint for the sensitivity run
- [../../scripts/run_reviewer_missingness_sensitivity.sh](../../scripts/run_reviewer_missingness_sensitivity.sh)
  - convenience wrapper that reruns the paired baseline and then launches the sensitivity analysis

These reviewer utilities are maintained repo scripts under `scripts/`; they are intentionally outside `run all` and the public CLI stage map.

## Leakage Safeguards

- Outer evaluation reuses the maintained patient-grouped manifests from `evaluate generate`.
- The reviewer workflow rejects `legacy_repeated_cv`.
- The reviewer workflow rebuilds the combined feature matrix from the upstream preop and intraop feature artifacts rather than reusing the already processed tabular export.
- Missingness indicators are generated from raw missingness before imputation.
- For each outer fold, any outlier replacement uses quantiles fit on outer-train rows only.
- For each outer fold, medians for `>10%` missingness features are fit on outer-train rows only.
- For each outer fold, KNN imputation for `0-10%` missingness features is fit on outer-train rows only.
- Continuous scaling is also fit on outer-train rows only.
- Indicator columns remain explicit `0/1` columns and are excluded from scaling.
- No fresh HPO is run by default; the workflow reuses `models.tabular_hpo_params.combined.xgb`.
- Calibration reuses the maintained grouped isotonic path on held-out predictions with `op_id` grouping.

## Reproduction

Fresh paired baseline plus sensitivity run:

```bash
bash scripts/run_reviewer_missingness_sensitivity.sh
```

The wrapper uses the active environment's `inspire-aki` / `python` when available and falls back to the repo-local `.venv` if needed.

Direct sensitivity rerun against an already prepared baseline artifact tree:

```bash
python scripts/combined_xgb_missingness_sensitivity.py \
  --config configs/aki/reviewer_combined_xgb_baseline.yaml \
  --baseline-artifacts-dir /media/volume/ncs_inspire_data/ncs_aki/artifacts/reviewer_combined_xgb_baseline \
  --sensitivity-artifacts-dir /media/volume/ncs_inspire_data/ncs_aki/artifacts/reviewer_combined_xgb_baseline_median_plus_indicator_gt10 \
  --out-dir reports
```

## Expected Outputs

The reviewer workflow writes:

- baseline artifacts under the baseline config artifact root
- sensitivity artifacts under a separate reviewer artifact root
- repo-local comparison outputs under `reports/`

Key repo-local comparison outputs:

- `reports/missingness_sensitivity_performance_comparison.csv`
- `reports/missingness_sensitivity_shap_comparison.csv`
- `reports/missingness_sensitivity_converted_features.csv`
- `reports/missingness_sensitivity_indicator_ranks.csv`
- `reports/missingness_sensitivity_summary.md`

The generated markdown summary is the reviewer-facing result surface. It records the design decision, leakage safeguards, affected features, side-by-side performance deltas, indicator ranks, and two draft reviewer-response variants keyed to the observed results.

## Reviewer Framing

Preferred framing for the response letter:

“We agree that missingness may itself be informative in routinely collected perioperative data. Our concern was therefore not whether the model could learn from missingness, but whether encoding missing values as a fixed sentinel could conflate missingness with the continuous value itself and affect tree-based interpretability. To address this, we performed a targeted sensitivity analysis in the combined GBT model, replacing the sentinel encoding used for variables with >10% missingness with median imputation plus explicit missingness indicators, while preserving the original KNN-based handling for variables with <10% missingness and otherwise keeping the grouped evaluation and grouped calibration framework unchanged.”

The generated `reports/missingness_sensitivity_summary.md` file adds two reviewer-response variants after a run:

- Version A for broadly stable performance and SHAP conclusions
- Version B for meaningful attribution shifts that warrant interpretive caution
