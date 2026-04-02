# Manuscript Alignment

This document is the code-first bridge between the manuscript story and the current repository.

Archive code references below live under `legacy/code/`. Current package behavior lives under `src/inspire_aki/`.

| Topic | Evidence | Current position | Reviewer note |
| --- | --- | --- | --- |
| Study framing | `legacy/code/data_preprocessing/01`-`06`, `legacy/code/create_results/07`-`10`, `src/inspire_aki/` | The repo supports preop, intraop, combined tabular paths plus a sequence path | The maintained interface is the CLI, not the archived script chain |
| AKI target | `legacy/code/data_preprocessing/04_AKI_data_selection.py` | The historical binary target is effectively severe AKI (`aki_2 or aki_3`) | Reviewer text should describe the implemented rule, not only KDIGO prose |
| Dialysis handling | `legacy/code/data_preprocessing/04_AKI_data_selection.py` | Historical stage 3 AKI includes dialysis via `ward_vitals.csv` `crrt` | This remains an explicit manuscript/code detail to document |
| Cohort counts | `legacy/code/data_preprocessing/consort_diagram_data.ipynb`, `legacy/code/create_results/13_performance_metrics.ipynb`, current reports | Counts vary by stage and workflow | Use the specific table/figure artifact being cited rather than asserting one universal count from the repo alone |
| Dataset regimes | `legacy/code/data_preprocessing/03_create_base.py`, current tabular datasets | Preop-only, intraop-only, and combined datasets are part of both the archived and maintained workflows | No conflict at the high level |
| Sequence path | `legacy/code/data_preprocessing/05_time_series_cleaner.py`, `legacy/code/data_preprocessing/06_create_lstm_trainable.py`, current sequence preprocessing | The sequence path pads to `200` steps and drops longer operations | This is important when explaining differences between tabular and sequence cohorts |
| Missing-data handling | `legacy/code/data_preprocessing/03_create_base.py` | Historical preprocessing normalizes before imputation and uses a `-99` sentinel for higher-missingness columns | Reviewer language should describe the code-defined rule, even if it differs from cleaner textbook phrasing |
| Reviewer-response clinical baselines | `src/inspire_aki/clinical_baselines/gs_aki.py`, `configs/clinical_baselines/intraperitoneal_proxy_map_5char.csv`, current AKI config | The maintained AKI pipeline now includes `asa_rule` plus an adapted GS-AKI baseline in the preop path | Adapted GS-AKI is the first maintained reviewer-feedback baseline extension; it is intentionally proxy-based rather than an exact recreation of the published source variables |
| Class imbalance | archived HPO/train scripts, current model code | Current maintained training uses explicit inverse-frequency weighting across trainable model families | This is directionally manuscript-aligned and more uniform than the mixed historical implementation |
| Evaluation design | archived HPO/train scripts, current configs and evaluation backends | The maintained shipped configs use patient-grouped evaluation modes built on `patient_id`; tuning still runs once before downstream evaluation | The design remains non-nested and should be described as such, but the maintained CLI closes the major patient-overlap leakage path from historical operation-level splitting |
| Calibration and thresholds | `legacy/code/create_results/13_performance_metrics.ipynb`, current evaluation/reporting code | Learned models still use isotonic calibration and F2-oriented thresholding, but the maintained rule baselines now keep prespecified thresholds | The maintained pipeline keeps repeated rows for the same `op_id` together during learned-model calibration CV, while `asa_rule` stays binary and `gs_aki_rule` stays an ordinal count/class baseline |
| Tabular versus deep performance | archived performance tables, current report outputs | The core repo story remains that tabular models outperform the maintained deep path on the checked-in historical results | Reviewer claims should cite specific current or archived report artifacts, not memory |
| SHAP and interpretability | `legacy/code/create_results/15_shap.ipynb`, `legacy/code/create_results/16_shap_batch.ipynb`, current SHAP reporting | The repo still preserves the historical interpretability workflows and a maintained CLI SHAP surface | Reviewer responses should distinguish maintained outputs from exploratory archive notebooks |

## Current CLI Manuscript Contract

- `report manuscript` is the top-level manuscript export command.
- Manuscript-facing tables are emitted as `html`, `md`, and `csv`.
- Manuscript-facing figures are emitted as `png` and `svg`.
- The maintained report layer supports consort, cohort tables, calibrated and uncalibrated performance tables, bootstrap CI tables, raw and FDR-corrected DeLong tables, DCA outputs, reclassification summaries, and SHAP outputs when configured.
- AKI compatibility aliases still exist through `compat export-legacy`, but they are explicit and AKI-only.
- Current shipped configs default to patient-grouped evaluation modes rather than the historical operation-level repeated-CV path.
- When stages are run manually, `evaluate generate` is the maintained entrypoint that materializes those patient-grouped manifests before tuning and training.
- Calibration in the maintained CLI is grouped on `op_id`, so repeated predictions from the same operation do not leak across isotonic calibration folds.
- The default AKI config now evaluates `asa_rule` and `gs_aki_rule` on the same maintained grouped manifests used by the learned models.
- `gs_aki_rule` is implemented as one adapted preoperative clinical baseline only; the current code does not refit or recalibrate a separate GS-AKI family of models.
- In manuscript-facing tables, `gs_aki_rule` should be treated primarily as an ordinal count/class score. Its threshold-dependent columns are intentionally blanked in the main performance table, while `asa_rule` remains a prespecified binary rule.

## Safe Reviewer Claims

- what the maintained CLI currently produces
- what the archived scripts and notebooks historically implemented
- where the maintained CLI intentionally fixes or tightens archived behavior
- where exact historical counts or point estimates may drift because of those fixes

For intentional behavior differences that matter scientifically or methodologically, see [legacy_cli_differences.md](legacy_cli_differences.md).

For the full source-backed implementation note behind the maintained adapted GS-AKI baseline, including the intraperitoneal proxy logic and explicit override table, see [gs_aki_adaptation.md](gs_aki_adaptation.md).
