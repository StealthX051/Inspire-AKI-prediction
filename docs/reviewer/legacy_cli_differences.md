# Legacy Versus CLI Differences

This file records the most important differences between the archived notebook/script workflow and the maintained `inspire-aki` CLI pipeline.

| Area | Archived behavior | Maintained CLI behavior | Why it matters |
| --- | --- | --- | --- |
| Cohort filtering | Archived preprocessing allowed some invalid operation durations to pass downstream | The CLI excludes `op_len <= 0` upstream | Small cohort-count drift is expected, but the CLI avoids invalid duration-normalized features |
| Intraop feature safety | Archived intraop feature engineering could carry fragile summary-stat and division edge cases forward | The CLI uses safer wrappers and fails the stage if non-finite values remain | This turns silent artifact-quality problems into explicit invariants |
| Tabular preprocessing outputs | Archived base preprocessing focused on the normalized tabular datasets plus normalization stats | The CLI also writes unnormalized combined data and fill-rate outputs | The maintained report layer has explicit inputs instead of notebook-local reconstruction |
| Label artifacts | Archived AKI labeling wrote labeled datasets directly | The CLI also writes standalone label and audit artifacts before downstream joins | Label reuse, auditability, and staged reporting are clearer |
| Split generation | Archived workflows often built splits inline inside model scripts | The CLI materializes split manifests and stage-owned manifests under the artifact root, with split audits alongside them | Split ownership is explicit, resumable, and easier to inspect during review |
| Default evaluation mode | Historical results often reflect operation-level repeated CV | Shipped CLI configs now default to patient-grouped holdout or grouped nested-CV modes keyed on `patient_id` | The maintained defaults keep one patient's operations from crossing train/test or train/validation boundaries, which closes a major leakage path |
| Default noncardiac cohort | Earlier maintained configs used the legacy noncardiac prefix filter as the default cohort and exposed the stricter cardiothoracic adjudication only as an alternate profile | Shipped CLI configs now default to the strict operation-level adjudicated noncardiac cohort, while the older less-strict profile is preserved only as an explicit legacy/debug option | Manuscript-facing cohort counts and cardiothoracic summaries must be regenerated after this promotion before they are cited |
| Manual grouped runs | Archived workflows did not have a dedicated grouped-manifest stage | The maintained CLI uses `evaluate generate` before grouped tuning/training when stages are run manually | The patient-grouped split policy is explicit rather than hidden inside downstream model code |
| Calibration CV | Archived calibration could pool repeated prediction rows without group protection | The CLI calibrates with grouped CV on `op_id` | Minor calibration drift is expected because the maintained path closes repeated-row leakage during isotonic fitting |
| Reporting surface | Archived reporting depended heavily on notebooks | The CLI owns manuscript tables, curves, consort, DeLong, DCA, reclassification, and optional SHAP outputs | Publication-facing outputs are now generated from stage-owned artifacts |
| Legacy exports | Historical outputs were often implicit filesystem handoffs | `compat export-legacy` is explicit, optional, and AKI-only | Compatibility support remains available without making archive paths primary again |
| Canonical outputs | Archived outputs were spread across notebook trees and saved result folders | The CLI writes canonical report outputs under the configured `reports/` directories and replaces them on rerun | The maintained surface is easier to review and less dependent on notebook state |

## Practical Reading Rule

Use the maintained CLI docs when describing how the repo should be run today. Use the legacy archive only when the historical workflow itself is part of the scientific or reviewer question.
