# Reproducibility

This repository is readable, testable, and partially runnable, but it is not fully reproducible from the repo alone.

## High-Impact Limitations

| Issue | Why it matters | Practical reading |
| --- | --- | --- |
| Private INSPIRE data dependency | End-to-end reruns require private source tables that are not checked in | External readers can inspect code, docs, and reference outputs, but not reproduce the study from scratch from the repo alone |
| Historical path brittleness | Archived scripts and notebooks were developed against older absolute-path layouts | The legacy archive is for audit/reference, not a supported execution surface |
| Host-dependent runtime planning | The maintained CLI resolves worker/thread budgets from the current machine | Behavior is more portable than the archive, but exact wall time and resource plans still vary by host |
| Non-nested evaluation design | Tuning runs once before downstream grouped evaluation | This is a methodological limitation, not a packaging bug |
| Leakage-control drift from the archive | The maintained CLI now uses patient-grouped evaluation manifests and `op_id`-grouped calibration rather than the older operation-level workflow | Small metric or count drift versus historical outputs should be read as an intentional correctness fix, not as evidence that the current docs are describing the wrong process |
| Adapted GS-AKI surgery proxy | INSPIRE does not ship a native intraperitoneal-surgery field, so the maintained GS-AKI baseline uses a repo-tracked 5-character ICD-10-PCS proxy map derived from CDC/NHSN and CMS resources plus explicit override review for residual observed code families | Treat GS-AKI as a maintained proxy-based clinical baseline rather than an exact source-variable reimplementation |
| Rule-baseline threshold policy | The maintained pipeline keeps prespecified thresholds for `asa_rule` and `gs_aki_rule` instead of optimizing them on evaluation data | Reviewer-facing binary metrics for these rules should be read as fixed-rule summaries, not tuned operating points |
| Archived-versus-current drift | The maintained CLI keeps several correctness and portability fixes relative to the archive | Exact historical point estimates and counts can drift even when the scientific intent is the same |
| Historical artifacts removed from the primary repo surface | Large generated model and AutoML trees were curated out of the main repo during cleanup | Small reference outputs remain in-repo; removed heavy artifacts are recorded in the legacy manifest |

## What The Repo Can Support Reliably

- reading the maintained package code and tests
- understanding the current CLI stage contracts
- inspecting reviewer-facing docs and archive notes
- inspecting small historical reference outputs kept under `legacy/reference_outputs/`
- running the synthetic test suite for the maintained package
- running the maintained CLI when the required private data and environment are available
- rerunning the maintained adapted GS-AKI baseline so long as the committed proxy map under `configs/clinical_baselines/` is kept in sync with the repo version being cited

## What The Repo Cannot Prove By Itself

- full raw-data cohort reconstruction without private INSPIRE tables
- exact recreation of the historical environments behind all checked-in legacy artifacts
- exact numerical equivalence between archived notebook-era results and the maintained CLI outputs
- a turnkey, data-free public rerun of the study

## Safe Claims

- what a maintained CLI command currently does
- what an archived script or notebook historically did
- which artifacts are current, archived, or externalized
- which workflow limitations are methodological versus operational
- that the maintained CLI uses patient-grouped evaluation and `op_id`-grouped calibration to reduce leakage relative to the archive
- that the maintained AKI config evaluates adapted GS-AKI on the same grouped manifests as the learned models, while still relying on a committed proxy map for the intraperitoneal factor
- that the maintained main performance table treats adapted GS-AKI primarily as an ordinal count/class baseline rather than a tuned binary rule

## Unsafe Claims

- that the repo is turnkey for external users
- that one historical artifact tree is itself the canonical source of truth
- that archived notebook-era results and current CLI results must match exactly
- that private-data-dependent steps are reproducible from the repo alone
- that the archived operation-level repeated-CV workflow is still the recommended validation design for current manuscript-facing runs

## Recommended Reviewer Posture

- cite `docs/current/` for the maintained interface
- cite `docs/reviewer/` for manuscript and limitation language
- cite [gs_aki_adaptation.md](gs_aki_adaptation.md) when the question is specifically about how the maintained adapted GS-AKI baseline and intraperitoneal proxy were implemented
- cite `legacy/` only when the historical workflow itself is the subject of the question
