# Externalized Artifacts

This manifest records heavy generated artifacts that were removed from the primary repo surface during the CLI-first cleanup.

| Historical path | Artifact type | Current treatment | Recovery path |
| --- | --- | --- | --- |
| `AutogluonModels/` | top-level saved AutoGluon runs | removed from the cleaned repo surface | recover from git history or a separate project archive if needed |
| `legacy/code/create_results/AutogluonModels/` | legacy training artifact tree | removed from the cleaned repo surface | recover from git history or a separate project archive if needed |
| `legacy/code/notebooks/AutogluonModels/` | notebook-era saved AutoGluon runs | removed from the cleaned repo surface | recover from git history or a separate project archive if needed |
| `legacy/code/notebooks/mljar_results_improved/` | large notebook-era AutoML output tree | removed from the cleaned repo surface | recover from git history or a separate project archive if needed |
| `legacy/code/preoperative_models/aki_experiments/AutogluonModels/` | experiment-local saved model tree | removed from the cleaned repo surface | recover from git history or a separate project archive if needed |
| `logs/`, `smoke.log`, `smoke_hpo.log` | generated runtime logs | removed from versioned repo content | regenerate from new runs instead of treating them as source artifacts |

Small manuscript-facing reference outputs that remain in-repo now live under `legacy/reference_outputs/`.
