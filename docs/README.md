# Docs Index

This directory separates current refactor-oriented docs from the archived numbered legacy docs.

## Current Docs

For the current implementation, start with:

- [current/README.md](current/README.md)
- [current/pipeline.md](current/pipeline.md)
- [../README.md](../README.md)
- [../AGENTS.md](../AGENTS.md)
- `src/inspire_aki/`
- [codex_workflow.md](codex_workflow.md)
- [refactor/behavior_drift.md](refactor/behavior_drift.md)

Project coordination notes live under:

- [TODO/README.md](TODO/README.md)
- [HANDOFF/README.md](HANDOFF/README.md)

## Legacy Docs

The numbered `01`-`08` documents now live under [legacy/README.md](legacy/README.md).
They remain useful for understanding the pre-refactor system and its issues, but they will drift from the current implementation under `src/inspire_aki/`.

## Current Handoff Snapshot

As of March 24, 2026:

- synthetic `pytest` coverage for the refactor is green
- the real-data preprocessing path on the mounted INSPIRE volume has been exercised
- the real-data HPO tuning path now completes after Optuna `4.x` state-handling fixes
- the full real-data `configs/aki/smoke_hpo.yaml` run is still pending final end-to-end validation from training through report

For the current portability/validation status, start with:

- [legacy/08_reproducibility_and_known_gaps.md](legacy/08_reproducibility_and_known_gaps.md)
- [codex_workflow.md](codex_workflow.md)

## Recommended Reading Order

1. [../README.md](../README.md)
2. [../AGENTS.md](../AGENTS.md)
3. [current/README.md](current/README.md)
4. [current/pipeline.md](current/pipeline.md)
5. `src/inspire_aki/`
6. [codex_workflow.md](codex_workflow.md)
7. [refactor/behavior_drift.md](refactor/behavior_drift.md)
8. [legacy/README.md](legacy/README.md)
9. [legacy/07_manuscript_alignment.md](legacy/07_manuscript_alignment.md)
10. [legacy/08_reproducibility_and_known_gaps.md](legacy/08_reproducibility_and_known_gaps.md)

## By Task

### Understand current refactor pipeline

- [current/README.md](current/README.md)
- [current/pipeline.md](current/pipeline.md)
- [refactor/behavior_drift.md](refactor/behavior_drift.md)

### Understand repo structure

- [legacy/01_repo_map.md](legacy/01_repo_map.md)

### Understand data construction

- [legacy/02_data_pipeline.md](legacy/02_data_pipeline.md)
- [legacy/03_labels_and_features.md](legacy/03_labels_and_features.md)

### Understand training and evaluation

- [legacy/04_modeling_and_evaluation.md](legacy/04_modeling_and_evaluation.md)
- [legacy/05_artifacts_and_outputs.md](legacy/05_artifacts_and_outputs.md)

### Find notebooks quickly

- [legacy/06_notebook_index.md](legacy/06_notebook_index.md)

### Reconcile code vs manuscript

- [legacy/07_manuscript_alignment.md](legacy/07_manuscript_alignment.md)

### Assess portability and risk

- [legacy/08_reproducibility_and_known_gaps.md](legacy/08_reproducibility_and_known_gaps.md)

### Understand runtime artifacts

- [legacy/05_artifacts_and_outputs.md](legacy/05_artifacts_and_outputs.md)
- [refactor/behavior_drift.md](refactor/behavior_drift.md)

### Work on the repo with Codex or another agent

- [current/pipeline.md](current/pipeline.md)
- [codex_workflow.md](codex_workflow.md)
- [refactor/behavior_drift.md](refactor/behavior_drift.md)

### Leave explicit notes for the next person

- [TODO/README.md](TODO/README.md)
- [HANDOFF/README.md](HANDOFF/README.md)
