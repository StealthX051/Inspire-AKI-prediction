# Legacy Docs

This folder preserves the pre-refactor numbered documentation set.

These files are still useful for:

- understanding the older pipeline and notebook/script layout
- auditing manuscript-era behavior and assumptions
- checking parity or drift against the refactored implementation
- documenting historical issues and limitations in the legacy system

They are **not** the source of truth for the current refactored pipeline under `src/inspire_aki/`.
The refactor now has its own execution surface through the `inspire-aki` CLI, artifact layout under `artifacts/`, and behavior drift notes under `docs/refactor/`.

Use the numbered legacy docs when you explicitly need historical context:

- [01_repo_map.md](01_repo_map.md)
- [02_data_pipeline.md](02_data_pipeline.md)
- [03_labels_and_features.md](03_labels_and_features.md)
- [04_modeling_and_evaluation.md](04_modeling_and_evaluation.md)
- [05_artifacts_and_outputs.md](05_artifacts_and_outputs.md)
- [06_notebook_index.md](06_notebook_index.md)
- [07_manuscript_alignment.md](07_manuscript_alignment.md)
- [08_reproducibility_and_known_gaps.md](08_reproducibility_and_known_gaps.md)

If you need the current implementation instead, start with:

- [../current/README.md](../current/README.md)
- [../current/pipeline.md](../current/pipeline.md)
- [../codex_workflow.md](../codex_workflow.md)
- [../refactor/behavior_drift.md](../refactor/behavior_drift.md)
