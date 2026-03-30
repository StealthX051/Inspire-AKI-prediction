# Current Refactor Docs

These docs describe the current refactored implementation that lives under `src/inspire_aki/` and is exposed through the `inspire-aki` CLI.

They are not the documentation for the historical numbered scripts.
For the pre-refactor system, manuscript-era assumptions, and legacy path issues, use [../legacy/README.md](../legacy/README.md).

Start here:

- [pipeline.md](pipeline.md)
- [../codex_workflow.md](../codex_workflow.md)
- [../refactor/behavior_drift.md](../refactor/behavior_drift.md)
- `configs/aki/default.yaml`
- `configs/macce/default.yaml`
- `src/inspire_aki/cli.py`

Project coordination notes live under `docs/`:

- [../TODO/README.md](../TODO/README.md)
- [../HANDOFF/README.md](../HANDOFF/README.md)
- [../HANDOFF/2026-03-30_grouped-cv-integration-merged-and-outcome-next-steps.md](../HANDOFF/2026-03-30_grouped-cv-integration-merged-and-outcome-next-steps.md)

Current outcome-extension note:

- the refactor now supports one active outcome per config/artifact root through `study.outcome_key`
- AKI remains the default shipped target, and MACCE is the first shipped grouped-holdout non-AKI config
