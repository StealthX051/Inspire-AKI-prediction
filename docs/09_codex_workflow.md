# Codex Workflow

This is the repo-specific workflow guide for Codex or any similar coding agent.

## Current Handoff State

As of March 24, 2026:

- the refactor test suite passes on synthetic data
- the real-data preprocessing chain on `/media/volume/ncs_inspire_data/ncs_aki/data/inspire` has been exercised through sequence creation
- the real-data HPO tuning chain now completes after patching Optuna `4.x` trial-state handling and tuning manifest completeness
- the remaining unvalidated step is a clean real-data `configs/aki/smoke_hpo.yaml` continuation from training through manuscript reporting

If resuming the current work rather than restarting from scratch, the recommended command sequence is:

```bash
source .venv/bin/activate
inspire-aki train tabular --config configs/aki/smoke_hpo.yaml
inspire-aki train sequence --config configs/aki/smoke_hpo.yaml
inspire-aki evaluate calibrate --config configs/aki/smoke_hpo.yaml
inspire-aki evaluate metrics --config configs/aki/smoke_hpo.yaml
inspire-aki evaluate delong --config configs/aki/smoke_hpo.yaml
inspire-aki evaluate dca --config configs/aki/smoke_hpo.yaml
inspire-aki report manuscript --config configs/aki/smoke_hpo.yaml
```

## First Principles

- Prefer code over prose when they disagree.
- Prefer numbered scripts over exploratory notebooks when both touch the same behavior.
- Prefer checked-in markdown/html result outputs over memory or guesswork when summarizing findings.
- Never pretend the repo is turnkey.

## Recommended Order Of Work

1. Read `README.md`.
2. Read `AGENTS.md`.
3. Read `docs/README.md`.
4. For data questions, read `docs/02_data_pipeline.md` and `docs/03_labels_and_features.md`.
5. For model/evaluation questions, read `docs/04_modeling_and_evaluation.md`.
6. For manuscript claims or disagreements, read `docs/07_manuscript_alignment.md`.
7. For execution feasibility, read `docs/08_reproducibility_and_known_gaps.md`.

## Preferred Sources By Question Type

| Question type | Preferred source |
| --- | --- |
| “How should I run the refactored pipeline?” | `src/inspire_aki/`, `configs/aki/default.yaml`, and `inspire-aki --help` |
| “What is the current pipeline?” | numbered scripts `01`-`10` |
| “How is AKI defined right now?” | `data_preprocessing/04_AKI_data_selection.py` |
| “What features exist?” | `01_extract_preop.py`, `02_extract_intraop.py`, `03_create_base.py`, `05_time_series_cleaner.py`, `06_create_lstm_trainable.py` |
| “How are models trained?” | `07_tabular_hpo.py`, `08_tabular_model_creation.py`, `09_lstm_hpo.py`, `10_lstm_model_creation.py` |
| “How are calibration / DCA / DeLong / SHAP done?” | `create_results/13`-`16`, `bootstrap_metrics.py`, `decision_curve.py` |
| “What did a previous run report?” | checked-in `create_results/*.md` and `*.html` outputs |

## Command Preferences

- Use `rg` / `rg --files` for search and inventory.
- Use targeted file reads, not whole-repo dumps.
- Prefer `inspire-aki ...` over directly reassembling legacy script chains when the task is forward-looking refactor work.
- Worker allocation in the refactor is centralized in `src/inspire_aki/runtime.py` and resolved per stage from detected CPU, RAM, and GPU resources.
- Use `inspire-aki runtime inspect --config ...` before large runs if the host class changed.
- Use `scripts/benchmark_runtime_profiles.sh` only as a non-CI benchmarking helper; it is not part of the canonical execution path.
- Treat `artifacts/predictions/raw/*.parquet` as the stage-owned prediction partitions and `artifacts/predictions/raw_predictions.parquet` as the combined evaluation view.
- Treat `reports.manuscript_sections` and `reports.shap_jobs` in `configs/aki/default.yaml` as the source of truth for report composition.
- Treat `/media/volume/ncs_inspire_data/ncs_aki/data/inspire` as the current default raw INSPIRE mount for the refactor.
- Treat `artifacts/staging/` as intentional refactor staging, not stray output, for the partitioned timeseries and sequence path.
- Treat notebooks as structured data:
  - inspect with `jq` or targeted text extraction
  - avoid editing them unless explicitly asked

## Things To Be Skeptical Of

- any assumption that later scripts consume earlier outputs without path drift
- any count mentioned only once in a notebook comment
- any package/setup instructions that ignore `environment.yml` or the saved `1.2` versus `1.3.1` environment drift
- any result that requires trusting checked-in model directories without corresponding code evidence
- any assumption that `run all` exports legacy aliases automatically; it does not
- any assumption that legacy `/home/server/...` paths are still the right default for this instance

## When To Update Docs

Update docs whenever:

- label logic changes
- feature derivation changes
- canonical script order changes
- model toggles or supported model families change
- portability improves or worsens
- manuscript-facing outputs materially change

Minimum doc update set after a behavior change:

- `README.md`
- the relevant `docs/*.md`
- `docs/07_manuscript_alignment.md` if the change affects a manuscript-facing claim
- `docs/refactor/behavior_drift.md` if the refactor intentionally deviates from the brittle legacy path

## When Not To Edit

- Do not edit checked-in model artifact directories unless explicitly asked.
- Do not bulk-clean exploratory notebooks as a side quest.
- Do not rewrite code for portability when the task is only documentation.
- Do not treat zero-byte or obviously stray files as active sources.

## Verification Loop For Future Changes

After any meaningful code change:

1. confirm the changed behavior in the relevant script/notebook
2. update the matching docs
3. re-check file links and command references
4. record any new drift or caveat instead of hiding it

## Why This Doc Exists

This repo fits the pattern where a short `AGENTS.md` is useful, but deeper task-specific docs are necessary because the research surface is too large and too messy to encode safely in one agent file. That structure matches current OpenAI/Codex guidance more closely than a single giant instruction file would.
