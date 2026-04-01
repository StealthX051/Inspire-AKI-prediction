# AGENTS.md

This file is the repo-wide operating contract for `Inspire-AKI-prediction`.

## Start Here

1. Read `README.md`.
2. Read `docs/README.md`.
3. Read `docs/current/README.md`.
4. Read `docs/current/pipeline.md`.
5. Treat `src/inspire_aki/`, `configs/`, `scripts/`, and `tests/` as the maintained implementation surface.

## Trust Order

- Highest trust:
  - `src/inspire_aki/`
  - `configs/`
  - `scripts/`
  - `tests/`
  - `pyproject.toml`
- High trust for reviewer/manuscript context:
  - `docs/reviewer/`
- Contributor-only coordination:
  - `docs/HANDOFF/`
  - `docs/TODO/`
  - `docs/codex_workflow.md`
- Archive-only reference:
  - `legacy/`

## What This Repo Is

- A research codebase for postoperative outcome prediction using INSPIRE data.
- A CLI-first package whose supported execution surface is `inspire-aki`.
- A maintained pipeline whose shipped configs use patient-grouped evaluation modes to reduce patient-overlap leakage; calibration also groups repeated rows by `op_id`.
- Not turnkey without private INSPIRE data.
- A repo that still preserves legacy scripts, notebooks, and historical outputs for audit and manuscript-reference work under `legacy/`.

## Current Interface

- Canonical entrypoint: `inspire-aki`
- Canonical code path: `src/inspire_aki/`
- Canonical docs: `docs/current/`
- Reviewer-facing support docs: `docs/reviewer/`
- Archive-only code and outputs: `legacy/`

## Safe Working Rules

- Prefer `rg` and targeted reads before editing.
- Prefer the smallest correct change that fully solves the task.
- Extend current CLI/package code paths before adding new abstractions.
- Do not rebuild a legacy-style path when the maintained CLI path can be updated safely.
- Keep diffs narrow and preserve current CLI/config/artifact contracts unless the request requires changing them.
- Do not casually edit notebooks when a maintained `.py` source or current doc already captures the behavior.
- Do not treat archived model/output trees as source code.
- Do not promise reproducibility without private INSPIRE data.

## Documentation Rules

When behavior changes, update the maintained docs first:

- `README.md`
- `docs/current/pipeline.md`
- `docs/reviewer/manuscript_alignment.md` if the change affects manuscript-facing claims
- `docs/reviewer/legacy_cli_differences.md` if the change intentionally diverges from archived behavior
- `docs/reviewer/reproducibility.md` if the change affects leakage controls, portability, or reviewer-facing limitations

Keep contributor notes selective:

- use `docs/HANDOFF/` for dated session handoffs that materially change next-step context
- use `docs/TODO/` for durable open tasks
- delete or consolidate superseded notes instead of accumulating near-duplicates

## Archive Rules

- `legacy/` is reference-only.
- Archived paths are not stable public interfaces.
- If a task needs parity or audit work, inspect the archive, but patch the maintained CLI path unless the request explicitly targets legacy material.
- If legacy context matters in user-facing docs, summarize it in `docs/reviewer/` rather than re-promoting archive docs as primary documentation.
- Do not describe the archived operation-level repeated-CV workflow as the recommended current process; the maintained default is patient-grouped evaluation plus grouped calibration.

## Deep Docs

- [docs/README.md](docs/README.md)
- [docs/current/README.md](docs/current/README.md)
- [docs/current/pipeline.md](docs/current/pipeline.md)
- [docs/reviewer/README.md](docs/reviewer/README.md)
- [docs/reviewer/manuscript_alignment.md](docs/reviewer/manuscript_alignment.md)
- [docs/reviewer/reproducibility.md](docs/reviewer/reproducibility.md)
- [docs/reviewer/legacy_cli_differences.md](docs/reviewer/legacy_cli_differences.md)
- [legacy/README.md](legacy/README.md)
- [docs/codex_workflow.md](docs/codex_workflow.md)
