# Contributor Workflow

This file is contributor-only guidance for Codex or another coding agent working inside the repo.

## Recommended Reading Order

1. `README.md`
2. `AGENTS.md`
3. `docs/current/README.md`
4. `docs/current/pipeline.md`
5. `docs/reviewer/README.md` when manuscript or rebuttal context matters
6. `legacy/README.md` only for audit or parity work

## Preferred Sources By Question Type

| Question | Preferred source |
| --- | --- |
| How should the repo be run today? | `src/inspire_aki/`, `configs/`, `inspire-aki --help`, `docs/current/pipeline.md` |
| What is the maintained package contract? | `src/inspire_aki/` and `tests/` |
| What should a manuscript or reviewer response claim? | `docs/reviewer/` |
| What did the historical scripts or notebooks do? | `legacy/` |

## Command Preferences

- Use `rg` / `rg --files` for search.
- Prefer targeted reads over whole-repo dumps.
- Prefer `inspire-aki ...` over rebuilding archived script chains.
- Use `inspire-aki runtime inspect --config ...` before large runs on a new host class.
- Treat notebooks as reference material unless a task explicitly requires editing them.

## Edit Policy

- Default to the smallest correct diff.
- Patch the maintained CLI path before touching archive material.
- Avoid opportunistic renames, moves, or cleanup outside the requested change unless they are required.
- Keep validation proportional to the change.

## Doc Update Policy

After a meaningful behavior change, update:

- `README.md`
- `docs/current/pipeline.md`
- `docs/reviewer/manuscript_alignment.md` if manuscript-facing behavior changed
- `docs/reviewer/legacy_cli_differences.md` if the maintained pipeline intentionally diverges from archive behavior
- `docs/reviewer/reproducibility.md` if leakage controls, portability, or reviewer-facing limitations changed

## What Not To Treat As Source Of Truth

- archived model directories
- exploratory notebooks
- stale handoff notes
- comments that disagree with code or checked-in current docs
- legacy operation-level repeated-CV habits when documenting the maintained CLI workflow
