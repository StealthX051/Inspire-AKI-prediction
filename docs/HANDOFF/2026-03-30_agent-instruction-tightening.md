# 2026-03-30 Agent Instruction Tightening

- Author: Codex
- Scope: tighten repo-level agent guidance to reduce over-editing during feature work

## What Changed

- Updated [AGENTS.md](/home/exouser/Inspire-AKI-prediction/AGENTS.md) to make the default policy explicit:
  - prefer the smallest correct change
  - reuse existing implementation surfaces before adding new helpers or layers
  - avoid opportunistic refactors and parallel feature reimplementation
  - keep verification proportional to the actual blast radius
- Updated [docs/codex_workflow.md](/home/exouser/Inspire-AKI-prediction/docs/codex_workflow.md) with the same operating policy plus guidance to use subtree `AGENTS.md` or `AGENTS.override.md` files when one area needs more specific instructions.

## Why

- Recent Codex feature work has been over-editing:
  - adding broader abstractions than required
  - reimplementing behavior that already exists
  - touching more files than necessary for the requested change
- The updated instruction set is meant to bias the default behavior toward surgical, professional, reuse-first edits rather than wide refactors.

## Next Use

- Keep the root `AGENTS.md` as the stable repo-wide contract.
- If a specific area such as reporting, runtime, or legacy audit work needs tighter local rules, add a subtree override instead of expanding the root guidance further.
