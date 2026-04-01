# Review `legacy_cli_differences.md` for Intentionality

- Author: nidhir
- Date: 2026-03-24
- Owner: justin
- Status: open

## Current Status

- [docs/reviewer/legacy_cli_differences.md](/home/exouser/Inspire-AKI-prediction/docs/reviewer/legacy_cli_differences.md) now carries the maintained summary of legacy-versus-CLI drift after reviewing many package modules against the archived pipeline.

## Suggested Focus

- Rows that call out likely accidental drift or reproducibility regressions.
- Evaluate/report rows that may deserve code fixes instead of long-term documented drift.
- Any entries that are overcalled and should be removed.

## Done Criteria

- Decide which documented drifts are intentional and acceptable.
- Identify which rows should become follow-up code fixes.
- Trim any entries that are not real drift after closer review.
