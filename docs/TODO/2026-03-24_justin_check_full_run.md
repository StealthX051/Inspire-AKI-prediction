# Check the Overnight Full Run

- Author: nidhir
- Date: 2026-03-24
- Owner: justin
- Status: open

## Current Status

- Detached `tmux` session `aki_full` was launched on `2026-03-24 07:48:49 UTC` to run the full default pipeline.
- Config: [configs/aki/default.yaml](/home/exouser/Inspire-AKI-prediction/configs/aki/default.yaml)
- Logs:
  - [/home/exouser/Inspire-AKI-prediction/logs/runtime_inspect_full_aki.log](/home/exouser/Inspire-AKI-prediction/logs/runtime_inspect_full_aki.log)
  - [/home/exouser/Inspire-AKI-prediction/logs/full_aki_run.log](/home/exouser/Inspire-AKI-prediction/logs/full_aki_run.log)

## Commands

- `tmux ls`
- `tmux attach -t aki_full`
- `tmux capture-pane -pt aki_full`
- `tmux capture-pane -pt aki_full -S -200`
- `tail -f /home/exouser/Inspire-AKI-prediction/logs/full_aki_run.log`

## Done Criteria

- Confirm whether `inspire-aki run all --config configs/aki/default.yaml` finished successfully.
- If it failed, identify the failing stage and capture the relevant traceback or log snippet.
- If it succeeded, note the main output and manifest locations under `artifacts/`.
