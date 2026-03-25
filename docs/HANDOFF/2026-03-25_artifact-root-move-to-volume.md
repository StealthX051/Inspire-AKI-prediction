# Artifact Root Move To Volume

Date: 2026-03-25

## What changed

- moved the shipped refactor artifact roots off the repo-local root disk and onto the attached volume:
  - `configs/aki/default.yaml` -> `/media/volume/ncs_inspire_data/ncs_aki/artifacts/default`
  - `configs/aki/smoke.yaml` -> `/media/volume/ncs_inspire_data/ncs_aki/artifacts/smoke`
  - `configs/aki/smoke_hpo.yaml` -> `/media/volume/ncs_inspire_data/ncs_aki/artifacts/smoke_hpo`
- updated `configs/aki/default.yaml` so AutoGluon now uses all `32` host CPUs on this node via `models.autogluon.num_cpus: 32`
- updated `scripts/benchmark_runtime_profiles.sh` to resolve its benchmark output under the selected config's `artifacts_dir` instead of hardcoding repo-local `artifacts/benchmarks`
- updated `runtime benchmark --output-dir <relative>` so relative paths now resolve under the selected config's artifact root instead of the repo root
- updated path/docs/tests to describe the configured artifact root rather than assuming repo-local `artifacts/`

## Why

- the repo root is on `/dev/sda1` (`58G`, `95%` used at inspection time)
- the attached volume is mounted at `/media/volume/ncs_inspire_data` on `/dev/sdb` (`492G`, `1%` used at inspection time)
- the previous shipped configs wrote heavy runtime outputs under the repo-local `artifacts/` tree on the root disk

## Disk findings

- root-owned repo artifacts at inspection time:
  - `artifacts/` -> `31G`
  - `.venv/` -> `7.2G`
  - `/home/exouser/.cache/pip` -> `3.7G`
- largest refactor runtime subtree:
  - `artifacts/models/tabular/preop/autogluon` -> `8.6G`
- tabular model size breakdown seen in the existing root-owned run:
  - `autogluon` -> `8.6G`
  - `knn` -> `277M`
  - `rf` -> `21M`
  - `xgb` -> `3.5M`
  - `mlp`, `log_reg`, `asa_rule` -> negligible by comparison
- the observed AutoGluon storage blow-up is consistent with the current repeated-CV training layout plus `presets: best_quality`; existing fold directories contained many internal bagged model subtrees per outer fold

## Commands run

```bash
df -h / /media/volume/ncs_inspire_data
lsblk -f
du -sh artifacts .venv ~/.cache
du -sh artifacts/models/tabular/preop/*
.venv/bin/python - <<'PY'
from inspire_aki.config import load_config
for path in [None, 'configs/aki/smoke.yaml', 'configs/aki/smoke_hpo.yaml']:
    cfg = load_config(path)
    print((path or 'default'), cfg['paths']['artifacts_dir'])
PY
.venv/bin/pytest -q tests/test_config_and_artifacts.py tests/test_cli.py
```

## Verification

- config resolution now reports the attached-volume artifact roots above
- targeted tests passed: `16 passed`

## Next recommended step

1. move or delete the old repo-local `artifacts/` tree if those root-disk outputs are no longer needed
2. if disk growth on the attached volume is still a concern, consider a separate model-policy change for AutoGluon rather than another path fix
