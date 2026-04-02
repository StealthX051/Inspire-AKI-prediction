# Current CLI Docs

These docs describe the maintained package and CLI pipeline under `src/inspire_aki/`.

Start here:

- [pipeline.md](pipeline.md)
- `src/inspire_aki/cli.py`
- `configs/aki/default.yaml`
- `configs/macce/default.yaml`
- `tests/`

Key points:

- the supported execution surface is `inspire-aki`
- the shipped configs default to patient-grouped evaluation rather than the archived operation-level repeated-CV workflow
- the maintained evaluation path also groups calibration on `op_id` so repeated prediction rows from the same operation stay together
- the default AKI config now includes two maintained preop clinical baselines, `asa_rule` and `gs_aki_rule`
- stage outputs and reports are owned by the configured artifact root
- AKI remains the default shipped target
- legacy scripts and notebooks are preserved only for audit/reference under [../../legacy/README.md](../../legacy/README.md)
