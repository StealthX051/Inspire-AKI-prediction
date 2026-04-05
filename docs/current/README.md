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
- the maintained evaluation path groups learned-model calibration on `op_id` so repeated prediction rows from the same operation stay together
- the default AKI config now includes two maintained preop clinical baselines, `asa_rule` and `gs_aki_rule`
- `gs_aki_rule` is reported primarily as an ordinal count/class baseline rather than a calibrated probability model
- focused reviewer-response audit scripts under `scripts/` are part of the maintained surface when a question needs a narrow provenance or subgroup check without changing the CLI stage map
- stage outputs and reports are owned by the configured artifact root
- AKI remains the default shipped target
- legacy scripts and notebooks are preserved only for audit/reference under [../../legacy/README.md](../../legacy/README.md)
