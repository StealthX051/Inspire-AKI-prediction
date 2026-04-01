# Archived Artifacts And Outputs

This note separates the cleaned repo’s current artifact story from the preserved historical one.

## Maintained Current Surface

- current CLI artifacts live under the configured `paths.artifacts_dir`
- current manuscript-facing report outputs are produced by `inspire-aki report ...`
- current docs for that surface live in [`../../docs/current/pipeline.md`](../../docs/current/pipeline.md)

## Archived In-Repo Outputs

Small historical reference outputs still kept in-repo now live under:

- `../reference_outputs/create_results/`

These are reference-only outputs preserved for manuscript and audit context.

## Externalized Historical Outputs

Large generated model and AutoML trees were removed from the primary repo surface during cleanup. See:

- [../externalized_artifacts.md](../externalized_artifacts.md)

## Reading Rule

- current CLI artifacts describe how the repo should be run now
- archived outputs describe what older workflows produced
