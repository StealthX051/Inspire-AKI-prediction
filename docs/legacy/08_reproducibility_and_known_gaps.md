# Reproducibility and Known Gaps

This repository is readable and partially runnable, but it is not fully portable or cleanly reproducible as checked in.

## Current Real-Data Validation Status

As of March 24, 2026:

- the synthetic refactor test suite is green:
  - `pytest -q` passes with `46` tests
- the refactored real-data preprocessing path has been exercised successfully on the mounted INSPIRE volume through sequence construction
- the refactored HPO tuning path has now also been exercised successfully on real INSPIRE data after fixing Optuna `4.x` trial-state handling
- the full real-data `configs/aki/smoke_hpo.yaml` pipeline has **not** yet been validated in one uninterrupted run from preprocessing through manuscript reporting

That means the repo is now in a better state than “docs only” or “synthetic tests only,” but it is still not at the point where we should claim a fully validated refactored end-to-end rerun on INSPIRE.

## High-Impact Reproducibility Risks

| Issue | Evidence | Impact | Practical mitigation |
| --- | --- | --- | --- |
| Private raw data dependency | the repo still requires private INSPIRE tables; the refactor defaults now point at `/media/volume/ncs_inspire_data/ncs_aki/data/inspire`, while older scripts were developed against `/home/server/Projects/data/INSPIRE/...` | End-to-end reruns are impossible without private INSPIRE tables | Treat docs and checked-in outputs as the primary accessible artifact for external readers |
| Hard-coded absolute paths | throughout `01`-`10`, notebooks, and helper scripts | Fresh-machine execution will fail even with the right data if the path layout differs | Any future portability work should centralize paths into config/env vars |
| Legacy server-path drift still exists | the refactor defaults now point at the mounted volume, but many legacy scripts and notebooks still document or encode `/home/server/...` paths | The new CLI is easier to run here, but the legacy surface is still brittle | Prefer the refactored CLI and treat legacy paths as historical until they are fully migrated |
| Multiple historical environments | checked-in AutoGluon metadata spans at least `autogluon.tabular==1.2` and `1.3.1`; the repo now pins a baseline in `environment.yml` and `requirements.txt` | Fresh installs are easier, but exact historical artifact recreation is still not guaranteed | Use `environment.yml` as the baseline setup, and treat old artifacts as descriptive evidence rather than exact rerun targets |
| Resource-aware execution is host-dependent | the refactor now derives stage worker/thread budgets from detected CPU, RAM, and GPU resources | Runtime behavior is more scalable, but exact wall times and worker counts will differ by machine | Use `inspire-aki runtime inspect --config ...` to verify the resolved plan on the current host |
| Filename drift between stages | `01_extract_preop.py` writes `preop_data_test.csv`; later stages reference `preop_data.csv`; `05_time_series_cleaner.py` expects `preop_cleaned.csv` | The canonical pipeline does not connect perfectly as written | Document the drift explicitly and do not claim turnkey execution |
| Multiple cohort counts | manuscript brief, consort notebook, and performance notebook comments disagree | Easy to misstate study size or split sizes | Use a divergence table rather than forcing a single count |
| Intentional refactor cohort cleanup | the refactor now excludes `op_len <= 0`, while legacy scripts let zero-duration cases pass through | Refactor cohort counts can now differ slightly from both the manuscript and the older scripts | Treat this as a correctness fix and document the drift explicitly |
| Synthetic tests are broad and real-data validation is still partial | `pytest` now exercises the refactor package on synthetic data only, while current real-data validation has reached preprocessing and HPO tuning but not a full end-to-end HPO smoke rerun | The new package surface is safer, but scientific reruns on INSPIRE are still not fully proven here | Treat synthetic passing tests as contract checks and real-data smoke progress as partial validation only |
| Real-data end-to-end refactor validation is incomplete | preprocessing and HPO tuning have been exercised on real INSPIRE data, but the full `smoke_hpo` `train -> evaluate -> report` chain has not yet been completed in one uninterrupted run | It is still possible that downstream training/report integration issues remain | Resume from `inspire-aki train tabular --config configs/aki/smoke_hpo.yaml` or rerun the full HPO smoke wrapper |

## Medium-Impact Code Gaps

| Issue | Evidence | Impact | Notes |
| --- | --- | --- | --- |
| AutoGluon weight-column issue | `create_results/08_tabular_model_creation.py` sets `sample_weight='balance_weight'` without creating that column | The current branch may fail or differ from historical runs | Checked-in results may predate the current code snapshot |
| Normalize-before-impute order | explicit warning comment in `data_preprocessing/03_create_base.py` | Potential methodological mismatch | Current docs should describe it, not defend it |
| Sequence path cap at 200 steps | `data_preprocessing/06_create_lstm_trainable.py` | Long operations are dropped from the deep path | This can change the patient set relative to the tabular path |
| Zero-byte legacy script | `data_preprocessing/AKI_data_selection.py` | Confusing repo surface | Treat as debris, not source |
| Duplicated utility modules | three copies of `mlstatkit/toolkit.py` | Risk of silent drift between copies | Current docs mark them as duplicated copies |

## Artifact and History Risks

| Issue | Evidence | Impact | Notes |
| --- | --- | --- | --- |
| Checked-in model artifact directories | `AutogluonModels/`, `notebooks/mljar_results_improved/` | Easy to confuse outputs with source | Read as evidence of prior runs only |
| Heavy notebook surface | 47 notebooks across several folders | Intent is spread across exploratory code | Prefer numbered `.py` scripts for behavior, notebooks for figure-generation details |
| Legacy naming | `README.md` previously used `VitalDB-Dimensionality-Reduction` | Repo identity is inconsistent | Current docs normalize to `Inspire-AKI-prediction` while noting legacy names |

## What Can Be Reliably Done From the Repo Alone

- read the code
- inspect the current pipeline structure
- inspect checked-in performance and HTML outputs
- audit notebook inventory and artifact layout
- understand the current code-defined label and feature logic
- run the refactored synthetic pytest suite
- inspect the resolved stage runtime plan with `inspire-aki runtime inspect`
- inspect or resume the partially validated real-data HPO smoke artifacts under `artifacts/smoke_hpo/`

## What Cannot Be Reliably Reproduced From the Repo Alone

- raw-data cohort extraction
- full retraining of the models
- clean reruns of the figure notebooks
- exact manuscript cohort counts
- exact original environment used for the strongest checked-in results
- proof that the refactored CLI reproduces legacy real-data results exactly
- proof that the full refactored HPO smoke path has already completed cleanly on this instance

## Safe Claims vs Unsafe Claims

### Safe claims

- what a given script currently does
- what a notebook currently contains
- what checked-in markdown/html outputs currently report
- where the repo has drift or ambiguity

### Unsafe claims

- “the repo is fully reproducible”
- “the manuscript counts match the code exactly”
- “the current code snapshot necessarily produced the checked-in artifacts”
- “`environment.yml` guarantees exact recreation of every checked-in artifact”

## If Reproducibility Work Is Ever Requested Later

The likely first steps would be:

1. centralize all paths into config or environment variables
2. reconcile filename drift between preprocessing stages
3. run a clean-machine smoke test against `environment.yml`
4. decide which scripts/notebooks are canonical and archive the rest
5. reconcile the older `1.2` and newer `1.3.1` saved-model environments
6. add a minimal test or smoke-check layer for the numbered scripts

That work is intentionally out of scope for the current markdown-only pass.
