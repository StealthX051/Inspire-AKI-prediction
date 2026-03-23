# Reproducibility and Known Gaps

This repository is readable and partially runnable, but it is not fully portable or cleanly reproducible as checked in.

## High-Impact Reproducibility Risks

| Issue | Evidence | Impact | Practical mitigation |
| --- | --- | --- | --- |
| Private raw data dependency | numbered scripts read `/home/server/Projects/data/INSPIRE/...` | End-to-end reruns are impossible without private INSPIRE tables | Treat docs and checked-in outputs as the primary accessible artifact for external readers |
| Hard-coded absolute paths | throughout `01`-`10`, notebooks, and helper scripts | Fresh-machine execution will fail even with the right data if the path layout differs | Any future portability work should centralize paths into config/env vars |
| Refactor uses config, but defaults still point at the server | `configs/aki/default.yaml` centralizes the paths, yet still defaults to `/home/server/...` | New CLI is more portable, but not self-contained unless config is overridden | Use a local override config for any non-server execution |
| Multiple historical environments | checked-in AutoGluon metadata spans at least `autogluon.tabular==1.2` and `1.3.1`; the repo now pins a baseline in `environment.yml` and `requirements.txt` | Fresh installs are easier, but exact historical artifact recreation is still not guaranteed | Use `environment.yml` as the baseline setup, and treat old artifacts as descriptive evidence rather than exact rerun targets |
| Filename drift between stages | `01_extract_preop.py` writes `preop_data_test.csv`; later stages reference `preop_data.csv`; `05_time_series_cleaner.py` expects `preop_cleaned.csv` | The canonical pipeline does not connect perfectly as written | Document the drift explicitly and do not claim turnkey execution |
| Multiple cohort counts | manuscript brief, consort notebook, and performance notebook comments disagree | Easy to misstate study size or split sizes | Use a divergence table rather than forcing a single count |
| Synthetic tests are broad, real-data validation is still absent | `pytest` now exercises the refactor package on synthetic data only | The new package surface is safer, but scientific reruns on INSPIRE are still unproven here | Treat synthetic passing tests as contract checks, not clinical reproduction |

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

## What Cannot Be Reliably Reproduced From the Repo Alone

- raw-data cohort extraction
- full retraining of the models
- clean reruns of the figure notebooks
- exact manuscript cohort counts
- exact original environment used for the strongest checked-in results
- proof that the refactored CLI reproduces legacy real-data results exactly

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
