# Archived Modeling And Evaluation

This file summarizes the historical modeling and evaluation surface preserved under `legacy/code/`.

## Archived Model Scripts

- tabular HPO: `code/create_results/07_tabular_hpo.py`
- tabular training: `code/create_results/08_tabular_model_creation.py`
- sequence HPO: `code/create_results/09_lstm_hpo.py`
- sequence training: `code/create_results/10_lstm_model_creation.py`

## Archived Notebook Evaluation Surface

- consort: `code/create_results/11_consort.ipynb`
- cohort tables: `code/create_results/12_cohort_characteristics.ipynb`
- calibration, thresholds, CI, DCA, reclassification: `code/create_results/13_performance_metrics.ipynb`
- DeLong testing: `code/create_results/14_delong_table.ipynb`
- SHAP: `code/create_results/15_shap.ipynb`, `code/create_results/16_shap_batch.ipynb`

## Important Archived Caveats

- the archived evaluation design is non-nested
- the archived workflow mixes scripts and notebooks heavily
- the archived sequence path drops cases longer than the fixed sequence cap
- archived model/result directories should be read as historical evidence, not primary source code

For the maintained CLI pipeline and current artifact contracts, use [`../../docs/current/pipeline.md`](../../docs/current/pipeline.md).
