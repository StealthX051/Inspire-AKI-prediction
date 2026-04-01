# Archived Data Pipeline

This note summarizes the historical numbered-script workflow preserved under `legacy/code/`.

The paths below refer to the archived files now stored under `legacy/code/`.

## Historical Execution Order

### Tabular path

1. `code/data_preprocessing/01_extract_preop.py`
2. `code/data_preprocessing/02_extract_intraop.py`
3. `code/data_preprocessing/03_create_base.py`
4. `code/data_preprocessing/04_AKI_data_selection.py`
5. `code/create_results/07_tabular_hpo.py`
6. `code/create_results/08_tabular_model_creation.py`

### Sequence path

1. `code/data_preprocessing/05_time_series_cleaner.py`
2. `code/data_preprocessing/06_create_lstm_trainable.py`
3. `code/create_results/09_lstm_hpo.py`
4. `code/create_results/10_lstm_model_creation.py`

### Historical notebook-heavy evaluation path

- `code/create_results/11_consort.ipynb`
- `code/create_results/12_cohort_characteristics.ipynb`
- `code/create_results/13_performance_metrics.ipynb`
- `code/create_results/14_delong_table.ipynb`
- `code/create_results/15_shap.ipynb`
- `code/create_results/16_shap_batch.ipynb`

## Historical Environment Assumptions

- the archived workflow was tightly coupled to private INSPIRE data
- many archived scripts assume older absolute server-path layouts
- the archived chain should be treated as reference behavior, not a supported runnable interface

For the maintained pipeline, use [`../../docs/current/pipeline.md`](../../docs/current/pipeline.md).
