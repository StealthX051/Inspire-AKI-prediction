# Data Pipeline

This document describes the current code-defined data flow. It intentionally prefers what the scripts do today over what the manuscript may have intended.

## High-Level Flow

### Tabular path

1. `data_preprocessing/01_extract_preop.py`
2. `data_preprocessing/02_extract_intraop.py`
3. `data_preprocessing/03_create_base.py`
4. `data_preprocessing/04_AKI_data_selection.py`
5. `create_results/07_tabular_hpo.py`
6. `create_results/08_tabular_model_creation.py`

### Sequence path

1. `data_preprocessing/05_time_series_cleaner.py`
2. `data_preprocessing/06_create_lstm_trainable.py`
3. `create_results/09_lstm_hpo.py`
4. `create_results/10_lstm_model_creation.py`

### Evaluation path

- `create_results/11_consort.ipynb`
- `create_results/12_cohort_characteristics.ipynb`
- `create_results/13_performance_metrics.ipynb`
- `create_results/14_delong_table.ipynb`
- `create_results/15_shap.ipynb`
- `create_results/16_shap_batch.ipynb`

## Source Tables Expected by the Current Environment

The refactored package now defaults to the mounted INSPIRE volume under:

- `/media/volume/ncs_inspire_data/ncs_aki/data/inspire/operations.csv`
- `/media/volume/ncs_inspire_data/ncs_aki/data/inspire/labs.csv`
- `/media/volume/ncs_inspire_data/ncs_aki/data/inspire/vitals.csv`
- `/media/volume/ncs_inspire_data/ncs_aki/data/inspire/diagnosis.csv`
- `/media/volume/ncs_inspire_data/ncs_aki/data/inspire/ward_vitals.csv`

Historical note:

- the legacy numbered scripts were originally written against `/home/server/Projects/data/...`
- many stage-specific output paths below still reflect that older server-era workflow because those scripts have not all been refactored yet

## Stage-by-Stage Behavior

## Stage 1: Preoperative extraction

`data_preprocessing/01_extract_preop.py`

### Inputs

- `operations.csv`
- `labs.csv`
- `diagnosis.csv`
- `ward_vitals.csv`

### Core transformations

- Starts from operations-level rows and keeps a restricted column set.
- Derives:
  - `BSA`
  - `BMI`
  - `booking_case_length`
  - `num_card_events` from pre-op cardiovascular ICD-10-CM diagnoses starting with `I`
- Filters:
  - `asa < 6`
  - `age >= 18`
  - non-null `opstart_time` and `opend_time`
  - non-null and non-zero `height` and `weight`
  - excludes `antype == 'Regional'`
  - excludes `department == 'PED'`
  - excludes ICD-10-PCS prefixes `10`, `0TY`, `B50`, `B51`
- Encodes:
  - `sex` as boolean `M`
  - `antype`: `General -> 0`, `MAC -> 1`, `Neuraxial -> 1`
  - department via one-hot columns
- Uses `merge_asof` to take the latest value within 90 days before surgery for:
  - preop labs
  - ward vitals

### Output

- `/home/server/Projects/data/AKI/preop_data_test.csv`

### Notes

- The output filename here does not match some later scripts and the old README, which expect `preop_data.csv`.
- This file is the main source of preop cohort variables for the newer pipeline.

## Stage 2: Intraoperative tabular feature engineering

`data_preprocessing/02_extract_intraop.py`

### Inputs

- `vitals.csv`
- `/home/server/Projects/data/AKI/preop_data.csv` according to the script

### Core transformations

- Restricts to operations present in the preop table.
- Builds 24 regular intraoperative signals from:
  - 18 high-frequency labels
  - 6 medium-frequency labels
- Summarizes those signals with 8 statistics:
  - `mean`
  - `max`
  - `min`
  - `entropy`
  - `kurtosis`
  - `skew`
  - `trend`
  - `energy`
- Adds:
  - mean-only cross-sectional variables
  - weight- and operation-length-adjusted summed variables
  - operation-length-adjusted summed variables
  - `fluids_agg`
  - `equiv_MAC_totals`

### Output

- `/home/server/Projects/data/AKI/feature_engineered.csv`

### Notes

- The script uses `preop_data.csv`, while `01_extract_preop.py` writes `preop_data_test.csv`.
- `equiv_MAC_totals` is named as a total but the code computes the mean MAC-equivalent value over the interpolated operation window.

## Stage 3: Base dataset creation

`data_preprocessing/03_create_base.py`

### Inputs

- `/home/server/Projects/data/AKI/preop_data_test.csv`
- `/home/server/Projects/data/AKI/feature_engineered.csv`

### Merge behavior

- Preferred merge key is `op_id`.
- The legacy numbered script path was permissive about merge wiring.
- The refactored package path is stricter:
  - `src/inspire_aki/datasets/tabular.py` now requires `op_id` in both upstream frames
  - it fails fast instead of falling back to a `subject_id` to `op_id` join

### Core preprocessing

- Replaces `inf` and `-inf` with `NaN`.
- Drops likely leakage or ID fields where present:
  - `postop_creatinine`
  - `subject_id`
  - `opstart_time`
  - `opend_time`
  - `inhosp_death_time`
  - `allcause_death_time`
- Performs percentile-window outlier replacement for non-ignored numeric columns.
- Standardizes numeric columns with `StandardScaler`.
- Saves normalization statistics (`mean`, `var`).
- Missing-value handling:
  - `>=10%` missingness -> fill with `-99`
  - `0-10%` missingness -> `KNNImputer(n_neighbors=5)`

### Outputs

- `/home/server/Projects/data/base/tabular_combined.csv`
- `/home/server/Projects/data/base/tabular_preop.csv`
- `/home/server/Projects/data/base/tabular_intraop.csv`
- `/home/server/Projects/data/base/normalization_stats.csv`

### Notes

- The script comment itself warns that the normalize-then-impute order may be methodologically questionable.

## Stage 4: AKI label derivation

`data_preprocessing/04_AKI_data_selection.py`

### Inputs

- `labs.csv`
- `ward_vitals.csv`
- `/home/server/Projects/data/AKI/preop_data.csv`
- `/home/server/Projects/data/base/tabular_*.csv`

### Cohort restrictions inside the labeler

- Keeps only operations also present in the base combined table.
- Requires non-null `preop_creatinine`.
- Excludes `preop_creatinine >= 4.5`.
- Builds postoperative creatinine maxima in both:
  - 2-day window
  - 7-day window
- Merges dialysis from `ward_vitals.csv` rows where `item_name == 'crrt'`.
- Drops operations only if all three are missing:
  - `postop_creatinine_2_days`
  - `postop_creatinine_7_days`
  - `dialysis`

### Outputs

- `/home/server/Projects/data/AKI/tabular_combined.csv`
- `/home/server/Projects/data/AKI/tabular_preop.csv`
- `/home/server/Projects/data/AKI/tabular_intraop.csv`

## Stage 5: Sequence cleaning

`data_preprocessing/05_time_series_cleaner.py`

### Inputs

- `vitals.csv`
- `/home/server/Projects/data/AKI/preop_cleaned.csv`

### Core transformations

- Restricts to the 24 regular intraoperative signals.
- Deduplicates on `(op_id, chart_time, item_name)`.
- Replaces outliers per signal using percentile-window random replacement.
- For each operation:
  - pivots the signal set onto a 5-minute grid
  - fills within-operation missing values with the operation's column means
- Standardizes all sequence feature columns with `StandardScaler`.

### Output

- `/home/server/Projects/data/AKI/time_series_cleaned.csv`

### Notes

- This script depends on `preop_cleaned.csv`, which is not produced by the numbered preprocessing path.
- That file/path mismatch is a reproducibility issue, not a documentation typo.

## Stage 6: Sequence trainable dataset creation

`data_preprocessing/06_create_lstm_trainable.py`

### Inputs

- `/home/server/Projects/data/AKI/tabular_combined.csv`
- `/home/server/Projects/data/AKI/time_series_cleaned.csv`

### Core transformations

- Drops operations whose row-wise time-series presence ratio is `<= 1/4`.
- Fills remaining sequence `NaN`s with `0`.
- Casts boolean tabular columns to float.
- Groups sequence rows by `op_id`, converts to tensors, and pads only operations with `length < 200`.
- Drops operations with `length >= 200`.
- Stores:
  - `op_id`
  - padded time tensor
  - original `seq_len`
  - merged tabular features and AKI label

### Output

- `/home/server/Projects/data/AKI/lstm_trainable.pkl`

## Data Regimes Produced by the Repo

| Regime | Main file | Used by |
| --- | --- | --- |
| Preop tabular | `/home/server/Projects/data/AKI/tabular_preop.csv` | Tabular models |
| Intraop tabular | `/home/server/Projects/data/AKI/tabular_intraop.csv` | Tabular models |
| Combined tabular | `/home/server/Projects/data/AKI/tabular_combined.csv` | Tabular models and hybrid path |
| Cleaned intraop sequences | `/home/server/Projects/data/AKI/time_series_cleaned.csv` | Sequence preparation |
| Combined sequence + static dataset | `/home/server/Projects/data/AKI/lstm_trainable.pkl` | LSTM and hybrid models |

## Related but Non-Canonical Data Scripts

- `data_preprocessing/MACCE_data_selection.py`
  - Simpler MACCE label generation for the base datasets.
- `data_preprocessing/outcomes_data_selection.py`
  - Broader outcome pipeline for MACCE, pneumonia, pulmonary embolism, respiratory failure, extended LOS, postoperative ICU admission, and 30-day mortality.
- `create_results/11_consort.ipynb`
  - Re-implements cohort filtering logic in notebook form for figure-generation purposes rather than importing the numbered scripts.

## What Not To Assume

- Do not assume all numbered scripts line up cleanly on filenames.
- Do not assume the same exact cohort count is used by every downstream notebook.
- Do not assume the sequence branch and tabular branch operate on the same final patient set.
