# Labels and Features

This document records the confirmed current label and feature logic from code.

## AKI Label Logic

The current canonical labeler is `data_preprocessing/04_AKI_data_selection.py`.

## Confirmed fields used by the labeler

- `preop_creatinine`
- `postop_creatinine_2_days`
- `postop_creatinine_7_days`
- `dialysis` from `ward_vitals.csv` `crrt`

## Confirmed stage logic

Let:

- `crt_7_day_ratio = postop_creatinine_7_days / preop_creatinine`

Then the script defines:

- `aki_1`
  - `((crt_7_day_ratio > 1.5) and (crt_7_day_ratio < 2))`
  - or `((postop_creatinine_2_days - preop_creatinine) > 0.3)`
- `aki_2`
  - `(crt_7_day_ratio >= 2) and (crt_7_day_ratio < 3)`
- `aki_3`
  - `(crt_7_day_ratio >= 3)`
  - or `(postop_creatinine_7_days > 4)`
  - or `dialysis`
- `aki_boolean`
  - `aki_2 or aki_3`

## Important implications

- The current binary target is effectively severe AKI (`stage 2/3`) rather than any AKI.
- The code **does include dialysis** in stage 3.
- The code uses both a 2-day delta rule and 7-day ratio/max rules.

## Preoperative Feature Set

The main preop extraction is in `data_preprocessing/01_extract_preop.py`.

## Core operation-level variables kept up front

- `op_id`
- `subject_id`
- `age`
- `sex`
- `height`
- `weight`
- `asa`
- `emop`
- `opstart_time`
- `opend_time`
- `inhosp_death_time`
- `allcause_death_time`
- `orin_time`
- `orout_time`

## Derived preop/static variables

- `BSA`
- `BMI`
- `booking_case_length = orout_time - orin_time`
- `op_len = opend_time - opstart_time`
- `num_card_events`
  - count of pre-op cardiovascular diagnoses with ICD-10-CM code prefix `I`
- encoded `sex`
  - `True` for male
- encoded `antype`
  - `General -> 0`
  - `MAC -> 1`
  - `Neuraxial -> 1`
- one-hot `department_*` columns

## Preop lab features merged within 90 days

- `preop_total_protein`
- `preop_sodium`
- `preop_potassium`
- `preop_platelet`
- `preop_glucose`
- `preop_wbc`
- `preop_alt`
- `preop_chloride`
- `preop_lymphocyte`
- `preop_phosphorus`
- `preop_albumin`
- `preop_fibrinogen`
- `preop_creatinine`
- `preop_ptinr`
- `preop_total_bilirubin`
- `preop_alp`
- `preop_aptt`
- `preop_calcium`
- `preop_bun`
- `preop_ast`
- `preop_crp`
- `preop_hb`
- `preop_hct`
- `preop_seg`

## Ward features merged within 90 days

- `ward_spo2`
- `ward_bt`
- `ward_rr`
- `ward_nibp_dbp`
- `ward_nibp_sbp`
- `ward_hr`

## Intraoperative Tabular Features

The main intraop tabular feature engineering is in `data_preprocessing/02_extract_intraop.py`.

## Regular signals summarized with 8 statistics

The script builds a pivot table over 24 signals using:

- `mean`
- `max`
- `min`
- `entropy`
- `kurtosis`
- `skew`
- `trend`
- `energy`

### High-frequency labels

- `rr`
- `hr`
- `spo2`
- `fio2`
- `pmean`
- `etco2`
- `peep`
- `pip`
- `art_mbp`
- `cpat`
- `vt`
- `art_sbp`
- `art_dbp`
- `minvol`
- `pplat`
- `bt`
- `etgas`
- `cvp`

### Medium-frequency labels

- `pap_mbp`
- `pap_sbp`
- `pap_dbp`
- `nibp_mbp`
- `nibp_dbp`
- `nibp_sbp`

This yields up to `24 * 8 = 192` regular-summary columns before later filtering.

## Mean-only sparse variables

- `bis`
- `ci`
- `rfti`
- `dobui`
- `mlni`
- `ppfi`
- `o2`
- `air`
- `cbro2`
- `ntgi`

## Weight- and operation-length-adjusted summed variables

- `eph`
- `mdz`
- `ppf`
- `sft`

The code divides each measurement by `weight * op_len` before summing within operation.

## Operation-length-adjusted summed variables

- `n2o`
- `ebl`
- `rbc`
- `uo`
- `ftn`
- `ffp`
- `pc`
- `cryo`
- `pheresis`

The code divides each measurement by `op_len` before summing within operation.

## Additional aggregated variables

- `fluids_agg`
  - summed from `d5w`, `hes`, `psa`, `hs`, `ns`, `hns`, `alb20`, `alb5`, `d10w`, `d50w`
  - divided by `op_len`
- `equiv_MAC_totals`
  - derived from `etdes` and `etsevo`
  - the code forward-fills onto a 5-minute grid
  - computes `(des / 6) + (sevo / 2)`
  - stores the **mean** of that interpolated series, despite the variable name implying a total

## Base Preprocessing Rules

The main base preprocessing is in `data_preprocessing/03_create_base.py`.

## Ignore list during outlier handling and normalization

- `op_id`
- `age`
- `emop`
- `num_card_events`
- `antype`
- `sex`
- `asa`
- any `department_*`
- any `*_isna`
- any `*aki*`

## Outlier handling

For eligible numeric columns:

- values `< 1st percentile` are replaced with a random value sampled between the `0.5th` and `5th` percentiles
- values `> 99th percentile` are replaced with a random value sampled between the `95th` and `99.5th` percentiles

## Missing-data handling

- `>=10%` missing -> fill with `-99`
- `0-10%` missing -> `KNNImputer(n_neighbors=5)`

## Sequence Feature Logic

The sequence path uses `data_preprocessing/05_time_series_cleaner.py` and `data_preprocessing/06_create_lstm_trainable.py`.

## Signals used in the sequence path

The same 24 regular intraoperative variables listed above.

## Cleaning behavior

- deduplicate repeated `(op_id, chart_time, item_name)` rows
- outlier replacement per signal
- interpolate onto a 5-minute grid from operation-specific min to max time
- fill missing values within operation with per-operation column means
- standardize sequence columns globally

## Sequence selection and padding behavior

- compute a per-row `presence` score as non-null feature fraction over 24 sequence variables
- drop rows with `presence <= 1/4`
- fill remaining `NaN`s with `0`
- keep only operations with sequence length `< 200`
- pad shorter sequences with zeros to length `200`

## Unresolved or Risky Feature Questions

These are real implementation ambiguities that should not be smoothed over in downstream docs or analysis.

- `02_extract_intraop.py` expects `preop_data.csv`, while `01_extract_preop.py` writes `preop_data_test.csv`.
- `05_time_series_cleaner.py` expects `preop_cleaned.csv`, which the numbered path does not generate.
- `equiv_MAC_totals` is named as a total but computed as a mean over the interpolated case-level series.
- The repo does not expose a single definitive feature dictionary for every final modeled column after one-hot expansion and preprocessing.
- The exact final feature counts can vary by stage because later filtering and merge behavior remove rows and potentially columns.
