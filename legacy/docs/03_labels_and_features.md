# Archived Labels And Features

This note captures the main label and feature logic from the archived workflow under `legacy/code/`.

## Historical AKI Label Logic

Primary archived labeler:

- `code/data_preprocessing/04_AKI_data_selection.py`

Key points:

- `aki_boolean` is effectively `aki_2 or aki_3`
- the implementation uses both a 2-day creatinine delta rule and 7-day ratio/max rules
- dialysis is included through `ward_vitals.csv` `crrt`

## Historical Feature Families

### Preop

- operation-level demographics and perioperative metadata
- derived variables such as `BMI`, `BSA`, `booking_case_length`, and `num_card_events`
- recent preop labs and ward-vital summaries

### Intraop tabular

- summary statistics for regular intraoperative signals
- sparse intraop variables and aggregate fluid/anesthetic features

### Sequence

- cleaned regular intraoperative time series
- padded sequence tensors capped at `200` steps in the archived preprocessing path

Use this file for archived logic lookup only. For the maintained CLI contract, use [`../../docs/current/pipeline.md`](../../docs/current/pipeline.md).
