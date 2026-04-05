# Ophthalmology Department Reviewer Note

## Executive Summary

- The original reviewer concern was sharpened by a reporting-label issue: the maintained human-readable labels had swapped `OS` and `OT`, although the underlying one-hot encoded feature columns were correct.
- True ophthalmology corresponds to raw `department == "OT"` and contributes 476 operations from 415 patients in the final analytic cohort.
- Ophthalmology-coded rows are overwhelmingly genuine eye procedures: 471 (98.9%) have ICD-10-PCS `08..` Eye body-system prefixes, and the few non-eye outliers are AKI-negative.
- The observed row-level AKI signal is most consistent with case mix rather than procedure mechanism: ophthalmology patients are older and have higher baseline renal markers, and repeated staged/bilateral eye operations inflate the row-level frequency (49 of 79 (62.0%)).

## Methods

- Loaded the maintained default config `configs/aki/default.yaml`.
- Joined the final labeled cohort (`cohort/labels.csv` and `datasets/tabular/tabular_combined_unnormalized.csv`) back to raw `operations.csv` on `op_id`.
- Used the active outcome `aki_boolean` from the maintained default config.
- Summarized ophthalmology-coded rows (`department == "OT"`) at both operation and patient level and enriched ICD-10-PCS prefixes with the configured CMS order reference.

## Summary Table

| Metric                                                 | Overall cohort                        | Ophthalmology                        | Ophthalmology AKI+                 | Ophthalmology AKI-                   |
| ------------------------------------------------------ | ------------------------------------- | ------------------------------------ | ---------------------------------- | ------------------------------------ |
| Operations, n                                          | 60,103                                | 476                                  | 79                                 | 397                                  |
| Patients, n                                            | 48,550                                | 415                                  | 53                                 | 362                                  |
| AKI-positive operations, n (%)                         | 2,047 (3.4%)                          | 79 (16.6%)                           | Reference                          | Reference                            |
| Patients with >=1 AKI-positive cohort op, n (%)        | 1,414 (2.9%)                          | 53 (12.8%)                           | Reference                          | Reference                            |
| Age, y, median (IQR)                                   | 60.0 (50.0-70.0)                      | 70.0 (60.0-75.0)                     | 70.0 (65.0-75.0)                   | 65.0 (60.0-75.0)                     |
| ASA, median (IQR)                                      | 2.0 (1.0-2.0)                         | 2.0 (2.0-2.0)                        | 2.0 (2.0-3.0)                      | 2.0 (2.0-2.0)                        |
| Preop creatinine, mg/dL, median (IQR)                  | 0.80 (0.68-0.92)                      | 0.92 (0.74-1.33)                     | 1.16 (0.80-1.69)                   | 0.88 (0.71-1.33)                     |
| Preop BUN, mg/dL, median (IQR)                         | 15.00 (12.00-18.00)                   | 17.00 (12.00-24.00)                  | 21.00 (16.00-29.00)                | 16.00 (12.00-23.25)                  |
| Op length, min, median (IQR)                           | 130.0 (80.0-215.0)                    | 30.0 (20.0-55.0)                     | 25.0 (17.5-42.5)                   | 30.0 (20.0-60.0)                     |
| Anesthesia distribution                                | MAC 725 (1.2%); General 52412 (87.2%) | MAC 342 (71.8%); General 134 (28.2%) | MAC 66 (83.5%); General 13 (16.5%) | MAC 276 (69.5%); General 121 (30.5%) |
| Eye ICD-10-PCS prefix (`08..`), n (%)                  | Not summarized                        | 471 (98.9%)                          | 79 (100.0%)                        | 392 (98.7%)                          |
| Non-eye outlier codes, n                               | Not summarized                        | 5                                    | 0                                  | 5                                    |
| AKI-positive OT ops on patients with >1 positive OT op | N/A                                   | 49 of 79 (62.0%)                     | 23 of 53 patients (43.4%)          | N/A                                  |

## Top Ophthalmology ICD-10-PCS Groups

| rank | ICD-10-PCS4 | OT ops | % of OT ops | AKI+ ops | AKI rate | Body system | Representative procedure      | Sample title                                                                           |
| ---- | ----------- | ------ | ----------- | -------- | -------- | ----------- | ----------------------------- | -------------------------------------------------------------------------------------- |
| 1    | 08DJ        | 178    | 37.4%       | 37       | 20.8%    | Eye         | Extraction of Right Lens      | Extraction of Right Lens, Percutaneous Approach                                        |
| 2    | 08R4        | 100    | 21.0%       | 17       | 17.0%    | Eye         | Replacement of Right Vitreous | Replacement of Right Vitreous with Autologous Tissue Substitute, Percutaneous Approach |
| 3    | 08RJ        | 63     | 13.2%       | 9        | 14.3%    | Eye         | Replacement of Right Lens     | Replacement of Right Lens with Autologous Tissue Substitute, Percutaneous Approach     |
| 4    | 08RK        | 49     | 10.3%       | 9        | 18.4%    | Eye         | Replacement of Left Lens      | Replacement of Left Lens with Autologous Tissue Substitute, Percutaneous Approach      |
| 5    | 08R5        | 18     | 3.8%        | 4        | 22.2%    | Eye         | Replacement of Left Vitreous  | Replacement of Left Vitreous with Autologous Tissue Substitute, Percutaneous Approach  |
| 6    | 08R9        | 11     | 2.3%        | 0        | 0.0%     | Eye         | Replacement of Left Cornea    | Replacement of Left Cornea with Autologous Tissue Substitute, Percutaneous Approach    |
| 7    | 08Q1        | 6      | 1.3%        | 0        | 0.0%     | Eye         | Repair Left Eye               | Repair Left Eye, External Approach                                                     |
| 8    | 08T0        | 5      | 1.1%        | 0        | 0.0%     | Eye         | Resection of Right Eye        | Resection of Right Eye, External Approach                                              |

## Interpretation

- The ophthalmology indicator is not capturing mislabeled orthopedic or other large-service cases; it mostly marks short lens, vitreous, and related eye procedures.
- The high row-level AKI frequency is partly inflated by repeated operations in a small subset of patients, which matters because the model and SHAP beeswarm are row-based.
- The subgroup also looks medically higher-risk than the full cohort, with older age and worse baseline renal labs, which is a more plausible explanation for the predictive signal than a direct nephrotoxic effect of ophthalmology itself.
- For manuscript interpretation, department indicators should therefore be framed as service/procedural case-mix variables rather than mechanistic AKI risk factors.

## Draft Reviewer Response

We audited the ophthalmology-coded rows directly in the final labeled analytic cohort by joining the final `labels.csv` and `tabular_combined_unnormalized.csv` artifacts back to raw `operations.csv` on `op_id`. The earlier reporting confusion reflected a human-readable `OS`/`OT` label swap; the underlying feature construction was correct, and true ophthalmology corresponds to raw `department == "OT"`. In the final cohort, ophthalmology accounted for 476 operations from 415 patients, and 471 (98.9%) were genuine eye procedures with ICD-10-PCS `08..` prefixes. The row-level AKI frequency was 79 (16.6%), but this corresponded to 53 (12.8%), and 49 of 79 (62.0%) occurred in patients with repeated positive ophthalmology rows, suggesting that staged or bilateral procedures amplified the row-level signal. These patients were older and more renally vulnerable than the overall cohort (ophthalmology median age 70.0 (60.0-75.0) years; preoperative creatinine 0.92 (0.74-1.33) mg/dL; preoperative BUN 17.00 (12.00-24.00) mg/dL), while the procedures themselves remained brief (30.0 (20.0-55.0) minutes) and predominantly MAC/general anesthesia cases (MAC 342 (71.8%); General 134 (28.2%)). Accordingly, we interpret the ophthalmology indicator as a surgical-service/case-mix proxy that helps partition a small, medically higher-risk subgroup rather than as a mechanistic renal effect of eye surgery itself. We will revise the Discussion to distinguish predictive importance from causal interpretation and to describe department indicators as administrative case-mix features.
