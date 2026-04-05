# Department `OS` Provenance Audit

## Executive Summary
- The final labeled analytic cohort contained 60,103 operations across 48,550 patients. `department_OS` matched raw `department == "OS"` exactly in the joined cohort with 0 row-level mismatches; `department_OT` matched raw `department == "OT"` with 0 mismatches.
- Raw INSPIRE `department.csv` defines `OS` as `Orthopedic Surgery` and `OT` as `Ophthalmology`. In the operation-level final cohort, `OS` contributed 14,790 operations (24.61%) with a `Postoperative AKI` rate of 1.22%, while `OT` contributed 476 operations (0.79%) with a `Postoperative AKI` rate of 16.60%.
- The current data support a reporting/manuscript labeling problem rather than a one-hot encoding bug. `department_OS` is an operation-level Orthopedic Surgery indicator, and its prominence is most consistent with surgical-service case mix rather than a direct renal mechanism.
- The currently mounted cohort table is patient-level by construction: it drops duplicate `subject_id` rows before counting departments. That aggregation, rather than the feature construction itself, explains why the cohort table and the operation-level model feature can appear inconsistent.

## Methods And Provenance
- Config loaded via `inspire_aki.config.load_config` from `configs/aki/default.yaml` with raw data rooted at `/media/volume/ncs_inspire_data/ncs_aki/data/inspire` and artifacts rooted at `/media/volume/ncs_inspire_data/ncs_aki/artifacts/default`.
- Code-path audit focused on the maintained implementation surface: `src/inspire_aki/cohort/filters.py`, `src/inspire_aki/cohort/preop.py`, `src/inspire_aki/reporting/tables.py`, and `src/inspire_aki/reporting/procedure_audit.py`.
- Operation-level audit cohort: `datasets/tabular/tabular_combined_unnormalized.csv` inner-joined to `cohort/labels.csv` on `op_id`, then joined back to raw `operations.csv` on `op_id`.
- Outcome column audited: `aki_boolean`.
- Manifest/config warnings: Manifest preprocess_preop.json has config_hash=04b9d84d5ebd51b7, which differs from the loaded config_hash=406addce996e53ff.; Manifest preprocess_labels.json has config_hash=04b9d84d5ebd51b7, which differs from the loaded config_hash=406addce996e53ff.
- No maintained anonymized operation-name or procedure-grouping variable was present in `operations.csv` or documented in `schema.csv`; interpretation therefore relied on raw department codes, anesthesia type, operation length, and ICD-10-PCS group summaries.

## Code-Path Findings
- In `src/inspire_aki/cohort/filters.py`, the preoperative cohort filter one-hot encodes raw `department` with `pd.get_dummies(..., columns=["department"])`.
- In `src/inspire_aki/cohort/preop.py`, the resulting `department_*` indicator columns are merged into the maintained preoperative feature artifact and then carried into the tabular modeling datasets.
- In the final labeled cohort, the joined data confirm that the feature columns still retain raw code identity rather than any downstream remapping.

| department_code | indicator_column | indicator_positive_n | raw_positive_n | mismatch_n | positive_mismatch_n | matches_exactly | mismatched_op_ids |
| --- | --- | --- | --- | --- | --- | --- | --- |
| OS | department_OS | 14790 | 14790 | 0 | 0 | yes |  |
| OT | department_OT | 476 | 476 | 0 | 0 | yes |  |

### Raw Dictionary Versus Current Maintained Code Labels
| department_code | raw_dictionary_label | current_report_label | current_procedure_audit_label |
| --- | --- | --- | --- |
| AN | Anesthesiology | Anesthesiology | Anesthesiology |
| CTS | Cardio-Thoracic Surgery | Cardio-Thoracic Surgery | Cardio-Thoracic Surgery |
| DM | Dermatology | Dermatology | Dermatology |
| EM | Emergency Medicine | Emergency Medicine | Emergency Medicine |
| GS | General Surgery | General Surgery | General Surgery |
| IM | Internal Medicine | Internal Medicine | Internal Medicine |
| NS | Neurosurgery | Neurosurgery | Neurosurgery |
| OG | Obstetrics & Gynecology | Obstetrics & Gynecology | Obstetrics & Gynecology |
| OL | Oto-laryngology | Oto-laryngology | Oto-laryngology |
| OS | Orthopedic Surgery | Orthopedic Surgery | Orthopedic Surgery |
| OT | Ophthalmology | Ophthalmology | Ophthalmology |
| PED | Pediatrics | Pediatrics | Pediatrics |
| PS | Plastic Surgery | Plastic Surgery | Plastic Surgery |
| RAD | Radiology | Radiology | Radiology |
| RO | Radiation Oncology | Radiation Oncology | Radiation Oncology |
| UR | Urology | Urology | Urology |

## OS Characterization
- The summary table below reports operation-level counts and rates for the requested slices.

| slice | slice_label | n_ops | pct_ops | n_patients | pct_patients | positive_n | positive_rate | op_len_median | op_len_q1 | op_len_q3 | op_len_mean | op_len_sd | antype_general_n | antype_general_pct | antype_mac_n | antype_mac_pct | antype_neuraxial_n | antype_neuraxial_pct | antype_regional_n | antype_regional_pct |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| overall | Overall cohort | 60,103 | 100.00% | 48,550 | 100.00% | 2,047 | 3.41% | 130 | 80 | 215 | 160.70 | 110.16 | 52,412 | 87.20% | 725 | 1.21% | 6,966 | 11.59% | 0 | 0.00% |
| OS | Orthopedic Surgery | 14,790 | 24.61% | 11,312 | 23.30% | 180 | 1.22% | 85 | 60 | 135 | 110.49 | 79.33 | 8,167 | 55.22% | 39 | 0.26% | 6,584 | 44.52% | 0 | 0.00% |
| OT | Ophthalmology | 476 | 0.79% | 415 | 0.85% | 79 | 16.60% | 30 | 20 | 55 | 46.44 | 45.68 | 134 | 28.15% | 342 | 71.85% | 0 | 0.00% | 0 | 0.00% |
| GS | General Surgery | 16,788 | 27.93% | 14,747 | 30.37% | 673 | 4.01% | 160 | 90 | 230 | 171.51 | 99.33 | 16,562 | 98.65% | 64 | 0.38% | 162 | 0.96% | 0 | 0.00% |
| NS | Neurosurgery | 9,282 | 15.44% | 7,860 | 16.19% | 83 | 0.89% | 150 | 90 | 235 | 173.94 | 105.11 | 9,241 | 99.56% | 39 | 0.42% | 2 | 0.02% | 0 | 0.00% |
| UR | Urology | 5,630 | 9.37% | 5,232 | 10.78% | 230 | 4.09% | 135 | 95 | 185 | 149.76 | 83.50 | 5,450 | 96.80% | 2 | 0.04% | 178 | 3.16% | 0 | 0.00% |

- The top raw `OS` ICD-10-PCS 4-character groups were:

| rank | icd10_pcs4 | n_ops | pct_of_os_ops | positive_n | positive_rate | representative_5char_prefix | body_system_desc | root_op_desc | example_long_title |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 0SBC | 813 | 5.496957403651115 | 7 | 0.8610086100861009 | 0SBC0 | Lower Joints | Excision | Excision of Right Knee Joint, Open Approach |
| 2 | 0SRA | 807 | 5.456389452332657 | 23 | 2.8500619578686495 | 0SRA0 | Lower Joints | Replacement | Replacement of Right Hip Joint, Acetabular Surface with Autologous Tissue Substitute, Open Approach |
| 3 | 0SBD | 727 | 4.915483434753211 | 5 | 0.687757909215956 | 0SBD0 | Lower Joints | Excision | Excision of Left Knee Joint, Open Approach |
| 4 | 0PQ0 | 687 | 4.645030425963489 | 11 | 1.6011644832605532 | 0PQ00 | Upper Bones | Repair | Repair Sternum, Open Approach |
| 5 | 0RG0 | 620 | 4.192021636240703 | 3 | 0.4838709677419355 | 0RG00 | Upper Joints | Fusion | Fusion of Occipital-cervical Joint with Autologous Tissue Substitute, Anterior Approach, Anterior Column, Open Approach |
| 6 | 0SRD | 612 | 4.137931034482759 | 2 | 0.32679738562091504 | 0SRD0 | Lower Joints | Replacement | Replacement of Left Knee Joint with Autologous Tissue Substitute, Lateral Meniscus, Open Approach |
| 7 | 0SRC | 601 | 4.063556457065585 | 0 | 0.0 | 0SRC0 | Lower Joints | Replacement | Replacement of Right Knee Joint with Articulating Spacer, Open Approach |
| 8 | 0PB0 | 556 | 3.759296822177147 | 4 | 0.7194244604316546 | 0PB00 | Upper Bones | Excision | Excision of Sternum, Open Approach |
| 9 | 009T | 477 | 3.2251521298174444 | 1 | 0.20964360587002095 | 009T0 | Central Nervous System and Cranial Nerves | Drainage | Drainage of Spinal Meninges with Drainage Device, Open Approach |
| 10 | 0LQ1 | 440 | 2.974983096686951 | 0 | 0.0 | 0LQ14 | Tendons | Repair | Repair Right Shoulder Tendon, Percutaneous Endoscopic Approach |

- As a control, the much smaller raw `OT` group was dominated by these ICD-10-PCS 4-character families:

| rank | icd10_pcs4 | n_ops | pct_of_ot_ops | positive_n | positive_rate | representative_5char_prefix | body_system_desc | root_op_desc | example_long_title |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 1 | 08DJ | 178 | 37.39495798319328 | 37 | 20.786516853932586 | 08DJ3 | Eye | Extraction | Extraction of Right Lens, Percutaneous Approach |
| 2 | 08R4 | 100 | 21.008403361344538 | 17 | 17.0 | 08R43 | Eye | Replacement | Replacement of Right Vitreous with Autologous Tissue Substitute, Percutaneous Approach |
| 3 | 08RJ | 63 | 13.235294117647058 | 9 | 14.285714285714286 | 08RJ3 | Eye | Replacement | Replacement of Right Lens with Autologous Tissue Substitute, Percutaneous Approach |
| 4 | 08RK | 49 | 10.294117647058824 | 9 | 18.367346938775512 | 08RK3 | Eye | Replacement | Replacement of Left Lens with Autologous Tissue Substitute, Percutaneous Approach |
| 5 | 08R5 | 18 | 3.7815126050420167 | 4 | 22.22222222222222 | 08R53 | Eye | Replacement | Replacement of Left Vitreous with Autologous Tissue Substitute, Percutaneous Approach |

## Patient-Vs-Operation Reconciliation
- The maintained cohort table computes `Total operations` from unique `op_id`, then drops duplicate `subject_id` rows before counting department indicators. That means the department rows in the current cohort table are patient-level counts, not operation-level counts.
- The rows below show which current artifact-table labels reproduce the raw `OS` and `OT` patient counts:

| department_code | expected_raw_label | expected_patient_finding | artifact_table_label | artifact_table_finding |
| --- | --- | --- | --- | --- |
| OS | Orthopedic Surgery | 10904 (22.46%) | Orthopedic Surgery | 10904 (22.46%) |
| OT | Ophthalmology | 365 (0.75%) | Ophthalmology | 365 (0.75%) |

## Interpretation
1. `department_OS` most likely represents `Orthopedic Surgery` in this analytic dataset. The raw INSPIRE dictionary, the feature-engineering code path, and the row-level joined data all point to the same conclusion.
2. The current `ophthalmology` interpretation for `department_OS` is not defensible. Ophthalmology corresponds to raw `OT`, not raw `OS`, in the INSPIRE department dictionary.
3. `department_OS` is likely acting as a case-mix proxy rather than a direct renal signal. It defines a large service slice with a distinctive anesthesia mix and procedure profile, and its `Postoperative AKI` rate (1.22%) is below the overall cohort rate (3.41%).
4. Nothing in this audit suggests data leakage or a one-hot feature-construction mistake. The engineered indicator columns preserve the raw department codes exactly in the final labeled cohort.
5. The smallest manuscript change is to relabel `department_OS` as Orthopedic Surgery, clarify that department indicators are administrative service/procedural case-mix features, and reconcile patient-level cohort-table counts against operation-level feature provenance.

## Draft Reviewer-Response Paragraph
We audited the provenance of `department_OS` directly against the maintained feature-engineering code and the raw INSPIRE operations table. In the current pipeline, department indicators are created by one-hot encoding the raw `department` field from `operations.csv`, and row-level verification in the final labeled cohort showed that `department_OS` maps exactly to raw `department == "OS"`. The public INSPIRE department dictionary defines `OS` as Orthopedic Surgery and `OT` as Ophthalmology. In our final analytic cohort, raw `OS` accounted for 14,790 operations with a `Postoperative AKI` rate of 1.22%, and its most common ICD-10-PCS groups were orthopedic/joint procedure families rather than ophthalmic procedures. We therefore interpret `department_OS` as an orthopedic surgical-service indicator and a case-mix proxy rather than a mechanistic kidney-risk factor. The inconsistency arose because the current cohort table aggregates department counts at the patient level and the previously generated manuscript-facing labels swapped `OS` and `OT`. This audit did not identify evidence of leakage or a feature-construction error, so the appropriate correction is to fix the label/writing and to describe the feature more carefully as a service-level case-mix marker.

## Draft Manuscript Revision
The engineered feature `department_OS` corresponds to the raw INSPIRE department code `OS`, which in the INSPIRE data dictionary denotes Orthopedic Surgery rather than Ophthalmology. We interpret this variable as an administrative surgical-service indicator that captures procedural case mix, anesthesia mix, and operation context rather than a direct mechanistic renal risk factor. Accordingly, we avoid overinterpreting this SHAP signal as a biologic AKI predictor and instead describe it as a service-level case-mix feature.

1. The likely meaning of `department_OS` is Orthopedic Surgery.
2. Yes; the manuscript previously mislabeled it, and the maintained code plus regenerated reviewer-facing outputs now use the corrected labels.
3. This requires wording/table fixes rather than a feature-construction correction.
