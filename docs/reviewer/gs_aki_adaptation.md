# Adapted GS-AKI Implementation Note

This note is the detailed reviewer-facing record of how the maintained pipeline implements the adapted GS-AKI baseline. It is intentionally more detailed than the manuscript methods text so it can be used as source material for a supplement or reviewer response.

## Scope

We implemented exactly one GS-AKI baseline in the maintained AKI pipeline:

- model key: `gs_aki_rule`
- display name: `Adapted GS-AKI`
- outcome support: AKI only
- dataset regime: preop only
- implementation surface:
  - runtime derivation: [src/inspire_aki/clinical_baselines/gs_aki.py](/home/exouser/Inspire-AKI-prediction/src/inspire_aki/clinical_baselines/gs_aki.py)
  - intraperitoneal map builder: [src/inspire_aki/clinical_baselines/intraperitoneal_map_builder.py](/home/exouser/Inspire-AKI-prediction/src/inspire_aki/clinical_baselines/intraperitoneal_map_builder.py)
  - maintainer wrapper: [scripts/build_intraperitoneal_proxy_map.py](/home/exouser/Inspire-AKI-prediction/scripts/build_intraperitoneal_proxy_map.py)
  - committed proxy map: [configs/clinical_baselines/intraperitoneal_proxy_map_5char.csv](/home/exouser/Inspire-AKI-prediction/configs/clinical_baselines/intraperitoneal_proxy_map_5char.csv)

We did not implement:

- the original coefficient-based logistic GS-AKI model
- multiple GS-AKI variants
- diabetes treatment subtype from medications
- any GS-AKI refit or recalibration model
- GS-AKI outside the maintained preop comparison path

## Why This Is Adapted

The original GS-AKI was developed as a preoperative bedside risk index in general surgery.[1] We adapted it for INSPIRE rather than claiming exact source-variable reproduction because INSPIRE does not expose all original source constructs in the same form.

The main reasons were:

- INSPIRE diagnoses are truncated to 3-character ICD-10-CM codes in the public documentation and Science Data paper.[2][3]
- INSPIRE procedure-code documentation is inconsistent in public sources:
  - the PhysioNet v1.3 page describes `operations.icd10_pcs` as the initial 5 characters of ICD-10-PCS.[2]
  - the Science Data paper describes operation names as converted to the first 4 ICD-10-PCS characters.[3]
- the same public sources are also not fully aligned on diagnosis timing windows around time zero.[2][3]
- INSPIRE does not provide a defensible outpatient medication history suitable for reconstructing the original NSQIP diabetes-treatment split.

For that reason, the maintained implementation uses one explicit label throughout the code and docs:

- `Adapted GS-AKI`

## Source Documents Used

The implementation and documentation were grounded in the following sources:

1. Original GS-AKI development paper: Kheterpal et al., *Development and validation of an acute kidney injury risk index for patients undergoing general surgery: results from a national data set*.[1]
2. INSPIRE PhysioNet dataset page, version 1.3.[2]
3. INSPIRE Science Data paper.[3]
4. CDC NHSN SSI procedure-code FAQ, especially the ICD-10-PCS approach-character and HYST/VHYS guidance.[4]
5. CMS ICD-10 files page, used to source the official 2026 ICD-10-PCS order-file and code-table downloads.[5]

## High-Level Design Decision

We implemented the simplified grouped bedside form, not the original regression coefficients.

The final adapted count uses 9 binary grouped factors:

1. age `>= 56`
2. male sex
3. emergency surgery
4. intraperitoneal surgery proxy
5. diabetes diagnosis-history proxy
6. CHF proxy
7. ascites proxy
8. hypertension proxy
9. renal insufficiency from preoperative creatinine `>= 1.2 mg/dL`

The final class mapping follows the published simplified cutpoints:

- `I`: `0-2`
- `II`: `3`
- `III`: `4`
- `IV`: `5`
- `V`: `>=6`

In code, these cutpoints are configured in [config.py](/home/exouser/Inspire-AKI-prediction/src/inspire_aki/config.py) and shipped in [default.yaml](/home/exouser/Inspire-AKI-prediction/configs/aki/default.yaml).

## Exact Variable Mapping

| Adapted factor | Runtime column | Source | Exact rule in maintained code | Adaptation note |
| --- | --- | --- | --- | --- |
| Age | `gs_aki_age_ge_56` | `preop_features.csv` | `age >= 56` | direct |
| Male sex | `gs_aki_male` | `preop_features.csv` | normalized sex indicator | direct |
| Emergency surgery | `gs_aki_emergency` | `preop_features.csv` | `emop == 1` | direct |
| Intraperitoneal surgery | `gs_aki_intraperitoneal` | `operations.csv` + committed map | binary join on `icd10_pcs_5char` | proxy |
| Diabetes | `gs_aki_diabetes` | `diagnosis.csv` | any preop ICD-10-CM prefix in `E08,E09,E10,E11,E13` | proxy; no treatment subtype |
| Hypertension | `gs_aki_hypertension` | `diagnosis.csv` | any preop ICD-10-CM prefix in `I10,I11,I12,I13,I15,I16,I1A` | proxy |
| CHF | `gs_aki_chf_30d` | `diagnosis.csv` | any ICD-10-CM prefix `I50` with `chart_time < opstart_time` and within 30 days preop | proxy |
| Ascites | `gs_aki_ascites_30d` | `diagnosis.csv` | any ICD-10-CM prefix `R18` with `chart_time < opstart_time` and within 30 days preop | proxy |
| Renal insufficiency | `gs_aki_renal_insufficiency` | `preop_features.csv` | `preop_creatinine >= 1.2` | grouped simplification |

Audit-only renal columns are also written:

- `gs_aki_renal_mild`: `1.2 <= creatinine < 2.0`
- `gs_aki_renal_moderate`: `creatinine >= 2.0`

These are used only for audit and sanity checking. The scored renal factor is still the single grouped indicator `preop_creatinine >= 1.2`.

## Timing And Leakage Rules

The maintained implementation is operation-specific and preoperative only.

Exact timing rules:

- indexed operation fields come from the retained operation row for that `op_id`
- diagnosis features require `chart_time < opstart_time`
- CHF and ascites also require `chart_time >= opstart_time - 30 days`
- preoperative creatinine comes from the existing retained preop feature artifact
- no postoperative diagnosis rows are used
- no postoperative labs are used
- no intraoperative summaries are used
- no downstream label information is used in factor construction

The grouped evaluation design remains unchanged:

- manifests are grouped on `patient_id`
- calibration is grouped on `op_id`
- `gs_aki_rule` reuses the same maintained split/calibration/report path as the learned models

## Missing-Data Policy

GS-AKI uses complete cases for required score inputs, but the complete-case boundary is applied at the correct stage.

Important implementation detail:

- the broader retained preop cohort can still contain operations that are later excluded from AKI comparison because `preop_creatinine` is missing
- GS-AKI therefore excludes missing required inputs in its own feature artifact and records those exclusions in `gs_aki_audit.csv`
- alignment is then enforced at label time by requiring `tabular_gs_aki_labeled.csv` to have the same number of rows as the maintained retained preop labeled cohort

This preserves the intended head-to-head comparison cohort without causing GS-AKI to fail too early in preprocessing.

Local audit on the current default AKI artifact root:

- retained preop rows before GS-AKI complete-case filtering: `122,508`
- GS-AKI complete-case rows: `120,718`
- excluded for missing required inputs before GS-AKI scoring: `1,790`

## Intraperitoneal Proxy: Why A Proxy Was Needed

The original GS-AKI uses an intraperitoneal-surgery factor, but INSPIRE does not provide that field directly.

We therefore built a deterministic, repo-tracked ICD-10-PCS proxy using:

- the official CDC/NHSN ICD-10-PCS to NHSN operative-category workbook, obtained from the NHSN SSI procedure-code documents pathway described by CDC.[4]
- the official CMS ICD-10-PCS order-file download for 2026, used for title auditing and unmatched-code review.[5]

We did not use:

- department
- case length
- operation length
- outcome-informed heuristics
- hand-built free-text procedure guesses

## Intraperitoneal Proxy: Exact Build Rule

The maintainer builder downloads or consumes the CDC and CMS source files, then derives a 5-character binary map for the retained AKI cohort.

The exact primary rule is:

```text
code5 = pcs7[:5]
approach = pcs7[4]

intraperitoneal_proxy = 1
if:
  nhsn_category in {"APPY","BILI","CHOL","COLO","GAST","HYST","OVRY","REC","SB","SPLE","XLAP","LTP"}
  and approach in {"0","4","F"}
else 0
```

The CDC FAQ was the key clinical coding reference for this step because it states:

- the 5th ICD-10-PCS character indicates approach
- `0` is open
- `4` is percutaneous endoscopic
- `F` is via natural or artificial opening with percutaneous endoscopic assistance
- the 5th character determines whether hysterectomy procedures map to NHSN `HYST` versus `VHYS`
- NHSN `HYST` includes abdominal-incision procedures, including trocar insertion.[4]

These CDC rules are the basis for both the primary mapping and the uterus-family expert overrides described below.

## Intraperitoneal Proxy: Builder Guardrails

The builder in [intraperitoneal_map_builder.py](/home/exouser/Inspire-AKI-prediction/src/inspire_aki/clinical_baselines/intraperitoneal_map_builder.py) applies several hard checks:

1. observed retained INSPIRE codes must be normalized 5-character ICD-10-PCS strings
2. observed codes cannot contain the disallowed ICD-10 letters `O` or `I`
3. 7-character CDC rows are collapsed to 5-character observed codes only
4. if one 5-character code would inherit contradictory labels across 7-character expansions, the builder fails
5. unmatched observed codes default to `0`, but only after a CMS title audit
6. if unmatched CMS titles contain obvious intraperitoneal keywords and cover more than `0.1%` of retained operations, the builder fails
7. cohort-excluded NHSN categories such as `CSEC` and `KTP` trigger failure if they appear in the retained mapping step

CMS titles are used for audit only, not as the primary positive classifier.[5]

## Intraperitoneal Proxy: Expert Overrides

After the CDC/NHSN primary map and CMS audit, 19 observed 5-character code families remained clinically ambiguous enough to require explicit forced binary review. Those overrides are now committed in code and in the final CSV.

These overrides are not presented as official GS-AKI source truth. They are explicit INSPIRE-specific adaptation decisions layered on top of the CDC/NHSN primary mapping.

### Override Summary

- override code families: `19`
- retained operations covered by override families during map build: `1,377`
- override positives in the retained-code audit: `1,194`
- override negatives in the retained-code audit: `183`

### Full Override Table

| Code | Retained ops at map-build audit | Final label | Group | Rationale |
| --- | ---: | ---: | --- | --- |
| `0UB90` | 669 | 1 | uterus/cervix | open uterine excision; treated as abdominal uterine surgery under the same abdominal-access logic used by CDC for `HYST` |
| `0UB94` | 476 | 1 | uterus/cervix | laparoscopic uterine excision; same uterus-family logic |
| `0UQ90` | 10 | 1 | uterus/cervix | open uterine repair; same uterus-family logic |
| `0UQ94` | 3 | 1 | uterus/cervix | laparoscopic uterine repair; same uterus-family logic |
| `0UJD0` | 9 | 1 | uterus/cervix | open inspection of uterus/cervix; same uterus-family logic |
| `0UJD4` | 8 | 1 | uterus/cervix | laparoscopic inspection of uterus/cervix; same uterus-family logic |
| `0US94` | 4 | 1 | uterus/cervix | laparoscopic uterine reposition; same uterus-family logic |
| `0U990` | 1 | 1 | uterus/cervix | open uterine drainage; same uterus-family logic |
| `0T160` | 47 | 0 | ureter/bladder diversion | urinary diversion/reimplant family, not counted as intraperitoneal-visceral surgery |
| `0T164` | 23 | 0 | ureter/bladder diversion | same ureter/bladder diversion logic |
| `0T170` | 22 | 0 | ureter/bladder diversion | same ureter/bladder diversion logic |
| `0T174` | 11 | 0 | ureter/bladder diversion | same ureter/bladder diversion logic |
| `0T1B0` | 3 | 0 | ureter/bladder diversion | bladder-diversion family, kept negative conservatively |
| `0T1B4` | 1 | 0 | ureter/bladder diversion | same bladder-diversion logic |
| `04100` | 52 | 0 | abdominal vascular | abdominal vascular bypass, not counted as intraperitoneal-visceral surgery |
| `04C50` | 1 | 0 | abdominal vascular | mesenteric artery vascular procedure, kept negative conservatively |
| `0FYG0` | 13 | 1 | transplant/GI | open pancreas transplantation; forced positive at 5-character resolution |
| `0DY60` | 1 | 1 | transplant/GI | open stomach transplantation; direct intraperitoneal GI organ surgery |
| `001U0` | 23 | 0 | mixed truncation artifact | 5-character truncation collapses mixed 7-character expansions; conservative forced negative |

## Local Mapping Audit

On the retained AKI cohort used for the mapping build:

- retained operations: `122,508`
- unique observed 5-character ICD-10-PCS codes: `2,194`
- positive intraperitoneal operations after final map build: `21,786`
- positive fraction: `17.78%`
- positive 5-character codes: `184`
- mapping source breakdown by observed codes:
  - `cdc_nhsn_primary`: `428`
  - `expert_review_override`: `19`
  - `default_zero_unmatched`: `1,747`

On the current GS-AKI complete-case feature artifact:

- scored operations: `120,718`
- intraperitoneal positives: `21,637`
- intraperitoneal positive fraction: `17.92%`

Top positive codes in the retained mapping audit were dominated by CDC/NHSN primary categories rather than overrides, including `0FT44`, `0DB64`, `0DTP0`, `0UT94`, and `0WJG0`.

## Exact Prediction Representation

`gs_aki_rule` is deterministic and no-fit, but the maintained prediction pipeline expects a probability-like ranking field.

To stay compatible with the shared prediction artifact schema:

- `gs_aki_count` is preserved in the dedicated GS-AKI dataset
- the runtime ranking score is:

```text
y_prob_raw = gs_aki_count / 9.0
```

- the default raw binary prediction is:

```text
y_pred = (y_prob_raw >= 0.5)
```

This scaling is an artifact-compatibility choice. It is not presented as a refit probability model for GS-AKI itself. Reviewer-facing interpretation should prioritize:

- raw count
- class
- held-out incidence by count
- held-out incidence by class

## Output Artifacts

When `gs_aki_rule` is enabled in the default AKI config, the maintained pipeline writes these relative artifacts under the configured artifact root:

- `features/preop/gs_aki_features.csv`
- `cohort/gs_aki_audit.csv`
- `datasets/tabular/tabular_gs_aki_labeled.csv`
- GS-AKI rows in the shared tabular prediction partition
- a held-out GS-AKI incidence table by raw count and class during reporting

GS-AKI is intentionally kept out of the standard ML feature matrices.

## Short Manuscript-Friendly Summary

If this note needs to be compressed into manuscript or supplement prose, the shortest faithful version is:

> We implemented one adapted GS-AKI baseline as a deterministic preoperative clinical comparator in the maintained AKI pipeline. Age, sex, emergency status, and preoperative creatinine were taken directly from the retained operation-level cohort. Diabetes, hypertension, CHF, and ascites were derived from diagnosis history restricted to records documented before the indexed operation. Because INSPIRE does not provide a native intraperitoneal-surgery variable and public documentation is inconsistent about ICD-10-PCS truncation, intraperitoneal surgery was approximated using a committed 5-character ICD-10-PCS proxy map derived primarily from CDC/NHSN operative-category mappings and CMS order-file audits, with a small number of explicit expert-reviewed overrides for residual observed code families. The final adapted score used 9 grouped factors and the published simplified GS-AKI class cutpoints.

## References

1. Kheterpal S, Tremper KK, Heung M, et al. *Development and validation of an acute kidney injury risk index for patients undergoing general surgery: results from a national data set*. [PubMed](https://pubmed.ncbi.nlm.nih.gov/19212261/)
2. Lim L, Lee H. *INSPIRE, a publicly available research dataset for perioperative medicine* (PhysioNet v1.3). [PhysioNet](https://physionet.org/content/inspire/)
3. Lim L, Lee H, et al. *INSPIRE, a publicly available research dataset for perioperative medicine*. *Scientific Data*. [PMC](https://pmc.ncbi.nlm.nih.gov/articles/PMC11192876/)
4. CDC NHSN. *FAQs: Surgical Site Procedure Codes*. [CDC](https://www.cdc.gov/nhsn/faqs/faq-ssi-proc-codes.html)
5. CMS. *ICD-10*. [CMS](https://www.cms.gov/medicare/coding-billing/icd-10-codes)
