# Cardiothoracic Procedure Audit

This note documents the maintained operation-level cardiothoracic procedure adjudication used by the current `inspire-aki` pipeline to define the default noncardiac cohort for manuscript-facing runs.

The explicit `report procedure-audit` stage remains available because it is the transparency companion to the default cohort rule. It is no longer documented as a separate reviewer-only filtering method or as an alternate sensitivity-analysis cohort definition.

## Why This Exists

The underlying reviewer concern is methodologically sound: a cohort described as noncardiac surgery should not rely on a hospital `department` label as a proxy for operative phenotype.

In INSPIRE:

- `department` is an administrative service field
- `icd10_pcs` is the procedure field
- one admission can contribute more than one operation

The maintained pipeline therefore treats noncardiac status as an operation-level classification problem on `op_id`, not as a department-level shortcut and not as a patient-level summary.

The practical goals of the maintained logic are:

- exclude definite cardiac procedures from the final analytic cohort
- retain clearly noncardiac thoracic procedures that legitimately fall under cardiothoracic services
- conservatively exclude the smaller residual set of CPB-supported or unresolved cardiac-adjacent outliers
- fail loudly if any operation cannot be assigned both an audit class and a final retain/exclude decision

## Current Position

The current maintained position in this repository is:

- the default paper cohort is the strict operation-level adjudicated noncardiac cohort
- the older less-strict prefix-only cohort remains available only as an explicit legacy/debug profile
- the explicit `report procedure-audit` stage audits the same operation-level logic used by the default cohort
- `department` is used only to define the reviewer-facing cardiothoracic subset in audit summaries; it is never used alone to decide cardiac versus noncardiac status

Because this default-cohort promotion was implemented without an immediate end-to-end rerun, any existing mounted or checked-in `artifacts/default` outputs should be treated as stale until they are regenerated under the promoted default.

## External Clinical And Coding Basis

The maintained logic is grounded in official ICD-10-PCS coding references plus a small number of domain-specific surgical references that explain why thoracic, cardiac, aortic, and CPB-supported cases should not be treated as one undifferentiated bucket.

### ICD-10-PCS structure

- CMS defines ICD-10-PCS as a 7-character procedure code system in which, for Medical and Surgical codes, character 2 is the body system and character 3 is the root operation.
- The CMS development manual explicitly uses `027` to illustrate `0` Medical and Surgical, `2` Heart and Great Vessels, and `7` Dilation.
- CMS also identifies `B` as the Respiratory System body-system value.

Those facts are the coding anchors for the maintained `02...` cardiac logic and `0B...` respiratory/thoracic logic.

References:

- [CMS ICD-10-PCS Development Manual](https://edit.cms.gov/Medicare/Coding/ICD10/Downloads/2014-pcs-procedure-coding-system.pdf)
- [CMS ICD-10-PCS Reference Manual](https://www.cms.gov/files/document/icd-10-pcs-reference-manual-2016.pdf)

Because the public INSPIRE release stores truncated procedure prefixes rather than full 7-character codes, this repository does not pretend to do full-code adjudication from the public field alone. Instead, it uses the mounted 5-character prefix contract, CMS order-file decoding, and conservative rule-based resolution.

### Thoracic versus cardiac clinical framing

- STS treats Adult Cardiac Surgery and General Thoracic Surgery as separate clinical domains rather than one undifferentiated cardiothoracic category.
- STS General Thoracic Surgery Database specifications include esophageal and thoracic procedure families, supporting the interpretation that esophagus and other thoracic operations can legitimately live under cardiothoracic services while still being noncardiac.
- STS treats ascending-aorta and aortic-root surgery as part of the adult cardiac domain, which is why the maintained adjudication is intentionally conservative around CPB-supported aortic and vascular families.

References:

- [STS General Thoracic Surgery Database Data Specifications](https://www.sts.org/sites/default/files/documents/STSThoracicDataSpecsV2_41.pdf)
- [STS risk calculator launch for ascending aorta and aortic root surgery](https://www.sts.org/press-releases/sts-launches-new-risk-calculator-ascending-aorta-and-aortic-root-surgery)

### CPB interpretation

- CPB is not exclusive to intracardiac surgery and can appear in selected noncardiac thoracic contexts.
- At the same time, CPB is a strong physiologic signal that a case is not representative of an ordinary noncardiac operative phenotype, especially when paired with aortic or vascular families or unresolved titles.

That is why the maintained logic treats CPB as a secondary but high-concern sanity signal rather than as the primary classifier.

Reference:

- [Cardiopulmonary Bypass in Non-Cardiac Surgery](https://pubmed.ncbi.nlm.nih.gov/29753653/)

### Broader noncardiac-surgery literature

- Published noncardiac-surgery prediction studies are not perfectly uniform in what they count as noncardiac.
- Some broader perioperative models include intrathoracic and major vascular categories inside noncardiac surgery risk groupings.
- The maintained default cohort in this repository deliberately adopts the more conservative interpretation for reviewer-facing AKI work: retain clearly noncardiac thoracic cases, but exclude the residual CPB-supported ambiguous buckets.

Reference:

- [Developing a Machine Learning Model for Predicting 30-Day Major Adverse Cardiac and Cerebrovascular Events in Patients Undergoing Noncardiac Surgery](https://pmc.ncbi.nlm.nih.gov/articles/PMC12018863/)

### INSPIRE-specific data contract

- Public INSPIRE documentation is useful for provenance, but the maintained code is keyed to the currently mounted raw-data contract rather than to documentation ambiguity alone.
- The code verifies the observed `icd10_pcs` contract directly from mounted `operations.csv` and raises if the expected 5-character contract is not observed.

Reference:

- [INSPIRE on PhysioNet](https://physionet.org/content/inspire/1.3/)

## Current Maintained CLI Surface

Run the explicit audit report:

```bash
inspire-aki report procedure-audit --config configs/aki/default.yaml
```

This audits the same operation-level noncardiac rule used by the canonical default cohort, but it is still kept as an explicit stage rather than being folded into `report manuscript` or `run all`.

Run the default pipeline:

```bash
inspire-aki run all --config configs/aki/default.yaml
```

That default config now uses `strict_noncardiac_adult_procedure_audit` as the canonical cohort profile.

If the same strict cohort needs to be rerun into a separate artifact root for isolated review work, the compatibility alias remains available:

```bash
inspire-aki run all --config configs/aki/strict_noncardiac_review.yaml
```

That alias does not define a different cohort. It only points the same strict cohort logic at a different artifact root.

## Current Mounted Data Contract

The maintained adjudication assumes and verifies the current mounted INSPIRE contract used by this repository:

- source table: raw `operations.csv`
- analytic denominator: final operation-level `op_id` rows in `cohort/labels.csv`
- procedure field: 5-character `icd10_pcs`
- CPB support fields: `cpbon_time` and `cpboff_time`

The code raises if the mounted `icd10_pcs` field is not uniformly 5 characters after normalization. That check exists because public INSPIRE materials have historically described truncated ICD-10-PCS storage in slightly inconsistent ways.

## Processing Logic In The Maintained Code

The maintained implementation lives primarily in:

- [src/inspire_aki/reporting/procedure_audit.py](../../src/inspire_aki/reporting/procedure_audit.py)
- [src/inspire_aki/cohort/preop.py](../../src/inspire_aki/cohort/preop.py)
- [src/inspire_aki/config.py](../../src/inspire_aki/config.py)
- [configs/aki/default.yaml](../../configs/aki/default.yaml)
- [configs/aki/strict_noncardiac_review.yaml](../../configs/aki/strict_noncardiac_review.yaml)

The processing sequence is:

1. Read the final analytic `op_id` set from `cohort/labels.csv`.
2. Join those operations back to raw `operations.csv`.
3. Normalize `op_id`, `department`, `icd10_pcs`, `cpbon_time`, and `cpboff_time`.
4. Verify that the observed `icd10_pcs` contract is 5 characters.
5. Parse the configured CMS ICD-10-PCS order zip and collapse 7-character CMS titles down to a usable 5-character prefix reference.
6. Assign an `audit_class` to every operation.
7. For residual prefix-level gray-zone cases, assign a `clinician_review_bucket`.
8. Look up same-4-character observed neighbors for unresolved prefixes.
9. Assign a final `final_noncardiac_action` of either `retain` or `exclude`.

The code explicitly validates coverage at the end of annotation. If any operation lacks an `audit_class`, or if any operation lacks a `final_noncardiac_action`, the audit raises an error rather than silently leaving the cohort partially categorized.

## Audit Taxonomy

The maintained audit classes are:

| Audit class | Meaning | Manuscript interpretation | Default-cohort action |
| --- | --- | --- | --- |
| `cardiac_exclude` | Official `02...` Heart and Great Vessels family or an explicit cardiac title family | definite cardiac contamination | exclude |
| `thoracic_keep` | `0B...` respiratory family without CPB discordance | definite thoracic noncardiac keep | retain |
| `thoracic_or_chest_related_noncardiac` | thoracic, mediastinal, foregut, or chest-related noncardiac family outside the respiratory prefix family | retain and describe explicitly | retain |
| `ct_service_noncardiac_keep` | clearly noncardiac nonthoracic CTS-labeled family | administrative CTS label, not cardiac phenotype | retain |
| `vascular_noncardiac_describe` | non-`02` vascular family without CPB | noncardiac but explicitly described | retain |
| `manual_review` | residual prefix-level ambiguity or CPB discordance | resolved by explicit final-action bucket rule | mixed by bucket |
| `other_operation` | outside the focused CTS audit subset | outside the cardiothoracic review question | retain |

## Residual Clinician-Review Buckets

The maintained clinician-review buckets are not an invitation for ad hoc case-by-case rewriting. They are the structured gray-zone families that feed the final default cohort rule.

| Clinician-review bucket | Typical meaning | Default-cohort action |
| --- | --- | --- |
| `cpb_positive_aortic_or_vascular` | CPB-supported aortic or vascular family | exclude |
| `respiratory_plus_cpb` | respiratory family with CPB timing populated | exclude |
| `other_cpb_discordant_nonvascular_nonrespiratory` | nonvascular nonrespiratory family with CPB timing populated | exclude |
| `unresolved_prefix_or_title` | unresolved or malformed prefix/title family | retain only when CPB-negative and supported by a benign same-4 neighbor; otherwise exclude |
| `other_prefix_level_review` | residual future catch-all | exclude under the current strict rule |

## Default Cohort Resolution

The default cohort profile now uses the same operation-level adjudication logic during preprocessing.

The default cohort excludes:

- `cardiac_exclude`
- `cpb_positive_aortic_or_vascular`
- `respiratory_plus_cpb`
- `other_cpb_discordant_nonvascular_nonrespiratory`
- unresolved prefixes with CPB support
- unresolved CPB-negative prefixes that do not have convincing benign same-4-character support

The default cohort retains:

- `thoracic_keep`
- `thoracic_or_chest_related_noncardiac`
- `ct_service_noncardiac_keep`
- `vascular_noncardiac_describe`
- unresolved CPB-negative prefixes only when the same-4 neighbor supports a benign noncardiac family

Under the current configuration, benign same-4 support means that the observed neighbor label maps to a clearly noncardiac family such as:

- scalp skin
- sternum
- chest wall
- external ear
- conjunctiva
- breast

This is intentionally a one-way rescue rule for low-risk unresolved prefixes. It does not rescue CPB-supported unresolved cases, and it does not rescue unresolved cases whose nearest observed neighbor is still clinically ambiguous.

## Why The Same-4 Neighbor Rule Exists

INSPIRE stores a truncated procedure prefix rather than the full 7-character ICD-10-PCS code. That means some prefixes cannot be resolved cleanly from the CMS order-file collapse alone.

The same-4-character neighbor rule is therefore a conservative fallback, not a replacement for full-code decoding:

- it is used only for unresolved prefixes
- it is used only when CPB is absent
- it is used only when the observed neighbor label is clearly benign and noncardiac
- it results in retention only for low-risk families such as scalp-skin or sternum/chest-wall variants

Everything else is excluded in the default strict cohort.

## Reviewer Outputs

The explicit audit report writes tables under `reports/tables/`:

- `procedure_audit_qc_summary`
- `procedure_audit_global_summary`
- `procedure_audit_ct_department_summary`
- `procedure_audit_ct_top_prefixes`
- `procedure_audit_ct_manuscript_summary`
- `procedure_audit_clinician_review_summary`
- `procedure_audit_flagged_cardiac_cases`
- `procedure_audit_manual_review`

The most important outputs are:

- `procedure_audit_flagged_cardiac_cases.csv`
  - machine-readable manifest of definite cardiac cases still present in the audited cohort
- `procedure_audit_ct_manuscript_summary.csv`
  - compact cardiothoracic summary table for manuscript and rebuttal drafting
- `procedure_audit_clinician_review_summary.csv`
  - bucket-level summary of the residual clinician-review groups
  - includes retained-versus-excluded counts and the final recommended action
- `procedure_audit_manual_review.csv`
  - row-level ledger for the residual gray-zone families, with final retain/exclude actions already recorded
  - includes `audit_reason_code`, `clinician_review_bucket`, `same4_neighbor_prefix`, `same4_neighbor_label`, `final_noncardiac_action`, and `final_noncardiac_note`

## Manuscript And Supplement Framing

The maintained reviewer-facing position is:

- department is not surgery type
- the relevant denominator is operation-level `op_id`, not patient-level rows
- thoracic does not equal cardiac
- the coding anchor for definite cardiac procedures is the official ICD-10-PCS Heart and Great Vessels family
- CPB is a secondary but high-concern signal
- the real threat to the noncardiac label is the smaller set of CPB-supported ambiguous families, not the mere presence of CTS-labeled cases

That is why the maintained docs and tables describe the cohort-characteristics row as `Department, n (%)` rather than any wording that implies department is equivalent to operative phenotype.

## Manuscript-Ready Wording

These text blocks are intentionally count-free until the default artifact tree is rerun under the promoted cohort rule.

### Methods sentence

Noncardiac status was adjudicated at the operation level using ICD-10-PCS procedure families rather than surgical department labels; definite cardiac procedures were excluded, clearly noncardiac thoracic procedures were retained, and residual CPB-supported or unresolved ambiguous cases were conservatively excluded.

### Supplement paragraph

Because hospital service labels do not map cleanly to operative phenotype, we adjudicated the noncardiac cohort at the operation level using ICD-10-PCS procedure families and CPB timing fields rather than department alone. Cardiothoracic service cases were separated into definite cardiac procedures, clearly noncardiac thoracic or other noncardiac families, and a smaller residual set of CPB-supported or unresolved ambiguous families. Definite cardiac procedures and residual CPB-supported or unresolved ambiguous cases were excluded from the default cohort, whereas clearly noncardiac thoracic and other clearly noncardiac families were retained.

### Table footnote sentence

Department reflects the administrative surgical service and was not used alone to define cardiac versus noncardiac operative phenotype.

### Rebuttal paragraph template

We agree that a noncardiac cohort should not be defined by department label alone. The apparent cardiothoracic proportion in the original table reflected an administrative service label rather than operative phenotype. The maintained default cohort now uses operation-level ICD-10-PCS adjudication, with definite cardiac procedures excluded, clearly noncardiac thoracic procedures retained, and residual CPB-supported or unresolved ambiguous cases conservatively excluded. The explicit cardiothoracic audit tables now provide a compact count summary and detailed supporting outputs for this adjudication step.

## Current Status Of Stored Artifacts

This documentation now reflects the promoted default cohort definition in code and configs.

However:

- current mounted or previously generated `artifacts/default` outputs may still reflect the older less-strict default cohort
- current count-bearing manuscript text should not be updated from those stale outputs
- a later full rerun is still required before citing updated cohort counts, event counts, performance tables, or cardiothoracic audit totals from the promoted default

Until that rerun happens, this document should be read as the authoritative description of the default filtering rule, not as proof that every mounted artifact tree has already been regenerated under that rule.

## Limitations

The maintained adjudication is deliberately conservative, but it still has real limits:

- it works from truncated 5-character prefixes, not full ICD-10-PCS codes
- it depends on the currently mounted raw-data contract
- it uses CPB timing fields as a sanity signal, not a full operative narrative
- it uses a benign-neighbor rescue rule only for low-risk unresolved prefixes

Those limits are why the explicit audit report remains useful even though the strict adjudication is now the default cohort rule.

## References

1. [INSPIRE on PhysioNet](https://physionet.org/content/inspire/1.3/)
2. [CMS ICD-10-PCS Development Manual](https://edit.cms.gov/Medicare/Coding/ICD10/Downloads/2014-pcs-procedure-coding-system.pdf)
3. [CMS ICD-10-PCS Reference Manual](https://www.cms.gov/files/document/icd-10-pcs-reference-manual-2016.pdf)
4. [STS General Thoracic Surgery Database Data Specifications](https://www.sts.org/sites/default/files/documents/STSThoracicDataSpecsV2_41.pdf)
5. [STS risk calculator launch for ascending aorta and aortic root surgery](https://www.sts.org/press-releases/sts-launches-new-risk-calculator-ascending-aorta-and-aortic-root-surgery)
6. [Cardiopulmonary Bypass in Non-Cardiac Surgery](https://pubmed.ncbi.nlm.nih.gov/29753653/)
7. [Developing a Machine Learning Model for Predicting 30-Day Major Adverse Cardiac and Cerebrovascular Events in Patients Undergoing Noncardiac Surgery](https://pmc.ncbi.nlm.nih.gov/articles/PMC12018863/)
