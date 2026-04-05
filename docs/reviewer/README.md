# Reviewer Docs

These docs collect the durable context most useful for manuscript revisions, reviewer responses, and publication-facing repo review.

They assume the maintained CLI workflow, which now uses patient-grouped evaluation manifests and grouped calibration on `op_id` rather than the archived operation-level process.

Start here:

- [manuscript_alignment.md](manuscript_alignment.md)
- [reproducibility.md](reproducibility.md)
- [missingness_sensitivity.md](missingness_sensitivity.md)
  - reviewer-specific combined `xgb` missing-data sensitivity workflow and the design decision to keep it separate from the default CLI path
- [ophthalmology_department_audit.md](ophthalmology_department_audit.md)
  - focused reviewer-response note for the `department_OS` / `department_OT` provenance fix and the ophthalmology subgroup interpretation
- [cardiothoracic_procedure_audit.md](cardiothoracic_procedure_audit.md)
  - default operation-level noncardiac adjudication logic, cardiothoracic audit companion outputs, and manuscript-grade reference links
- [gs_aki_adaptation.md](gs_aki_adaptation.md)
- [legacy_cli_differences.md](legacy_cli_differences.md)

Supporting materials:

- archived reviewer-response HTML: [archive/ResponsetoReviewed41026.html](archive/ResponsetoReviewed41026.html)
- legacy audit/reference material: [../../legacy/README.md](../../legacy/README.md)

The maintained execution surface is still the CLI pipeline under `src/inspire_aki/`; these docs exist to explain how that maintained surface relates to the archived research workflow and manuscript history.
