# README

## Article

This Dryad record provides reproducibility materials for the accepted JAMIA Open article:

**Integration of Intraoperative Data in Interpretable Machine Learning Models to Predict Postoperative AKI in Noncardiac Surgery Patients**

## Source data

The source data analyzed in the study are INSPIRE version 1.3, available from PhysioNet:

Lim L, Lee HC. INSPIRE, a publicly available research dataset for perioperative medicine. PhysioNet. Version 1.3. doi:10.13026/46m4-f655.

Access to INSPIRE requires PhysioNet credentialing and agreement to the applicable Data Use Agreement. INSPIRE source files and row-level derived datasets are not redistributed in this Dryad record.

Although PhysioNet may host later INSPIRE versions, this study used INSPIRE version 1.3.

## Files in this Dryad record

- `data_sources_manifest.csv`: Identifies the third-party source dataset used in the study, its version, DOI, access route, and the reason source data are not redistributed.

## Code and software

Analysis code is available at:

- GitHub repository: https://github.com/VivaSuresh/Inspire-AKI-prediction
- Archived software release: [insert GitHub release URL, Zenodo DOI, or Dryad-linked software DOI after release]

The code archive contains the analysis scripts, package files, configuration files, and environment files needed to reproduce the analyses after obtaining authorized access to INSPIRE.

## Reuse and reproduction

To reproduce the analyses:

1. Obtain authorized access to INSPIRE version 1.3 through PhysioNet.
2. Download the INSPIRE source files according to PhysioNet instructions.
3. Clone or download the archived code release.
4. Create the software environment using `environment.yml` or `requirements.txt`.
5. Update the data path in the configuration file as needed.
6. Run the analysis pipeline described in the repository README.

No INSPIRE source files, row-level derived datasets, patient-level predictions, patient-level train/test splits, patient-level SHAP values, generated reports, or trained model artifacts are included in this Dryad record.
