### VitalDB-Dimensionality-Reduction ###

ACTUALLY IT IS INSPIRE

This project contains methods and techniques for testing important covariates in the VitalDB dataset. The main focus is to use dimensionality reduction techniques to identify significant covariates that contribute to outcomes in the dataset.

**Using the Virtual Environment**

To start using the project, follow these steps to activate the pre-configured virtual environment on the Linux server.

1) Activate the Virtual Environment

From the project directory, run the following command to activate the virtual environment:

`source .venv/bin/activate`

You should see the prompt change to indicate that the `.venv` environment is active.

2) Start Coding
Once the environment is active, you can start working on the project. You can run Python scripts or access the interactive Python shell using:

To run a Python script:
`python your_script.py`

To open an interactive Python shell:
`python`

Deactivate the Environment
When you're done working, you can deactivate the virtual environment by running:
`deactivate`

**Project Description**

The project uses various Python libraries, including:

- numpy for numerical operations
- pandas for data manipulation
- scikit-learn for machine learning and dimensionality reduction
- matplotlib for data visualization
- torch for advanced computations

Ensure that the virtual environment is activated before running any scripts to have access to these dependencies.

**Data Paths**

 - "/home/server/Projects/data/AKI/tabular_combined.npz"    (preop and intraop merged and split into test and train, upsampled by repetition)   TABULAR
 - "/home/server/Projects/data/AKI/tabular_preop.npz"       (preop data split into test and train, upsampled by repetition)                     TABULAR
 - "/home/server/Projects/data/AKI/tabular_intraop.npz"     (intraop split into test and train, upsampled by repetition)                        TABULAR
 - '/home/server/Projects/data/AKI/time_series_cleaned.csv' (intraop data)                                                                      TIME SERIES

 - "/home/server/Projects/data/AKI/preop_trainable/upsampled.npz" (preop data split into test and train, upsampled by repetition)
 - "/home/server/Projects/data/AKI/preop_trainable/smoted.npz" (preop data split into test and train, upsampled by SMOTE)
 - '/home/server/Projects/data/AKI/cross_sectional_stats_longitudinal.csv' (8 cross sectional stats about longitudinal data)

 - "/home/server/Projects/data/AKI/aki_data.csv" (preop data with postop creatinine)
 - "/home/server/Projects/data/AKI/preop_data_nidhir.csv" (nidhir's experimental csv)
 - "/home/server/Projects/data/AKI/preop_trainable/unfiltered.npz" (preop data split into test and train, with aki, aki>0.3, and aki filtered to be positive(see data_preprocessing/create_aki_trainable.py))
 - "/home/server/Projects/data/AKI/aki_data_trainable.csv" (processed preop data with postop creatinine, ready for train/test split)
