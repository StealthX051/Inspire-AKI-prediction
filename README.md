# VitalDB-Dimensionality-Reduction 

The name is misleading, this is using the INSPIRE database. This projects evaluates several machine learning methods in the prediction of postoperative AKI in a non-cardiac surgery population using the INSPIRE database

# Create Results

All scripts are located in the create_results folder. 

## Setting up tmux 
1. Make a new tmux
    `tmux new -s mysession`
2. Activate the venv in the tmux
    `source .venv/bin/activate`
3. Run the script - navigate to location of script
    `python file.py`
4. How to attach to existing session
    `tmux ls`
    `tmux attach-session -t {name of session}`
5. Kill tmux
    `tmux kill-session -t {name of session}`

## Pushing
`Git add —all`
`Git commit -m “insert message”`
`Git push`

## Data Preparation

## Hyperparameter Optimization 

### tabular_hpo.py
This script will perform HPO using Optuna on all tabular models except AutoGluon (because AutoGluon explicitly recommends against HPO) and on three types of data. Results of HPO will be saved in **/home/server/Projects/data/AKI/results/tabular_hpo_results.txt** in an easily copy and paste format into **tabular_model_creation.py**

### lstm_hpo.py
Performs HPO optimization on a LSTM (to compare with intraoperative only models) and Hybrid MLP LSTM (to compare with combined models). Saves results to **/home/server/Projects/data/AKI/results/hybrid_hpo_results.txt**

## time_series testing

### justin_lstm.ipynb 
In preoperative_models, tests LSTM, LSTM variants, Transformer, and TCN for supplemental

## Final Model Creation
### tabular_model_creation.py
This script creates all the tabular models (HPO parameters provided by previous step) on all three datasets. Results are saved to **/home/server/Projects/data/AKI/results/tabular_{dataset type}_test.pkl**

### lstm.py
Trains a LSTM (intaop only) and Hybrid MLP + LSTM model (combined). Saves both results to **'/home/server/Projects/data/AKI/results/lstm_hybrid_test_optimized.pkl'**. Also saves the LSTM to **/home/server/Projects/data/AKI/results/tabular_intraop_test.pkl** as `lstm` and saves the Hybrid MLP + LSTM model to **/home/server/Projects/data/AKI/results/tabular_combined_test.pkl** as `hybrid`. Also saves `base_54k` to all three output pkls if the training/validation sets of the LSTM and Hybrid models are different than the tabular dataset.

## Results/Figure Creation

### cohort_characteristics.ipynb
This notebook creates Table 1 Cohort Characteristics and the Variable Fill Table in the first two cells. 

### performance_metrics.ipynb 
This notebook creates the AUROC, AUPRC, and Calibration curves for all models, including ASA for preop, LSTM for intraop, and Hybrid for combined. Uses two base truths, base which is for the tabular datasets which contains about 67k patients and `base_54k` which is for LSTM and Hybrid models, which contain about 54k patients. Also creates the detailed performance metrics supplemental table in teh 2nd cell. Saves all plots to **create_results/figures**. Can plot confidence intervals by setting `PLOT_CONFIDENCE_INTERVALS = True`

### delong_table.ipynb
Runs and creates the DeLong supplemental table that checks AUROC statistical difference between preop combined programs, within each group and between each group. Outputs a formatted html table that can be copy pasted into Google Docs and also outputs raw HTML code that can be rendered and then copy pasted into Google Docs 

# Code Stuff - Not Verified by Justin
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

***Training Data***
 - '/home/server/Projects/data/base/tabular_combined.csv'   (generic X data with no y variables,
 - '/home/server/Projects/data/base/tabular_preop.csv'      formatted as a dataframe with ~120k 
 - '/home/server/Projects/data/base/tabular_intraop.csv'    rows and ~500 columns. )



 - '/home/server/Projects/data/AKI/time_series_cleaned.csv' (intraop data)                                                                      TIME SERIES


 - "/home/server/Projects/data/AKI/aki_data.csv" (preop data with postop creatinine)
 - "/home/server/Projects/data/AKI/preop_data_nidhir.csv" (nidhir's experimental csv)
 - "/home/server/Projects/data/AKI/preop_trainable/unfiltered.npz" (preop data split into test and train, with aki, aki>0.3, and aki filtered to be positive(see data_preprocessing/create_aki_trainable.py))

 ***Results***
 - "/home/server/Projects/data/AKI/results/tabular_hpo_results.txt" (HPO results for Tabular Models - feed into tabular_model_creation.py)
 - "/home/server/Projects/data/AKI/results/hybrid_hpo_results.txt" (HPO results for LSTM & Hybrid Model - feed into LSTM_model_creation.py)
 - /home/server/Projects/data/AKI/results/tabular_{dataset type}_test.pkl (Split into preop, intraop, and combined. Outputs for bootstrapped models. Preop contains ASA, Intraop contains LSTM, Combined contains hybrid)
