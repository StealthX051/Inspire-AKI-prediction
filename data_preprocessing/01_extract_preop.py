# Used to require preopdata_file FROM AKI_DATA_CLEANER.IPYNB, that code
# is now in this file
# EXTRACTS PREOP DATA TO BE COMBINED WITH INTRAOP DATA IN CREATE_BASE.PY


from pathlib import Path
import pandas as pd
import numpy as np
from collections import Counter
from tqdm import tqdm

def nprint(string):
    print("="*25, string, "="*25)

# All the input paths
inspire_path = Path("/home/server/Projects/data/INSPIRE/physionet.org/files/inspire/1.3")
labs_path = inspire_path / "labs.csv"
vitals_path = inspire_path / "vitals.csv"
ops_path = inspire_path / "operations.csv"
diagnosis_path = inspire_path / "diagnosis.csv"
ward_vitals_path = inspire_path / "ward_vitals.csv"


# move from /aki to /base later
base_path = Path("/home/server/Projects/data/base/")

# if you change output_csv, also change in aki data selection and create base
output_csv = "/home/server/Projects/data/AKI/preop_data_test.csv"

nprint("starting")
df_labs = pd.read_csv(labs_path)
df_labs["chart_time"] = df_labs["chart_time"].astype(float)
df_ward = pd.read_csv(ward_vitals_path)
df_ward["chart_time"] = df_ward["chart_time"].astype(float)
df_vitals = pd.read_csv(vitals_path)
df_ops = pd.read_csv(ops_path)
df_diags = pd.read_csv(diagnosis_path.as_posix())

nprint("finished reading csvs")



### code from AKI_data_cleaner.ipynb
df_preop = df_ops.copy()
# Keeping only the necessary columns
desired_columns = [
    'op_id', 'subject_id', 'age', 'sex', 'height', 'weight', 
    'asa', 'emop', 'opstart_time', 'opend_time', 
    'inhosp_death_time', 'allcause_death_time', 'orin_time', 'orout_time',
]
# Select only the desired columns
df_preop = df_preop[desired_columns]
# Add BSA (Body Surface Area) and BMI (Body Mass Index) columns
# Ensure height and weight are valid (not NaN) for calculations
valid_mask = df_preop['height'].notna() & df_preop['weight'].notna()
# Initialize BSA and BMI with NaN
df_preop['BSA'] = np.nan
df_preop['BMI'] = np.nan
# Calculate BSA and BMI only for valid rows
df_preop.loc[valid_mask, 'BSA'] = np.sqrt((df_preop.loc[valid_mask, 'height'] * df_preop.loc[valid_mask, 'weight']) / 3600)
df_preop.loc[valid_mask, 'BMI'] = df_preop.loc[valid_mask, 'weight'] / ((df_preop.loc[valid_mask, 'height'] / 100) ** 2)
# Add Booking Case Length column
valid_mask = df_preop['orin_time'].notna() & df_preop['orout_time'].notna()
df_preop['booking_case_length'] = np.nan
df_preop.loc[valid_mask, 'booking_case_length'] = df_preop.loc[valid_mask, 'orout_time'] - df_preop.loc[valid_mask, 'orin_time']
# Remove orin_time and orout_time columns
df_preop = df_preop.drop(columns=['orin_time', 'orout_time'])
# Filter cardiovascular diagnoses (ICD-10-CM codes starting with 'I')
df_diags_cvd = df_diags[df_diags['icd10_cm'].str.startswith('I', na=False)]
# Merge operations and cardiovascular diagnoses on subject_id
merged = pd.merge(
    df_preop[['op_id', 'subject_id', 'opstart_time']],
    df_diags_cvd[['subject_id', 'chart_time']],
    on='subject_id',
    how='inner'
)
# Filter diagnoses where chart_time < opstart_time
merged = merged[merged['chart_time'] < merged['opstart_time']]
# Count the number of diagnoses for each operation
num_card_events = merged.groupby('op_id').size().reset_index(name='num_card_events')
# Merge the counts back into the operations DataFrame
df_preop = pd.merge(
    df_preop,
    num_card_events,
    on='op_id',
    how='left'
)
# Fill NaN values with 0 for operations with no past cardiovascular diagnoses
df_preop['num_card_events'] = df_preop['num_card_events'].fillna(0).astype(int)
### end of code from AKI_data_cleaner.ipynb



df_preop = df_preop[df_preop["asa"] < 6]
df_preop = df_preop[df_preop["age"] >= 18]
df_preop = df_preop.dropna(subset="opend_time")
df_preop = df_preop.dropna(subset="opstart_time")
df_preop["op_len"] = df_preop["opend_time"] - df_preop["opstart_time"]

# encode gender and remove rows with missing height/weight
df_preop["sex"] = df_preop["sex"] == "M"
df_preop = df_preop[~(df_preop['weight'].isna() | df_preop['height'].isna())]
df_preop = df_preop[(df_preop['weight'] != 0) & (df_preop['height'] != 0)] #& (df['op_id'] != 435191458)] # pride and

# Replace antypes with numbers, after removing rows with regional set as antype
df_ops = df_ops.drop(df_ops[df_ops['antype'] == 'Regional'].index)
df_ops.loc[df_ops['antype'] == 'General', 'antype'] = 0     
df_ops.loc[df_ops['antype'] == 'MAC', 'antype'] = 1
df_ops.loc[df_ops['antype'] == 'Neuraxial', 'antype'] = 1

# Replace departments with one-hot encodings
df_ops = df_ops[df_ops['department'] != 'PED']
df_ops = pd.get_dummies(df_ops, columns=['department'])
cols_to_keep = ['op_id', 'subject_id', 'antype']
for col in df_ops.columns:
    if 'department_' in col:
        cols_to_keep.append(col)
df_preop = pd.merge(df_preop, df_ops[cols_to_keep], on=['op_id', 'subject_id'], how='inner')

nprint("finished basic filtering")

preop_item_names = [
    "total_protein",
    "sodium",
    "potassium",
    "platelet",
    "glucose",
    "wbc",
    "alt",
    "chloride",
    "lymphocyte",
    "phosphorus",
    "albumin",
    "fibrinogen",
    "creatinine",
    "ptinr",
    "total_bilirubin",
    "alp",
    "aptt",
    "calcium",
    "bun",
    "ast",
    "crp",
    "hb",
    "hct",
    "seg"
]

for item_name in preop_item_names:
    df_preop = pd.merge_asof(df_preop.sort_values('opstart_time'), 
                    df_labs.loc[df_labs['item_name'] == item_name].sort_values('chart_time'), # grab rows w the item name we want and sort by chart_time
                    left_on='opstart_time', right_on='chart_time', by='subject_id',           # chooses row in df_labs w greatest chart_time that is still less than opstart_time and matches subject_id
                    tolerance=90 * 24 * 60, suffixes=('', '_'))                               # 90 day tolerance
    df_preop.drop(columns=['chart_time', 'item_name'], inplace=True)
    df_preop.rename(columns={'value':f'preop_{item_name}'}, inplace=True)
nprint("finished getting preop data from time series")

ward_item_names = [
    "spo2",
    "bt",
    "rr",
    "nibp_dbp",
    "nibp_sbp",
    "hr"
]

for item_name in ward_item_names:
    df_preop = pd.merge_asof(df_preop.sort_values('opstart_time'), 
                    df_ward.loc[df_ward['item_name'] == item_name].sort_values('chart_time'),
                    left_on='opstart_time', right_on='chart_time', by='subject_id',
                    tolerance=90 * 24 * 60, suffixes=('', '_'))
    df_preop.drop(columns=['chart_time', 'item_name'], inplace=True)
    df_preop.rename(columns={'value':f'ward_{item_name}'}, inplace=True)
nprint("finished getting ward data from time series")


prefixes_to_exclude = ["10", "0TY", "B50", "B51"]
mask = df_ops["icd10_pcs"].astype(str).str.startswith(tuple(prefixes_to_exclude))
ops_to_exclude = df_ops.loc[mask, "op_id"]
df_preop = df_preop[~df_preop["op_id"].isin(ops_to_exclude)]
nprint("finished filtering out some procedure prefixes")

df_preop.to_csv(output_csv, index=False)
nprint(f"wrote to {output_csv}")
