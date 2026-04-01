from pathlib import Path
import pandas as pd
import numpy as np

inspire_path = Path("/home/server/Projects/data/INSPIRE/physionet.org/files/inspire/1.3")
ops_path = inspire_path / "operations.csv"
labs_path = inspire_path / "labs.csv"
vitals_path = inspire_path / "vitals.csv"

# Set variables
OUTCOME_VAR = 'inhosp_death_30day'
INPUT_VARS = ['age', 'sex', 'emop', 'bmi', 'andur', 
              'preop_hb', 'preop_platelet', 'preop_wbc', 'preop_aptt', 'preop_ptinr', 'preop_glucose',
              'preop_bun', 'preop_albumin', 'preop_ast', 'preop_alt', 'preop_creatinine', 'preop_sodium', 'preop_potassium']

# Load operations
df = pd.read_csv('operations.csv')

# find the first operation for each patient
df.sort_values('orin_time', inplace=True)
df = df.loc[df[['op_id','subject_id']].groupby('subject_id')['op_id'].idxmin()]

df[OUTCOME_VAR] = (df['inhosp_death_time'] < df['orout_time'] + 30 * 24 * 60)
df = df[(df['asa'] < 6)]
df.loc[:, 'andur'] = df['anend_time'] - df['anstart_time']

valid_mask = df['height'] > 10
df['bmi'] = np.nan
df.loc[valid_mask, 'bmi'] = df.loc[valid_mask, 'weight'] / (df.loc[valid_mask, 'height'] / 100) ** 2

# # Load labs
# df_lab = pd.read_csv('labs.csv')
# for item_name in ('hb', 'platelet', 'aptt', 'wbc', 'ptinr', 'glucose', 'bun', 'albumin', 'ast', 'alt', 'creatinine', 'sodium', 'potassium'):
#     df = pd.merge_asof(df.sort_values('orin_time'), 
#                     df_lab.loc[df_lab['item_name'] == item_name].sort_values('chart_time'), 
#                     left_on='orin_time', right_on='chart_time', by='subject_id', 
#                     tolerance=6* 30 * 24 * 60, suffixes=('', '_'))
#     df.drop(columns=['chart_time', 'item_name'], inplace=True)
#     df.rename(columns={'value':f'preop_{item_name}'}, inplace=True)

# df['sex'] = df['sex'] == 'M'

