import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from sklearn.preprocessing import StandardScaler


inspire_path = Path("/home/server/Projects/data/INSPIRE/physionet.org/files/inspire/1.3")
vitals_file = inspire_path / "vitals.csv"
preop_cleaned = "/home/server/Projects/data/AKI/preop_cleaned.csv"
output_csv = '/home/server/Projects/data/AKI/time_series_cleaned.csv'

# Load data from CSVs
print(f"Loading Data")
df_vitals = pd.read_csv(vitals_file)
df_preop = pd.read_csv(preop_cleaned)

# Cut down df_vitals to only include op_ids included in df_preop
df_vitals = df_vitals[df_vitals['op_id'].isin(df_preop['op_id'].unique())]
df_vitals = df_vitals.drop_duplicates(subset=['op_id', 'chart_time', 'item_name'], keep='first')
# Cut down df_vitals to only include item_names of high-and-medium-frequency vitals
high_frequency_labels = ["rr", "hr", "spo2", "fio2", "pmean", "etco2", "peep", 
"pip", "art_mbp", "cpat", "vt", "art_sbp", "art_dbp", 
"minvol", "pplat", "bt", "etgas", "cvp"]
medium_frequency_labels = ["pap_mbp", "pap_sbp", "pap_dbp", "nibp_mbp", "nibp_dbp", "nibp_sbp"]
regular_labels = high_frequency_labels + medium_frequency_labels
df_regular = df_vitals.loc[df_vitals['item_name'].isin(regular_labels), ['op_id', 'item_name', 'value', 'chart_time']]

col = 'value'
contained_df = []
for label in tqdm(regular_labels):
    df = df_regular.loc[df_regular['item_name'] == label]
    lower_1 = df[col].quantile(0.01)
    upper_1 = df[col].quantile(0.99)
    lower_05 = df[col].quantile(0.005)
    lower_5 = df[col].quantile(0.05)
    upper_95 = df[col].quantile(0.95)
    upper_995 = df[col].quantile(0.995)
    mask_lower = df[col] <= lower_1
    mask_upper = df[col] >= upper_1
    df.loc[mask_lower, col] = np.random.uniform(lower_05, lower_5, mask_lower.sum())
    df.loc[mask_upper, col] = np.random.uniform(upper_95, upper_995, mask_upper.sum())
    contained_df.append(df)
df_regular = pd.concat(contained_df)


interpolated_dfs = []
for op_id, df in tqdm(df_regular.groupby('op_id')):
    df = df[['item_name', 'value', 'chart_time']]
    df_complete = pd.DataFrame({'chart_time': np.arange(df['chart_time'].min(), df['chart_time'].max() + 5, 5)})
    df = df.pivot(index=['chart_time'], columns='item_name', values='value')
    
    df_complete = df_complete.merge(df, on='chart_time', how='left')
    #DOES NOT ADD WHOLLY BLANK COLUMNS

    df_complete.fillna(df_complete.mean(), inplace=True)
    df_complete['op_id'] = op_id
    interpolated_dfs.append(df_complete)

df_final = pd.concat(interpolated_dfs)
df_final.insert(0, 'op_id', df_final.pop('op_id'))

ignore = ['op_id', 'chart_time', 'aki']
cols_to_norm = [col for col in df_final.columns if col not in ignore]
scaler = StandardScaler()
df_final[cols_to_norm] = scaler.fit_transform(df_final[cols_to_norm])

df_final.to_csv(output_csv, index=False)