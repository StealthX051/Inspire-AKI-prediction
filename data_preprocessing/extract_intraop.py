import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import entropy, kurtosis, skew
from tqdm import tqdm

inspire_path = Path("/home/server/Projects/data/INSPIRE/physionet.org/files/inspire/1.3")
vitals_file = inspire_path / "vitals.csv"
preop_cleaned = "/home/server/Projects/data/AKI/preop_cleaned.csv"
output_csv = '/home/server/Projects/data/AKI/feature_engineered.csv'

# INTENDED TO HEREAFTER BE COMBINED WITH PREOP DATA AND CLEANED WITH CREATE_AKI_TRAINABLE.PY

# Define statistical trend and energy
def trend(y):
    x = np.arange(len(y)).T
    x = np.vstack((np.ones(len(x)), x)).T
    y = y.T
    return (np.linalg.pinv(x) @ y)[1]
def energy(x):
    return np.inner(x, x)

# Load data from CSVs
print(f"Loading Data")
df_vitals = pd.read_csv(vitals_file)
df_preop = pd.read_csv(preop_cleaned)

# Cut down df_vitals to only include op_ids included in df_preop
df_vitals = df_vitals[df_vitals['op_id'].isin(df_preop['op_id'].unique())]


# REGULAR data summarized with eight statistical metrics
print("Summarizing regular longitudinal data")
# Cut down df_vitals to only include item_names of high-frequency vitals
high_frequency_labels = ["rr", "hr", "spo2", "fio2", "pmean", "etco2", "peep", 
"pip", "art_mbp", "cpat", "vt", "art_sbp", "art_dbp", 
"minvol", "pplat", "bt", "etgas", "cvp"]
medium_frequency_labels = ["pap_mbp", "pap_sbp", "pap_dbp", "nibp_mbp", "nibp_dbp", "nibp_sbp"]
regular_labels = high_frequency_labels + medium_frequency_labels
df_regular = df_vitals.loc[df_vitals['item_name'].isin(regular_labels), ['op_id', 'item_name', 'value']]
# Generate a table with 24 vitals X 8 statistical metrics = 192 columns of data
print(f"Calculating Pivot Table")
df_regular = df_regular.pivot_table(
    index='op_id', 
    columns='item_name', 
    values='value', 
    aggfunc=['mean', 'max', 'min', entropy, kurtosis, skew, trend, energy]
).reset_index()
df_regular.columns = [f"{feature}_{vital}" for feature, vital in df_regular.columns]
df_regular.columns.values[0] = 'op_id'

# Generically summed data
print("Aggregating summed variables")
cross_sec_avg_labels = ["bis", "ci", "rfti", "dobui", "mlni", "ppfi", "o2", "air", "cbro2", "ntgi"]
df_cs_average = df_vitals.loc[df_vitals['item_name'].isin(cross_sec_avg_labels), ['op_id', 'item_name', 'value']]
df_cs_average = df_cs_average.pivot_table(
    index='op_id', 
    columns='item_name', 
    values='value', 
    aggfunc=['mean']
).reset_index()
df_cs_average.columns = [f"{feature}_{vital}" for feature, vital in df_cs_average.columns]
df_cs_average.columns.values[0] = 'op_id'


# TIME and WEIGHT adjusted drugs aggregated by SUM per operation
print("Aggregating time/weight adjusted variables")
wt_adjusted_labels = ["eph", "mdz", "ppf", "sft"]
df_wt_adjusted = df_vitals.loc[df_vitals['item_name'].isin(wt_adjusted_labels), ['op_id', 'item_name', 'value']]
df_wt_adjusted = df_wt_adjusted.merge(df_preop[['op_id', 'weight', 'op_len']], on='op_id', how='inner')
df_wt_adjusted['value'] = df_wt_adjusted['value'] / (df_wt_adjusted['weight'] * df_wt_adjusted['op_len'])
df_wt_adjusted = df_wt_adjusted[['op_id', 'item_name', 'value']].pivot_table(
    index='op_id', 
    columns='item_name', 
    values='value', 
    aggfunc=['sum']
).reset_index()
# df_wt_adjusted.fillna(0, inplace=True)
df_wt_adjusted.columns = [f"{feature}_{vital}" for feature, vital in df_wt_adjusted.columns]
df_wt_adjusted.columns.values[0] = 'op_id'

# TIME adjusted measures aggregated by SUM per operation
print("Aggregating time adjusted variables")
time_adjusted_labels = ["n2o", "ebl", "rbc", "uo", "ftn", "ffp", "pc", "cryo", "pheresis"]
df_time_adjusted = df_vitals.loc[df_vitals['item_name'].isin(time_adjusted_labels), ['op_id', 'item_name', 'value']]
df_time_adjusted = df_time_adjusted.merge(df_preop[['op_id', 'op_len']], on='op_id', how='inner')
df_time_adjusted['value'] = df_time_adjusted['value'] / df_time_adjusted['op_len']
df_time_adjusted = df_time_adjusted[['op_id', 'item_name', 'value']].pivot_table(
    index='op_id', 
    columns='item_name', 
    values='value', 
    aggfunc=['sum']
).reset_index()
# df_time_adjusted.fillna(0, inplace=True)
df_time_adjusted.columns = [f"{feature}_{vital}" for feature, vital in df_time_adjusted.columns]
df_time_adjusted.columns.values[0] = 'op_id'
# "n2o" L/min ??

# TIME adjusted total fluid input (summed across 10 different types)
print("Aggregating total fluid input")
fluids_agg_labels = ["d5w", "hes", "psa", "hs", "ns", "hns", "alb20", "alb5", "d10w", "d50w"]
df_fluids_agg = df_vitals.loc[df_vitals['item_name'].isin(fluids_agg_labels), ['op_id', 'item_name', 'value']]
df_fluids_agg = df_fluids_agg.groupby("op_id")['value'].sum().reset_index()
df_fluids_agg = df_fluids_agg.merge(df_preop[['op_id', 'op_len']], on='op_id', how='inner')
df_fluids_agg['fluids_agg'] = df_fluids_agg.pop('value') / df_fluids_agg.pop('op_len')


# # Desflurane and Sevoflurane interpolated by forward fill, MAC equivalent found, and then summed across operation.
print("Aggregating MAC equivalents for anesthetics")
anesthetic_labels = ['etdes', 'etsevo']
df_anesthetic = df_vitals.loc[df_vitals['item_name'].isin(anesthetic_labels), ['op_id', 'item_name', 'value', 'chart_time']]
anesth_op_ids = []
anesth_means = []
for op_id, df in tqdm(df_anesthetic.groupby('op_id')):
    end = df['chart_time'].max()
    start = df['chart_time'].min()
    times = pd.DataFrame({'chart_time': np.arange(start, end + 5, 5)})

    df_complete = pd.merge(times, df.loc[df['item_name'] == 'etdes', ['chart_time', 'value']], on='chart_time', how='left')
    df_complete = pd.merge(df_complete, df.loc[df['item_name'] == 'etsevo', ['chart_time', 'value']], on='chart_time', how='left')

    # df_complete.interpolate(''inplace=True)
    df_complete.ffill(inplace=True)
    df_complete.fillna(0, inplace=True)
    df_complete['equiv_MAC'] = (df_complete['value_x'] / 6) + (df_complete['value_y'] / 2)

    anesth_op_ids.append(op_id)
    anesth_means.append(df_complete['equiv_MAC'].mean())
df_anesthetic = pd.DataFrame({'op_id':anesth_op_ids, 'equiv_MAC_totals': anesth_means})


df_final = pd.DataFrame({'op_id': sorted(df_vitals['op_id'].unique())})
df_list = [df_regular, df_cs_average, df_wt_adjusted, df_time_adjusted, df_fluids_agg, df_anesthetic]
for df in df_list:
    df_final = df_final.merge(df, on='op_id', how='left')


df_final.to_csv(output_csv, index=False)