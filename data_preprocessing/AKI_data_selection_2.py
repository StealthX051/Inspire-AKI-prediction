from pathlib import Path
import pandas as pd
from collections import Counter

def nprint(string):
    print("="*25, string, "="*25)

# All the input paths
inspire_path = Path("/home/server/Projects/data/INSPIRE/physionet.org/files/inspire/1.3")
labs_path = inspire_path / "labs.csv"
vitals_path = inspire_path / "vitals.csv"
ops_path = inspire_path / "operations.csv"

preopdata_file = "/home/server/Projects/data/AKI/preop_data.csv"
preopdata_file_andrew = "/home/server/Projects/data/AKI/preop_data_andrew.csv"

nprint("starting")
df_labs = pd.read_csv(labs_path)
df_labs["chart_time"] = df_labs["chart_time"].astype(float)
df_vitals = pd.read_csv(vitals_path)
df_preop = pd.read_csv(preopdata_file)
df_ops = pd.read_csv(ops_path)
nprint("finished reading csvs")

df_preop = df_preop[df_preop["asa"] < 6]
df_preop = df_preop[df_preop["age"] >= 18]
df_preop = df_preop.dropna(subset="opend_time")
df_preop = df_preop.dropna(subset="opstart_time")
df_preop["op_len"] = df_preop["opend_time"] - df_preop["opstart_time"]

#change
df_ops = df_ops.drop(df_ops[df_ops['antype'] == 'Regional'].index)
df_ops.loc[df_ops['antype'] == 'General', 'antype'] = 0     #replace antypes with numbers, after removing rows with regional set as antype
df_ops.loc[df_ops['antype'] == 'MAC', 'antype'] = 1
df_ops.loc[df_ops['antype'] == 'Neuraxial', 'antype'] = 1

col = df_ops.columns.get_loc('department')                  #don't want to just add encodings to end of dataframe, so insert it where department used to be
num_cols_added = len(Counter(df_ops['department']))
ops_general = pd.get_dummies(df_ops, columns=['department'])
ops_gen_cols_to_keep = ['op_id', 'subject_id', 'antype']
for column_idx in range(col, col + num_cols_added):
    department_name = ops_general.columns[-1]
    ops_gen_cols_to_keep.append(department_name)
    ops_general.insert(column_idx, department_name, ops_general.pop(department_name))
df_preop = pd.merge(df_preop, ops_general[ops_gen_cols_to_keep], on=['op_id', 'subject_id'], how='inner')

# ops_general = df_ops[df_ops['antype'] == 'General']
# df_preop = pd.merge(df_preop, ops_general[['op_id', 'subject_id']], on=['op_id', 'subject_id'], how='inner')

nprint("finished basic filtering")

item_names = [
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

for item_name in item_names:
    df_preop = pd.merge_asof(df_preop.sort_values('opstart_time'), 
                    df_labs.loc[df_labs['item_name'] == item_name].sort_values('chart_time'), # grab rows w the item name we want and sort by chart_time
                    left_on='opstart_time', right_on='chart_time', by='subject_id',           # chooses row in df_labs w greatest chart_time that is still less than opstart_time and matches subject_id
                    tolerance=90 * 24 * 60, suffixes=('', '_'))                               # 90 day tolerance
    df_preop.drop(columns=['chart_time', 'item_name'], inplace=True)
    df_preop.rename(columns={'value':f'preop_{item_name}'}, inplace=True)
nprint("finished getting preop data from time series")

df_creatinine = df_labs[df_labs['item_name'] == 'creatinine']
df_merge = pd.merge(df_preop, df_creatinine, on='subject_id', suffixes=('_preop', '_lab'))
df_merge_filtered = df_merge[
    (df_merge['chart_time'] > df_merge['opend_time']) &
    (df_merge['chart_time'] < (df_merge['opend_time'] + 2*24*60))
]
max_creatinine = (
    df_merge_filtered.groupby(['subject_id', 'op_id'])['value']
    .max()
    .reset_index()
    .rename(columns={'value': 'postop_creatinine'})
)
df_preop = pd.merge(df_preop, max_creatinine, on=['subject_id', 'op_id'], how='inner')
nprint("finished getting postop data from time series")

df_preop = df_preop[df_preop["preop_creatinine"] <= 4.5]

prefixes_to_exclude = ["10", "0TY", "B50", "B51"]
mask = df_ops["icd10_pcs"].astype(str).str.startswith(tuple(prefixes_to_exclude))
ops_to_exclude = df_ops.loc[mask, "op_id"]
df_preop = df_preop[~df_preop["op_id"].isin(ops_to_exclude)]
nprint("finished filtering out some procedure prefixes")

df_preop["aki"] = df_preop["postop_creatinine"] - df_preop["preop_creatinine"]
nprint("calculated aki")

df_preop.to_csv(preopdata_file_andrew)
nprint(f"wrote to {preopdata_file_andrew}")
