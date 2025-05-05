from pathlib import Path
import pandas as pd
from collections import Counter

def nprint(string):
    print("="*25, string, "="*25)


base_path = Path("/home/server/Projects/data/base/")
aki_path = Path("/home/server/Projects/data/AKI/")
inspire_path = Path("/home/server/Projects/data/INSPIRE/physionet.org/files/inspire/1.3")

labs_path =             inspire_path / "labs.csv"
preopdata_file =        aki_path / "preop_data.csv"
base_combined_csv =     base_path / 'tabular_combined.csv'


cols_to_keep = ["op_id", "subject_id",
                "preop_creatinine", "opend_time"]
df_preop = pd.read_csv(preopdata_file)[cols_to_keep]
df_labs = pd.read_csv(labs_path)
# only merging with this to get parity with create_base op_ids from start
base_combined = pd.read_csv(base_combined_csv)['op_id']
df_preop = df_preop.merge(base_combined, on='op_id', how='inner')


df_preop = df_preop[df_preop['preop_creatinine'].notna()]


# Cuts samples down to almost half because insufficient data
df_creatinine = df_labs[df_labs['item_name'] == 'creatinine']
df_merge = pd.merge(df_preop, df_creatinine, on='subject_id')
df_merge_filtered = df_merge[
    (df_merge['chart_time'] > df_merge['opend_time']) &
    (df_merge['chart_time'] < (df_merge['opend_time'] + 2*24*60))
]

max_creatinine = (
    df_merge_filtered.groupby('op_id')['value']
    .max()
    .reset_index()
    .rename(columns={'value': 'postop_creatinine'})
)
df_aki = pd.merge(df_preop, max_creatinine, on='op_id', how='inner')

df_aki = df_aki[df_aki["preop_creatinine"] <= 4.5]
df_aki["aki"] = df_aki["postop_creatinine"] - df_aki["preop_creatinine"]
df_final = df_aki[['op_id', 'aki']].copy()
df_final['aki_boolean'] = df_final['aki'] > 0.3





base_combined_unnormalized_csv =     base_path / 'tabular_combined_unnormalized.csv'
base_combined_csv =     base_path / 'tabular_combined.csv'
base_preop_csv =        base_path / 'tabular_preop.csv'
base_intraop_csv =      base_path / 'tabular_intraop.csv'

aki_combined_unnormalized_csv =      aki_path / "tabular_combined_unnormalized.csv"
aki_combined_csv =      aki_path / "tabular_combined.csv"
aki_preop_csv =         aki_path / "tabular_preop.csv"
aki_intraop_csv =       aki_path / "tabular_intraop.csv"


df_combined_unnormalized = pd.read_csv(base_combined_unnormalized_csv)
df_combined_unnormalized = df_combined_unnormalized.merge(df_final, on='op_id', how='inner')
df_combined_unnormalized.to_csv(aki_combined_unnormalized_csv, index=False)
nprint(f"wrote to {aki_combined_unnormalized_csv}")

df_combined = pd.read_csv(base_combined_csv)
df_combined = df_combined.merge(df_final, on='op_id', how='inner')
df_combined.to_csv(aki_combined_csv, index=False)
nprint(f"wrote to {aki_combined_csv}")

df_preop = pd.read_csv(base_preop_csv)
df_preop = df_preop.merge(df_final, on='op_id', how='inner')
df_preop.to_csv(aki_preop_csv, index=False)
nprint(f"wrote to {aki_preop_csv}")

df_intraop = pd.read_csv(base_intraop_csv)
df_intraop = df_intraop.merge(df_final, on='op_id', how='inner')
df_intraop.to_csv(aki_intraop_csv, index=False)
nprint(f"wrote to {aki_intraop_csv}")
