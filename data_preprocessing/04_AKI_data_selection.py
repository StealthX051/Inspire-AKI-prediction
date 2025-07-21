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
ward_vitals_path = inspire_path / "ward_vitals.csv"

df_ward = pd.read_csv(ward_vitals_path)
df_dialysis = df_ward[df_ward['item_name'] == 'crrt'][['subject_id', 'value']]
df_dialysis = df_dialysis.rename(columns={'value': 'dialysis'})
df_dialysis = df_dialysis.drop_duplicates(subset='subject_id', keep='first').reset_index(drop=True)

cols_to_keep = ["op_id", "subject_id",
                "preop_creatinine", "opend_time"]
df_preop = pd.read_csv(preopdata_file)[cols_to_keep]
df_labs = pd.read_csv(labs_path)
# only merging with this to get parity with create_base op_ids from start
base_combined = pd.read_csv(base_combined_csv)['op_id']
df_preop = df_preop.merge(base_combined, on='op_id', how='inner')


df_preop = df_preop[df_preop['preop_creatinine'].notna()]
df_preop = df_preop[df_preop['preop_creatinine'] < 4.5]

# Cuts samples down to almost half because insufficient data
df_creatinine = df_labs[df_labs['item_name'] == 'creatinine']
df_merge = pd.merge(df_preop, df_creatinine, on='subject_id')

df_aki = df_preop.copy()
df_aki = df_aki.merge(df_dialysis, on='subject_id', how='left')

for n_days in [2, 7]:
    n_minutes = n_days * 24 * 60
    df_merge_filtered = df_merge[
        (df_merge['chart_time'] > df_merge['opend_time']) &
        (df_merge['chart_time'] <= (df_merge['opend_time'] + n_minutes))
    ]
    max_creatinine = (
        df_merge_filtered.groupby('op_id')['value']
        .max()
        .reset_index()
        .rename(columns={'value': f'postop_creatinine_{n_days}_days'})
    )
    df_aki = pd.merge(df_aki, max_creatinine, on='op_id', how='outer')

df_aki = df_aki[~df_aki[['postop_creatinine_2_days', 'postop_creatinine_7_days', 'dialysis']].isna().all(axis=1)]
df_aki = df_aki.fillna({'dialysis': 0})

df_aki['crt_7_day_ratio'] = df_aki["postop_creatinine_7_days"] / df_aki["preop_creatinine"]
df_aki['aki_1'] =   ((df_aki['crt_7_day_ratio'] > 1.5) & (df_aki['crt_7_day_ratio'] < 2)) | \
                    ((df_aki['postop_creatinine_2_days'] - df_aki['preop_creatinine']) > 0.3)
df_aki['aki_2'] =   (df_aki['crt_7_day_ratio'] >= 2) & (df_aki['crt_7_day_ratio'] < 3)
df_aki['aki_3'] =   (df_aki['crt_7_day_ratio'] >= 3) | (df_aki["postop_creatinine_7_days"] > 4) | \
                    (df_aki['dialysis'])

df_aki['aki_boolean'] = df_aki[['aki_2', 'aki_3']].any(axis=1).astype(bool)
df_final = df_aki[['op_id', 'aki_boolean']].copy()


# base_combined_unnormalized_csv =     base_path / 'tabular_combined_unnormalized.csv'
base_combined_csv =     base_path / 'tabular_combined.csv'
base_preop_csv =        base_path / 'tabular_preop.csv'
base_intraop_csv =      base_path / 'tabular_intraop.csv'

# aki_combined_unnormalized_csv =      aki_path / "tabular_combined_unnormalized.csv"
aki_combined_csv =      aki_path / "tabular_combined.csv"
aki_preop_csv =         aki_path / "tabular_preop.csv"
aki_intraop_csv =       aki_path / "tabular_intraop.csv"

# deprecated
# df_combined_unnormalized = pd.read_csv(base_combined_unnormalized_csv)
# df_combined_unnormalized = df_combined_unnormalized.merge(df_final, on='op_id', how='inner')
# df_combined_unnormalized.to_csv(aki_combined_unnormalized_csv, index=False)
# nprint(f"wrote to {aki_combined_unnormalized_csv}")

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
