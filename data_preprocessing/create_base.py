import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

norm_csv = '/home/server/Projects/data/base/normalization_stats.csv'

preop_csv =     "/home/server/Projects/data/AKI/preop_data_test.csv"
intraop_csv =   "/home/server/Projects/data/AKI/feature_engineered.csv"


df_preop = pd.read_csv(preop_csv)
df_intraop = pd.read_csv(intraop_csv)

df = df_preop.merge(df_intraop, on='op_id', how='inner')

combined_csv = '/home/server/Projects/data/base/tabular_combined.csv'
preop_csv = '/home/server/Projects/data/base/tabular_preop.csv'
intraop_csv = '/home/server/Projects/data/base/tabular_intraop.csv'


df.replace([np.inf, -np.inf], np.nan, inplace=True)
cols_to_pop = ["postop_creatinine", "subject_id", 
            "opstart_time", "opend_time", "inhosp_death_time", 
            "allcause_death_time"]
removed = []
for col in cols_to_pop:
    if col in df.columns or "Unnamed" in col:
        removed.append(col)
        df.pop(col)
print(f"Removed columns: {removed}")

# # missing data indicators and zero out missing
# indicator_columns = []
# for col in df.columns:
#     if df[col].isnull().any():
#         # indicates missing, e.g. True => missing
#         indicator_columns.append(pd.Series(df[col].isna(), name=f'{col}_isna'))
# df = pd.concat([df] + indicator_columns, axis=1)
# df.fillna(df.mean(), inplace=True)


# replace outliers
int_columns = df.select_dtypes(include=['int']).columns
df[int_columns] = df[int_columns].astype(float)
np.random.seed(42)

# Ignore columns that are not numerical for outlier handling and normalization
ignore = ['op_id', 'age', 'emop', 'num_card_events', 'antype', 'sex', 'asa']
for col in df.columns:
    if ('department' in col) or ('_isna' in col) or ('aki' in col):
        ignore.append(col)

# remove outliers
for col in df.columns:
    if col not in ignore:
        lower_1 = df[col].quantile(0.01)
        upper_1 = df[col].quantile(0.99)
        lower_05 = df[col].quantile(0.005)
        lower_5 = df[col].quantile(0.05)
        upper_95 = df[col].quantile(0.95)
        upper_995 = df[col].quantile(0.995)
        mask_lower = df[col] < lower_1
        mask_upper = df[col] > upper_1
        df.loc[mask_lower, col] = np.random.uniform(lower_05, lower_5, mask_lower.sum())
        df.loc[mask_upper, col] = np.random.uniform(upper_95, upper_995, mask_upper.sum())


# normalize numeric columns
cols_to_norm = [col for col in df.columns if col not in ignore]
scaler = StandardScaler()

scaler.fit(df[cols_to_norm])

df_stats = pd.DataFrame({
    'mean': scaler.mean_,
    'var': scaler.var_
}, index=cols_to_norm)

# Apply normalization
df[cols_to_norm] = scaler.transform(df[cols_to_norm])

preop_cols = []
for col in df_preop.columns:
    for col_2 in df.columns:
        if col in col_2:
            preop_cols.append(col_2)
intraop_cols = []
for col in df_intraop.columns:
    for col_2 in df.columns:
        if col in col_2:
            intraop_cols.append(col_2)


# you need to give imputer.fit_transform the whole df, not just the columns
# thats why you should do the -99 first
from sklearn.impute import KNNImputer
nan_percentage = (df.isna().mean() * 100).to_dict()
# if missing rate is greater than 10%, replace NANs with -99 post normalization
cols = [col for col, nan_pct in nan_percentage.items() if nan_pct >= 10]
print(f'{len(cols)} columns to be simply flagged with -99')
df[cols] = pd.DataFrame(df[cols].fillna(-99), columns=cols, index=df.index)
# if missing rate is less than 10% use knn imputer. 
# takes 40 minutes
cols = [col for col, nan_pct in nan_percentage.items() if nan_pct < 10]
print(f'{len(cols)} columns to be imputed')
imputer = KNNImputer(n_neighbors=5)
df = pd.DataFrame(imputer.fit_transform(df), columns=df.columns, index=df.index)


print(f'saving output to {combined_csv}')
df.to_csv(combined_csv, index=False)
print(f'saving output to {preop_csv}')
df[preop_cols].to_csv(preop_csv, index=False)
print(f'saving output to {intraop_csv}')
df[intraop_cols].to_csv(intraop_csv, index=False)

print(f'saving normalization stats to {norm_csv}')
df_stats.to_csv(norm_csv)