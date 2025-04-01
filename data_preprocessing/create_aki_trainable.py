import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import sys
from collections import Counter
from imblearn.over_sampling import SMOTE

SMOTE_BOOLEAN = False
INCLUDE_PREOP = True
INCLUDE_INTRAOP = False


preop_csv =     "/home/server/Projects/data/AKI/preop_data_andrew.csv"
intraop_csv =   "/home/server/Projects/data/AKI/feature_engineered.csv"


df_preop = pd.read_csv(preop_csv)
df_intraop = pd.read_csv(intraop_csv)

if INCLUDE_PREOP and INCLUDE_INTRAOP:
    output_file =   "/home/server/Projects/data/AKI/tabular_combined.npz"
    csv_version_file = '/home/server/Projects/data/AKI/tabular_combined.csv'
    df = df_preop.merge(df_intraop, on='op_id', how='inner')
elif INCLUDE_PREOP:
    output_file =   "/home/server/Projects/data/AKI/tabular_preop.npz"
    csv_version_file = '/home/server/Projects/data/AKI/tabular_preop.csv'
    df = df_preop.copy()
elif INCLUDE_INTRAOP:
    output_file =   "/home/server/Projects/data/AKI/tabular_intraop.npz"
    csv_version_file = '/home/server/Projects/data/AKI/tabular_intraop.csv'
    # take only aki column from preop
    df = df_preop[['op_id', 'aki']].merge(df_intraop, on='op_id', how='inner') 
else:
    print('You have selected nothing')


df.replace([np.inf, -np.inf], np.nan, inplace=True)
cols_to_pop = ["postop_creatinine", "Unnamed: 0", "subject_id", 
            "opstart_time", "opend_time", "inhosp_death_time", 
            "allcause_death_time", "Unnamed: 0.1", "Unnamed: 0.2"
          ]
for col in cols_to_pop:
    if col in df.columns:
        df.pop(col)

# missing data indicators and zero out missing
indicator_columns = []
for col in df.columns:
    if df[col].isnull().any():
        indicator_columns.append(pd.Series(df[col].isna(), name=f'{col}_isna'))
df = pd.concat([df] + indicator_columns, axis=1)
df.fillna(df.mean(), inplace=True)

#create additional y variable columns
# y_vars = [(df["aki"] > 0.3).rename("aki_boolean"), df["aki"].clip(lower=0).rename("aki_positive")]
# df = pd.concat([df] + y_vars,axis=1)

# replace outliers
int_columns = df.select_dtypes(include=['int']).columns
df[int_columns] = df[int_columns].astype(float)
np.random.seed(42)

# Ignore columns that are not numerical for outlier handling and normalization
ignore = ["sex", "asa", "emop", "num_card_events", 'op_id']
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

cols_to_norm = [col for col in df.columns if col not in ignore]
scaler = StandardScaler()
df[cols_to_norm] = scaler.fit_transform(df[cols_to_norm])

df.to_csv(csv_version_file, index=False)


# # Split BEFORE upsampling (preserves real-world test distribution)
# df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["aki_boolean"])

# if not SMOTE_BOOLEAN:
#     # Upsample minority class from training set
#     print("Upsampling minority class")
#     df_majority = df_train[df_train["aki_boolean"] == 0]
#     df_minority = df_train[df_train["aki_boolean"] == 1]

#     df_minority_upsampled = resample(df_minority,
#                                     replace=True,  # Sample with replacement
#                                     n_samples=len(df_majority),  # Match majority class size
#                                     random_state=42)

#     # Combine upsampled minority with majority class
#     df_train_balanced = pd.concat([df_majority, df_minority_upsampled])
#     df_train_balanced = df_train_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
# elif SMOTE_BOOLEAN:
#     print("Upsampling minority class via IMBLEARN SMOTE")
#     y_binary_train = df_train.pop('aki_boolean')
#     X_train = df_train

#     smote = SMOTE(random_state=42)
#     X_smote, y_smote = smote.fit_resample(X_train, y_binary_train)
#     df_train_balanced = pd.concat([X_smote, y_smote], axis=1)
#     df_train_balanced = df_train_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# # Extract features and labels
# y_train = df_train_balanced.pop('aki').values
# y_binary_train = df_train_balanced.pop('aki_boolean').values
# y_positive_train = df_train_balanced.pop('aki_positive').values
# X_train = df_train_balanced.values

# y_test = df_test.pop('aki').values
# y_binary_test = df_test.pop('aki_boolean').values
# y_positive_test = df_test.pop('aki_positive').values
# X_test = df_test.values


# # Save the data
# np.savez_compressed(
#     output_file,
#     X_train=X_train,
#     X_test=X_test,
#     y_train=y_train,
#     y_test=y_test,
#     y_binary_train=y_binary_train,
#     y_binary_test=y_binary_test,
#     y_positive_train=y_positive_train,
#     y_positive_test=y_positive_test
# )
