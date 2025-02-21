import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.utils import resample
import sys
from collections import Counter
from imblearn.over_sampling import SMOTE

# Set to True to enable SMOTE resampling instead of simple upsampling
SMOTE_BOOLEAN = False
if 'smote' in sys.argv: #lol
    SMOTE_BOOLEAN = True

# df = pd.read_csv('/home/server/Projects/data/AKI/aki_data.csv')
df = pd.read_csv("/home/server/Projects/data/AKI/preop_data_andrew.csv")

df["sex"] = df["sex"] == "M"
df = df[(df['weight'] != 0) & (df['height'] != 0) & (df['op_id'] != 435191458)]
cols_to_pop = ["postop_creatinine", "Unnamed: 0", "op_id", "subject_id", 
            "opstart_time", "opend_time", "inhosp_death_time", 
            "allcause_death_time", "Unnamed: 0.1", "Unnamed: 0.2"
          ]
for col in cols_to_pop:
    if col in df.columns:
        df.pop(col)


# missing data indicators and zero out missing
for col in df.columns:
    # Create a new column with a suffix '_isna' indicating NaN status 
    # if the column contains any NA values
    if df[col].isnull().values.any() > 0:
        df[f'{col}_isna'] = df[col].isna()
df.fillna(df.mean(), inplace=True)

# move y vars to end
col_to_move = "aki"
df = df[[col for col in df.columns if col != col_to_move] + [col_to_move]]
df["aki_boolean"] = (df["aki"] > 0.3)
df["aki_positive"] = df["aki"].clip(lower=0)

# replace outliers w more reasonable values
int_columns = df.select_dtypes(include=['int']).columns
df[int_columns] = df[int_columns].astype(float)
np.random.seed(42)

# Ignore columns that are not numerical
ignore = ["sex", "asa", "emop", "num_card_events"]
for col in df.columns:
    if 'department' in col:
        ignore.append(col)

# remove outliers
for col in df.columns:
    if "isna" in col or "aki" in col or col in ignore:
        continue
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


# Normalize only the specified columns
cols_to_norm = ['age', 'height', 'weight', 'BSA', 'BMI', 'booking_case_length', 
        'num_card_events',  'last_preop_scr',
       'min_preop_scr', 'preop_total_protein', 'preop_sodium',
       'preop_potassium', 'preop_platelet', 'preop_glucose', 'preop_wbc',
       'preop_alt', 'preop_chloride', 'preop_lymphocyte', 'preop_phosphorus',
       'preop_albumin', 'preop_fibrinogen', 'preop_creatinine', 'preop_ptinr',
       'preop_total_bilirubin', 'preop_alp', 'preop_aptt', 'preop_calcium',
       'preop_bun', 'preop_ast', 'preop_crp', 'preop_hb', 'preop_hct',
       'preop_seg', 'op_len']
scaler = StandardScaler()
df[cols_to_norm] = scaler.fit_transform(df[cols_to_norm])



# Split BEFORE upsampling (preserves real-world test distribution)
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["aki_boolean"])

if not SMOTE_BOOLEAN:
    # Upsample minority class from training set
    print("Upsampling minority class")
    df_majority = df_train[df_train["aki_boolean"] == 0]
    df_minority = df_train[df_train["aki_boolean"] == 1]

    df_minority_upsampled = resample(df_minority,
                                    replace=True,  # Sample with replacement
                                    n_samples=len(df_majority),  # Match majority class size
                                    random_state=42)

    # Combine upsampled minority with majority class
    df_train_balanced = pd.concat([df_majority, df_minority_upsampled])
    df_train_balanced = df_train_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
elif SMOTE_BOOLEAN:
    print("Upsampling minority class via IMBLEARN SMOTE")
    y_binary_train = df_train.pop('aki_boolean')
    X_train = df_train

    smote = SMOTE(random_state=42)
    X_smote, y_smote = smote.fit_resample(X_train, y_binary_train)
    df_train_balanced = pd.concat([X_smote, y_smote], axis=1)
    df_train_balanced = df_train_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Extract features and labels
y_train = df_train_balanced.pop('aki').values
y_binary_train = df_train_balanced.pop('aki_boolean').values
y_positive_train = df_train_balanced.pop('aki_positive').values
X_train = df_train_balanced.values

y_test = df_test.pop('aki').values
y_binary_test = df_test.pop('aki_boolean').values
y_positive_test = df_test.pop('aki_positive').values
X_test = df_test.values


# Step 4: Save the data
# np.savez_compressed(
#     "/home/server/Projects/data/AKI/preop_trainable/unfiltered_andrew.npz",
#     X_train=X_train,
#     X_test=X_test,
#     y_train=y_train,
#     y_test=y_test,
#     y_binary_train=y_binary_train,
#     y_binary_test=y_binary_test,
#     y_positive_train=y_positive_train,
#     y_positive_test=y_positive_test
# )
