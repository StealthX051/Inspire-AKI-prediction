import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

# df = pd.read_csv('/home/server/Projects/data/AKI/aki_data.csv')
df = pd.read_csv("/home/server/Projects/data/AKI/preop_data_andrew.csv")

df["sex"] = df["sex"] == "M"
df = df[(df['weight'] != 0) & (df['height'] != 0)]  # remove rows with 0 weight or height
cols_to_pop = ["postop_creatinine", "Unnamed: 0", "op_id", "subject_id", 
            "opstart_time", "opend_time", "inhosp_death_time", "allcause_death_time",
            "Unnamed: 0.1", "Unnamed: 0.2"
          ]
for col in cols_to_pop:
    if col in df.columns:
        df.pop(col)

# missing data indicators and zero out missing
for col in df.columns:
    # Create a new column with a suffix '_isna' indicating NaN status
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
print(ignore)

# remove outliers and normalize
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


# # Create new csv to avoid running this script again
# print("Saving to trainable csv")
# df.to_csv("/home/server/Projects/data/AKI/aki_data_trainable.csv", index=False)

# df = pd.read_csv('/home/server/Projects/data/AKI/aki_data_trainable.csv')

# Smote with resampling ONLY on training data, not on test data.
# This is to preserve the real-world test distribution.

# Step 1: Split BEFORE upsampling (preserves real-world test distribution)
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["aki_boolean"])

# Step 2: Upsample minority class in the training set only
df_majority = df_train[df_train["aki_boolean"] == 0]
df_minority = df_train[df_train["aki_boolean"] == 1]

df_minority_upsampled = resample(df_minority,
                                 replace=True,  # Sample with replacement
                                 n_samples=len(df_majority),  # Match majority class size
                                 random_state=42)

# Combine upsampled minority with majority class
df_train_balanced = pd.concat([df_majority, df_minority_upsampled])
df_train_balanced = df_train_balanced.sample(frac=1, random_state=42).reset_index(drop=True)

# Step 3: Extract features and labels
y_train = df_train_balanced.iloc[:, -3].values
y_binary_train = df_train_balanced.iloc[:, -2].values
y_positive_train = df_train_balanced.iloc[:, -1].values
X_train = df_train_balanced.iloc[:, :-3].values

y_test = df_test.iloc[:, -3].values
y_binary_test = df_test.iloc[:, -2].values
y_positive_test = df_test.iloc[:, -1].values
X_test = df_test.iloc[:, :-3].values

# Step 4: Save the data
# np.savez_compressed(
#     "/home/server/Projects/data/AKI/preop_trainable/unfiltered.npz",
#     X_train=X_train,
#     X_test=X_test,
#     y_train=y_train,
#     y_test=y_test,
#     y_binary_train=y_binary_train,
#     y_binary_test=y_binary_test,
#     y_positive_train=y_positive_train,
#     y_positive_test=y_positive_test
# )
np.savez_compressed(
    "/home/server/Projects/data/AKI/preop_trainable/unfiltered_andrew.npz",
    X_train=X_train,
    X_test=X_test,
    y_train=y_train,
    y_test=y_test,
    y_binary_train=y_binary_train,
    y_binary_test=y_binary_test,
    y_positive_train=y_positive_train,
    y_positive_test=y_positive_test
)


# # SMOTe
# df_majority = df[df["aki_boolean"] == 0]
# df_minority = df[df["aki_boolean"] == 1]

# # Upsample minority class
# df_minority_upsampled = resample(df_minority,
#                                  replace=True,     # Sample with replacement
#                                  n_samples=len(df_majority),    # Match number in majority class
#                                  random_state=42)  # Reproducible results

# # Combine majority class with upsampled minority class
# df = pd.concat([df_majority, df_minority_upsampled])

# # # Create new csv to avoid running this script again
# # print("Saving to trainable csv")
# # df.to_csv("/home/server/Projects/data/AKI/aki_data_trainable.csv", index=False)

# 

# y = df.iloc[:, -3].values
# y_binary = df.iloc[:, -2].values
# y_positive = df.iloc[:, -1].values
# X = df.iloc[:, :-3].values

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# _, _, y_binary_train, y_binary_test = train_test_split(X, y_binary, test_size=0.2, random_state=42)
# _, _, y_positive_train, y_positive_test = train_test_split(X, y_positive, test_size=0.2, random_state=42)

# np.savez_compressed(
#     "/home/server/Projects/data/AKI/preop_trainable/unfiltered.npz", 
#     X_train=X_train, 
#     X_test=X_test, 
#     y_train=y_train, 
#     y_test=y_test, 
#     y_binary_train=y_binary_train, 
#     y_binary_test=y_binary_test, 
#     y_positive_train=y_positive_train, 
#     y_positive_test=y_positive_test
# )

# to read:
with np.load('/home/server/Projects/data/AKI/preop_trainable/unfiltered.npz', allow_pickle=True) as data:
    X_train=data["X_train"]
    X_test=data["X_test"]
    y_train=data["y_train"]
    y_test=data["y_test"]
    y_binary_train=data["y_binary_train"]
    y_binary_test=data["y_binary_test"]
    y_positive_train=data["y_positive_train"]
    y_positive_test=data["y_positive_test"]