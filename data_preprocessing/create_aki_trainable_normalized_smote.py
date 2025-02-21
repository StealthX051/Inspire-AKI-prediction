import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE

df = pd.read_csv('/home/server/Projects/data/AKI/aki_data_normalized.csv')

# Smote with resampling ONLY on training data, not on test data.
# This is to preserve the real-world test distribution.

# Step 1: Split BEFORE upsampling (preserves real-world test distribution)
df = df[[col for col in df.columns if col != "aki_boolean"] + ["aki_boolean"]]
df_train, df_test = train_test_split(df, test_size=0.2, random_state=42, stratify=df["aki_boolean"])

y_binary_train = df_train.iloc[:, -1]
X_train = df_train.iloc[:, :-1]

# smote
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_binary_train)

df = pd.concat([X_res, y_res], axis=1)
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Step 3: Extract features and labels
y_train = df.iloc[:, -3].values
y_binary_train = df.iloc[:, -1].values
y_positive_train = df.iloc[:, -2].values
X_train = df.iloc[:, :-3].values

# Step 4: Save the data
np.savez_compressed(
    "/home/server/Projects/data/AKI/preop_trainable/smoted.npz",
    X_train_smote=X_train,
    y_train_smote=y_train,
    y_binary_train_smote=y_binary_train,
    y_positive_train_smote=y_positive_train,
)

# to read:
with np.load('/home/server/Projects/data/AKI/preop_trainable/smoted.npz', allow_pickle=True) as data:
    X_train_smote=data["X_train_smote"]
    y_train_smote=data["y_train_smote"]
    y_binary_train_smote=data["y_binary_train_smote"]
    y_positive_train_smote=data["y_positive_train_smote"]
