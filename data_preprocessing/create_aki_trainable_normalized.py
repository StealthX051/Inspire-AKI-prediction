import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import resample

df = pd.read_csv('/home/server/Projects/data/AKI/aki_data_normalized.csv')

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
np.savez_compressed(
    "/home/server/Projects/data/AKI/preop_trainable/normalized.npz",
    X_train_normalized=X_train,
    X_test_normalized=X_test,
    y_train_normalized=y_train,
    y_test_normalized=y_test,
    y_binary_train_normalized=y_binary_train,
    y_binary_test_normalized=y_binary_test,
    y_positive_train_normalized=y_positive_train,
    y_positive_test_normalized=y_positive_test,
)

# to read:
with np.load('/home/server/Projects/data/AKI/preop_trainable/normalized.npz', allow_pickle=True) as data:
    X_train=data["X_train_normalized"]
    X_test=data["X_test_normalized"]
    y_train=data["y_train_normalized"]
    y_test=data["y_test_normalized"]
    y_binary_train=data["y_binary_train_normalized"]
    y_binary_test=data["y_binary_test_normalized"]
    y_positive_train=data["y_positive_train_normalized"]
    y_positive_test=data["y_positive_test_normalized"]
