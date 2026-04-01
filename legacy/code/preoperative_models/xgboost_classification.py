# this file cannot be called 'xgboost.py' or it interferes with the import

import numpy as np
import xgboost as xgb
from utils import performance_dict


file = "/home/server/Projects/data/AKI/tabular_intraop.npz"

# Load data
with np.load(file, allow_pickle=True) as data:
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_binary_train = data["y_binary_train"]
    y_binary_test = data["y_binary_test"]

# -------------------- XGBoost Classification --------------------
print("\nXGBoost Classification:")

# Define the XGBoost model
xgb_classifier = xgb.XGBClassifier(
    objective="binary:logistic",  # Binary classification (log loss)
    eval_metric="logloss",        # Logarithmic loss for better convergence
    use_label_encoder=False,      # Avoids unnecessary warnings
    n_estimators=1000,             # Number of boosting rounds
    learning_rate=0.1,            # Step size shrinkage
    max_depth=6,                  # Limits tree depth for regularization
    subsample=0.8,                # Prevents overfitting
    colsample_bytree=0.8,         # Reduces features per tree to avoid overfitting
    random_state=42
)

# Train the model
xgb_classifier.fit(X_train, y_binary_train)

# Predict binary labels
y_pred = xgb_classifier.predict(X_test)
y_prob = xgb_classifier.predict_proba(X_test)[:, 1]

performance_dict(y_binary_test, y_pred, y_prob, bool_print=True, plot=False, copy_print=True)
