import numpy as np
from sklearn.svm import SVC
from utils import performance_dict

file = "/home/server/Projects/data/AKI/tabular_combined.npz"

# Load data
with np.load(file, allow_pickle=True) as data:
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_binary_train = data["y_binary_train"]
    y_binary_test = data["y_binary_test"]

# -------------------- SUPPORT VECTOR CLASSIFICATION (SVC) --------------------
print("\nSVM Classification:")

# Train SVM classifier (RBF kernel works best for most cases)
svc = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
svc.fit(X_train, y_binary_train)

# Predict binary labels
y_pred = svc.predict(X_test)

y_prob = svc.predict_proba(X_test)[:, 1]

performance_dict(y_binary_test, y_pred, y_prob, bool_print=True, plot=False, copy_print=True)