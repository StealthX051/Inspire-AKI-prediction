import numpy as np
from sklearn.linear_model import LogisticRegression
from utils import performance_dict

file = "/home/server/Projects/data/AKI/tabular_intraop.npz"

with np.load(file, allow_pickle=True) as data:
    X_train=data["X_train"]
    X_test=data["X_test"]
    y_train=data["y_train"]
    y_test=data["y_test"]
    y_binary_train=data["y_binary_train"]
    y_binary_test=data["y_binary_test"]
    y_positive_train=data["y_positive_train"]
    y_positive_test=data["y_positive_test"]

model = LogisticRegression(max_iter=10000)
model.fit(X_train, y_binary_train)

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

performance_dict(y_binary_test, y_pred, y_prob, bool_print=True, plot=False, copy_print=True)