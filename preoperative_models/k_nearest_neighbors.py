import numpy as np
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsRegressor, KNeighborsClassifier
from utils import performance_dict

file = "/home/server/Projects/data/AKI/tabular_intraop.npz"

# Load data
with np.load(file, allow_pickle=True) as data:
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    y_binary_train = data["y_binary_train"]
    y_binary_test = data["y_binary_test"]
    y_positive_train = data["y_positive_train"]
    y_positive_test = data["y_positive_test"]

# # -------------------- K-NEAREST NEIGHBORS REGRESSION --------------------
# print("\nKNN Regression:")

# # Train KNN Regressor
# knn_regressor = KNeighborsRegressor(n_neighbors=200)
# knn_regressor.fit(X_train, y_train)

# # Predict
# y_pred = knn_regressor.predict(X_test)


# # Fit LOWESS smoother (better for non-parametric models like KNN)
# import statsmodels.api as sm
# lowess = sm.nonparametric.lowess(y_pred, y_test, frac=0.3)
# plt.plot(lowess[:, 0], lowess[:, 1], color='red', linewidth=2, label="LOWESS Trend")


# -------------------- K-NEAREST NEIGHBORS CLASSIFICATION --------------------
print("\nKNN Classification:")

# Train KNN Classifier
knn_classifier = KNeighborsClassifier(n_neighbors=200)
knn_classifier.fit(X_train, y_binary_train)

# Predict binary outcomes
y_pred_binary = knn_classifier.predict(X_test)

# Predict probabilities for ROC curve
y_prob = knn_classifier.predict_proba(X_test)[:, 1]

performance_dict(y_binary_test, y_pred_binary, y_prob, bool_print=True, plot=False, copy_print=True)