import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

# Load data
with np.load('/home/server/Projects/data/AKI/preop_trainable/unfiltered.npz', allow_pickle=True) as data:
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]  # Continuous target (for regression models)
    y_test = data["y_test"]
    y_binary_train = data["y_binary_train"]
    y_binary_test = data["y_binary_test"]

# -------------------- Base Models --------------------
# Logistic Regression (Used Directly)
logistic = LogisticRegression(max_iter=1000)

# Linear Regression (Will Convert Predictions Using `> 0.3`)
linear = LinearRegression()

# K-Nearest Neighbors
knn = KNeighborsClassifier(n_neighbors=250, weights='distance')

# Support Vector Machine (SVM)
svm = SVC(kernel='rbf', probability=True, random_state=42)

# Random Forest
random_forest = RandomForestClassifier(n_estimators=100, random_state=42)

# XGBoost
xgb_classifier = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    use_label_encoder=False,
    n_estimators=1000,
    learning_rate=0.1,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# -------------------- Train Regression Models Separately --------------------
# Train Linear Regression
linear.fit(X_train, y_train)

# Predict probabilities using Linear Regression & threshold at 0.3
y_pred_linear = (linear.predict(X_test) > 0.3).astype(int)

# -------------------- Ensemble Model --------------------
# Custom Class for Including Linear Regression in VotingClassifier
from sklearn.base import BaseEstimator, ClassifierMixin

class LinearRegressionBinaryClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, threshold=0.3):
        self.threshold = threshold
        self.model = LinearRegression()

    def fit(self, X, y):
        self.model.fit(X, y)
        return self

    def predict(self, X):
        y_pred = self.model.predict(X)
        return (y_pred > self.threshold).astype(int)

# Wrapping Linear Regression for VotingClassifier
linear_binary = LinearRegressionBinaryClassifier(threshold=0.3)

# Voting Classifier (Soft Voting: Uses Predicted Probabilities)
ensemble = VotingClassifier(
    estimators=[
        ('logistic', logistic),
        ('linear', linear_binary),
        ('knn', knn),
        ('svm', svm),
        ('random_forest', random_forest),
        ('xgb', xgb_classifier)
    ],
    voting='soft'  # Uses probability-based voting
)

# Train Ensemble Model
ensemble.fit(X_train, y_binary_train)

# Predict binary labels
y_pred = ensemble.predict(X_test)

# Classification Metrics
accuracy = accuracy_score(y_binary_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

cm = confusion_matrix(y_binary_test, y_pred)
print('Confusion Matrix:')
print(cm)

report = classification_report(y_binary_test, y_pred)
print('Classification Report:')
print(report)

# Predict probabilities for ROC curve
y_prob = ensemble.predict_proba(X_test)[:, 1]

# Compute ROC curve
fpr, tpr, _ = roc_curve(y_binary_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Full Ensemble Model: ROC Curve')
plt.legend(loc='lower right')
plt.show()
