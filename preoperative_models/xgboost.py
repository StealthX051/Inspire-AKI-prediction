import numpy as np
import matplotlib.pyplot as plt
import xgboost as xgb
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc

# Load data
with np.load('/home/server/Projects/data/AKI/preop_trainable/unfiltered.npz', allow_pickle=True) as data:
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
y_prob = xgb_classifier.predict_proba(X_test)[:, 1]

# Compute ROC curve
fpr, tpr, _ = roc_curve(y_binary_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('XGBoost Classification: ROC Curve')
plt.legend(loc='lower right')
plt.show()
