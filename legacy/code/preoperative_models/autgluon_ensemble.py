import numpy as np
import matplotlib.pyplot as plt
from autogluon.tabular import TabularPredictor
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import pandas as pd

# Load data
with np.load('/home/server/Projects/data/AKI/preop_trainable/unfiltered.npz', allow_pickle=True) as data:
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_binary_train = data["y_binary_train"]
    y_binary_test = data["y_binary_test"]

# Convert NumPy arrays into Pandas DataFrames (AutoGluon expects a DataFrame)
train_df = pd.DataFrame(X_train)
train_df["target"] = y_binary_train  # Append the target column

test_df = pd.DataFrame(X_test)
test_df["target"] = y_binary_test  # Append the target column

# -------------------- Train AutoGluon Model --------------------
print("\nTraining AutoGluon...")

predictor = TabularPredictor(
    label="target",  # Target column for prediction
    problem_type="binary"
).fit(train_df, presets="medium_quality", time_limit=600)  # 10-minute training limit

# -------------------- Make Predictions --------------------
print("\nEvaluating AutoGluon Model...")

# Predict binary labels
y_pred = predictor.predict(test_df.drop(columns=["target"]))

# Predict probabilities for ROC Curve
y_prob = predictor.predict_proba(test_df.drop(columns=["target"]))[1]  # Probability of class 1

# -------------------- Evaluate Model Performance --------------------
accuracy = accuracy_score(y_binary_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

cm = confusion_matrix(y_binary_test, y_pred)
print('Confusion Matrix:')
print(cm)

report = classification_report(y_binary_test, y_pred)
print('Classification Report:')
print(report)

# Compute ROC curve
fpr, tpr, _ = roc_curve(y_binary_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC Curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('AutoGluon Ensemble: ROC Curve')
plt.legend(loc='lower right')
plt.show()
