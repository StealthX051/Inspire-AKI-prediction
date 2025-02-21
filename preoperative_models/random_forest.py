import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error, mean_squared_error, r2_score, roc_curve, auc

# Load data
with np.load('/home/server/Projects/data/AKI/preop_trainable/unfiltered.npz', allow_pickle=True) as data:
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    y_binary_train = data["y_binary_train"]
    y_binary_test = data["y_binary_test"]
    y_positive_train = data["y_positive_train"]
    y_positive_test = data["y_positive_test"]

# -------------------- RANDOM FOREST REGRESSION --------------------
print("\nRandom Forest Regression:")

# Train model
regressor = RandomForestRegressor(n_estimators=100, random_state=42)
print("training")
regressor.fit(X_train, y_train)

# Predict
y_pred = regressor.predict(X_test)

# Regression Metrics
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

print(f'Mean Absolute Error: {mae:.4f}')
print(f'Mean Squared Error: {mse:.4f}')
print(f'Root Mean Squared Error: {rmse:.4f}')
print(f'R-squared: {r2:.4f}')

# # Plot actual vs predicted values
# plt.figure()
# plt.scatter(y_test, y_pred, alpha=0.5)
# plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
# plt.xlabel('Actual')
# plt.ylabel('Predicted')
# plt.title('Random Forest Regression: Actual vs Predicted')
# plt.show()

# -------------------- RANDOM FOREST CLASSIFICATION --------------------
print("\nRandom Forest Classification:")

# Train classification model
classifier = RandomForestClassifier(n_estimators=100, random_state=42)
classifier.fit(X_train, y_binary_train)

# Predict binary outcomes
y_pred_binary = classifier.predict(X_test)

# Classification Metrics
accuracy = accuracy_score(y_binary_test, y_pred_binary)
print(f'Accuracy: {accuracy:.2f}')

cm = confusion_matrix(y_binary_test, y_pred_binary)
print('Confusion Matrix:')
print(cm)

report = classification_report(y_binary_test, y_pred_binary)
print('Classification Report:')
print(report)

# Predict probabilities for ROC curve
y_prob = classifier.predict_proba(X_test)[:, 1]

# Compute ROC curve
fpr, tpr, _ = roc_curve(y_binary_test, y_prob)
roc_auc = auc(fpr, tpr)

# # Plot ROC curve
# plt.figure()
# plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
# plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
# plt.xlim([0.0, 1.0])
# plt.ylim([0.0, 1.05])
# plt.xlabel('False Positive Rate')
# plt.ylabel('True Positive Rate')
# plt.title('Random Forest Classification: ROC Curve')
# plt.legend(loc='lower right')
# plt.show()
