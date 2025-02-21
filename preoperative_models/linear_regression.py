import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, mean_absolute_error, mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

with np.load('/home/server/Projects/data/AKI/preop_trainable/unfiltered.npz', allow_pickle=True) as data:
    X_train=data["X_train"]
    X_test=data["X_test"]
    y_train=data["y_train"]
    y_test=data["y_test"]
    y_binary_train=data["y_binary_train"]
    y_binary_test=data["y_binary_test"]
    y_positive_train=data["y_positive_train"]
    y_positive_test=data["y_positive_test"]

with np.load('/home/server/Projects/data/AKI/preop_trainable/normalized.npz', allow_pickle=True) as data:
    X_train=data["X_train_normalized"]
    X_test=data["X_test_normalized"]
    y_train=data["y_train_normalized"]
    y_test=data["y_test_normalized"]
    y_binary_train=data["y_binary_train_normalized"]
    y_binary_test=data["y_binary_test_normalized"]
    y_positive_train=data["y_positive_train_normalized"]
    y_positive_test=data["y_positive_test_normalized"]

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# Calculate Mean Absolute Error
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae}')

# Calculate Mean Squared Error
mse = mean_squared_error(y_test, y_pred)
print(f'Mean Squared Error: {mse}')

# Calculate Root Mean Squared Error
rmse = np.sqrt(mse)
print(f'Root Mean Squared Error: {rmse}')

# Calculate R-squared
r2 = r2_score(y_test, y_pred)
print(f'R-squared: {r2}')

# Plotting the actual vs predicted values
plt.scatter(y_test, y_pred)
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
plt.xlabel('Actual')
plt.ylabel('Predicted')
plt.title('Actual vs Predicted')
plt.show()

y_pred_binary = y_pred > 0.3

accuracy = accuracy_score(y_binary_test, y_pred_binary)
print(f'Accuracy: {accuracy:.2f}')

cm = confusion_matrix(y_binary_test, y_pred_binary)
print('Confusion Matrix:')
print(cm)

report = classification_report(y_binary_test, y_pred_binary)
print('Classification Report:')
print(report)
