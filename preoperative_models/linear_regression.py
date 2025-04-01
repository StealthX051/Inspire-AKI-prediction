import numpy as np
from sklearn.linear_model import LinearRegression
from utils import performance_dict, sigmoid

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

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)

# # Calculate Mean Absolute Error
# mae = mean_absolute_error(y_test, y_pred)
# print(f'Mean Absolute Error: {mae}')

# # Calculate Mean Squared Error
# mse = mean_squared_error(y_test, y_pred)
# print(f'Mean Squared Error: {mse}')

# # Calculate Root Mean Squared Error
# rmse = np.sqrt(mse)
# print(f'Root Mean Squared Error: {rmse}')

# # Calculate R-squared
# r2 = r2_score(y_test, y_pred)
# print(f'R-squared: {r2}')

y_pred_binary = y_pred > 0.3
y_prob = sigmoid(y_pred - 0.3)

performance_dict(y_binary_test, y_pred_binary, y_prob, bool_print=True, plot=False, copy_print=True)
