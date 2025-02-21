import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc
import matplotlib.pyplot as plt

with np.load('/home/server/Projects/data/AKI/preop_trainable/normalized.npz', allow_pickle=True) as data:
    X_train=data["X_train_normalized"]
    X_test=data["X_test_normalized"]
    y_train=data["y_train_normalized"]
    y_test=data["y_test_normalized"]
    y_binary_train=data["y_binary_train_normalized"]
    y_binary_test=data["y_binary_test_normalized"]
    y_positive_train=data["y_positive_train_normalized"]
    y_positive_test=data["y_positive_test_normalized"]

with np.load('/home/server/Projects/data/AKI/preop_trainable/smoted.npz', allow_pickle=True) as data:
    X_train_smote=data["X_train_smote"]
    y_train_smote=data["y_train_smote"]
    y_binary_train_smote=data["y_binary_train_smote"]
    y_positive_train_smote=data["y_positive_train_smote"]

model = LogisticRegression(max_iter=1000)
model.fit(X_train_smote, y_binary_train_smote)

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_binary_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')

cm = confusion_matrix(y_binary_test, y_pred)
print('Confusion Matrix:')
print(cm)

report = classification_report(y_binary_test, y_pred)
print('Classification Report:')
print(report)

# Get predicted probabilities
y_prob = model.predict_proba(X_test)[:, 1]

# Compute ROC curve
fpr, tpr, thresholds = roc_curve(y_binary_test, y_prob)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc='lower right')
plt.show()
