import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_curve, auc, precision_recall_curve, average_precision_score
import matplotlib.pyplot as plt
import pandas as pd
from utils import performance_dict

input_file = "/home/server/Projects/data/AKI/tabular_combined.npz"
input_file = '/home/server/Projects/data/AKI/preop_trainable/unfiltered_andrew.npz'

with np.load(input_file, allow_pickle=True) as data:
    X_train=data["X_train"]
    X_test=data["X_test"]
    y_train=data["y_train"]
    y_test=data["y_test"]
    y_binary_train=data["y_binary_train"]
    y_binary_test=data["y_binary_test"]
    y_positive_train=data["y_positive_train"]
    y_positive_test=data["y_positive_test"]

architectures = [   (8, 8, 4, 32, 2),
                    (8, 16, 4, 16, 2),
                    (32, 16, 32, 2),
                    (64, 16, 16, 8)
                    ]
for arch in architectures:
    model = MLPClassifier(random_state=42, max_iter=1000, hidden_layer_sizes=arch)
    model.fit(X_train, y_binary_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    print(arch)
    performance_dict(y_binary_test, y_pred, y_prob, bool_print=True, plot=False)