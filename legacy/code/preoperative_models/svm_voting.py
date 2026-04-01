from joblib import Parallel, delayed
from utils import performance_dict
import numpy as np
from sklearn.svm import SVC
from tqdm import tqdm

file = "/home/server/Projects/data/AKI/tabular_combined.npz"

# Load data
with np.load(file, allow_pickle=True) as data:
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_binary_train = data["y_binary_train"]
    y_binary_test = data["y_binary_test"]

def train_and_predict(X_batch, y_batch, X_test):
    svc = SVC(kernel='rbf', C=1.0, probability=True, random_state=42)
    svc.fit(X_batch, y_batch)
    y_pred = svc.predict(X_test)
    y_prob = svc.predict_proba(X_test)[:, 1]
    return y_pred, y_prob

batch_size = 2000
num_batches = int(np.ceil(len(X_train) / batch_size))

# Parallel processing
results = Parallel(n_jobs=-1)(delayed(train_and_predict)(
    X_train[i * batch_size : (i + 1) * batch_size], 
    y_binary_train[i * batch_size : (i + 1) * batch_size], 
    X_test
) for i in tqdm(range(num_batches)))

# Aggregate results
y_pred_sum = np.sum([res[0] for res in results], axis=0)
y_prob_sum = np.sum([res[1] for res in results], axis=0)

y_pred = (y_pred_sum / num_batches) > 0.5
y_prob = y_pred_sum / num_batches
performance_dict(y_binary_test, y_pred, y_prob, bool_print=True, plot=False, copy_print=True)