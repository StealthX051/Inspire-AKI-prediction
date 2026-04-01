import os
import numpy as np
import scipy.stats as stats
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import pandas as pd
from utils import performance_dict, sigmoid
from tqdm import tqdm

# TARGET  = 'aki', 'macce', etc.

TARGET = 'aki_boolean'
TARGET_IS_BOOLEAN = True

file = '/home/server/Projects/data/AKI/tabular_preop.csv'
df = pd.read_csv(file)
output_pkl = '/home/server/Projects/data/AKI/results/tabular_preop_test.pkl'


# file = '/home/server/Projects/data/AKI/tabular_combined_old.csv'
# df = pd.read_csv(file)

class bootstrap_split(object):
    def __init__(self, df, target=TARGET, target_is_boolean=TARGET_IS_BOOLEAN, upsample_bool=True):
        self.df = df
        self.i_df = 0   # cycling index out of five
        self.i = 0      # total index out of 25
        self.df_fifths = []
        self.target = target
        self.upsample_bool = upsample_bool
        if target_is_boolean:
            self.target_boolean = target
        else:
            self.target_boolean = f"{self.target}_boolean"
        self.target_positive = f"{self.target}_positive"
        if target not in df.columns:
            print(f"Target {target} not in dataframe columns")

    def __iter__(self):
        return self
    def __next__(self):
        return self.next()
    
    def df_to_arrays(self, df):
        remove_cols = ['op_id', self.target, self.target_boolean, self.target_positive]
        # remove_cols += [col for col in df.columns if '_isna' in col]
        
        X = df.drop(columns=remove_cols, errors='ignore').values
        y = df[self.target].values
        y_binary = df[self.target_boolean].values
        return X, y, y_binary
    
    def upsample(self, df):
        if self.upsample_bool == False:
            return df.copy()
        df_majority = df[df[self.target_boolean] == 0]
        df_minority = df[df[self.target_boolean] == 1]

        df_minority_upsampled = resample(df_minority,
                    replace=True,  # Sample with replacement
                    n_samples=len(df_majority),  # Match majority class size
                    random_state=42)

        # Combine upsampled minority with majority class
        df_balanced = pd.concat([df_majority, df_minority_upsampled])
        df_balanced = df_balanced.sample(frac=1, random_state=42).reset_index(drop=True)
        return df_balanced

    def next(self):
        if self.i == 25:
            raise StopIteration()
        elif self.i % 5 == 0:
            self.i_df = 0
            self.df_fifths = [] 
            df_remainder = self.df
            for remaining_fifths in range(5, 1, -1):
                df_remainder, df_fifth = train_test_split(df_remainder, 
                                            test_size=(1.0/remaining_fifths), 
                                            random_state=42 + (self.i // 5), 
                                            stratify=df_remainder[self.target_boolean])
                self.df_fifths.append(df_fifth)
            self.df_fifths.append(df_remainder)
        df_temp = self.df_fifths.pop(self.i_df)
        X_test, y_test, y_binary_test = self.df_to_arrays(df_temp)
        X_train, y_train, y_binary_train = self.df_to_arrays(self.upsample(pd.concat(self.df_fifths)))
        self.df_fifths.insert(self.i_df, df_temp)
        self.i_df += 1
        self.i += 1
        return (X_test, y_test, y_binary_test), (X_train, y_train, y_binary_train)

def print_confidence_intervals(df_results):
    if 'y_pred_binary' in df_results.columns:
        df_results = df_results.drop(columns=['y_pred_binary'])
    if 'y_prob' in df_results.columns:
        df_results = df_results.drop(columns=['y_prob'])
    lows = []
    means = []
    highs = []
    for col in df_results.columns:
        mean = np.mean(df_results[col].values)
        sem = stats.sem(df_results[col].values)  # Standard error of the mean

        # Get 95% confidence interval
        confidence = 0.95
        n = len(df_results[col].values)
        dof = n - 1  # Degrees of freedom
        ci = stats.t.interval(confidence, dof, loc=mean, scale=sem)
        lows.append(ci[0])
        means.append(mean)
        highs.append(ci[1])
    for col in df_results.columns:
        print(col)
    print()
    for arr in [lows, means, highs]:
        for num in arr:
            print(f"{num:.4f}")
        print()

def save_results(model_name, df_results, output_pkl):
    # make into one row
    df_collapsed = pd.DataFrame({col: [df_results[col].values] for col in df_results.columns})
    df_collapsed['model_name'] = model_name

    if os.path.exists(output_pkl):
        df_output = pd.read_pickle(output_pkl)
        if df_output.empty:
            df_output = df_collapsed
        else:
            df_output = df_output[df_output['model_name'] != model_name]
        df_output = pd.concat([df_output, df_collapsed], ignore_index=True)
    else:
        df_output = df_collapsed
    df_output.to_pickle(output_pkl)

# -------------------- Record Correct Answers --------------------
df_results = pd.DataFrame()
for test, train in tqdm(bootstrap_split(df)):
    X_test, y_test, y_binary_test = test
    X_train, y_train, y_binary_train = train

    output_dict = performance_dict(y_binary_test, y_binary_test, y_binary_test)


    output_dict['y_pred_binary'] = y_binary_test
    output_dict['y_prob'] = sigmoid(y_test - 0.3)
    if df_results.empty:
        df_results = pd.DataFrame(columns=output_dict.keys())
    df_results.loc[len(df_results)] = output_dict

save_results('base', df_results, output_pkl)

# -------------------- LOGISTIC REGRESSION BASED CLASSIFICATION --------------------
from sklearn.linear_model import LogisticRegression

df_results = pd.DataFrame()
for test, train in tqdm(bootstrap_split(df)):
    X_test, y_test, y_binary_test = test
    X_train, y_train, y_binary_train = train




    model = LogisticRegression(max_iter=10000)
    model.fit(X_train, y_binary_train)

    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    output_dict = performance_dict(y_binary_test, y_pred, y_prob)


    output_dict['y_pred_binary'] = y_pred
    output_dict['y_prob'] = y_prob
    if df_results.empty:
        df_results = pd.DataFrame(columns=output_dict.keys())
    df_results.loc[len(df_results)] = output_dict

save_results('log_reg', df_results, output_pkl)
    
eval_metric = 'balanced_accuracy'
time_in_min = 5

# -------------------- AutoGluon Classification --------------------

from autogluon.tabular import TabularPredictor

df_results = pd.DataFrame()
for test, train in tqdm(bootstrap_split(df, upsample_bool=False)):
    X_test, y_test, y_binary_test = test
    X_train, y_train, y_binary_train = train




    train_df = pd.DataFrame(X_train)
    train_df["target"] = y_binary_train
    test_df = pd.DataFrame(X_test)
    test_df["target"] = y_binary_test

    train_df["sample_weight"] = (y_binary_train.astype(int) * 17 * 1) + 1

    predictor = TabularPredictor(
            label="target",  # Target column for prediction
            problem_type="binary",
            eval_metric=eval_metric,  # Evaluation metric for the model
            sample_weight = "sample_weight"
            ).fit(train_df, presets="best_quality", time_limit=time_in_min * 60)

    

    y_pred_binary = predictor.predict(test_df.drop(columns=["target"])).values
    y_prob = predictor.predict_proba(test_df.drop(columns=["target"]))[True].values
    output_dict = performance_dict(y_binary_test, y_pred_binary, y_prob)
    break


    # output_dict['y_pred_binary'] = y_pred_binary
    # output_dict['y_prob'] = y_prob
    # if df_results.empty:
    #     df_results = pd.DataFrame(columns=output_dict.keys())
    # df_results.loc[len(df_results)] = output_dict

# save_results('autogluon_', df_results, output_pkl)
# print_dict(output_dict)

# -------------------- AutoGluon Classification --------------------

from autogluon.tabular import TabularPredictor

df_results = pd.DataFrame()
for test, train in tqdm(bootstrap_split(df)):
    X_test, y_test, y_binary_test = test
    X_train, y_train, y_binary_train = train




    train_df = pd.DataFrame(X_train)
    train_df["target"] = y_binary_train
    test_df = pd.DataFrame(X_test)
    test_df["target"] = y_binary_test

    predictor = TabularPredictor(
            label="target",  # Target column for prediction
            problem_type="binary"
            ).fit(train_df, presets="best_quality", time_limit=2 * 60)

    

    y_pred_binary = predictor.predict(test_df.drop(columns=["target"])).values
    y_prob = predictor.predict_proba(test_df.drop(columns=["target"]))[True].values
    output_dict = performance_dict(y_binary_test, y_pred_binary, y_prob)



    output_dict['y_pred_binary'] = y_pred_binary
    output_dict['y_prob'] = y_prob
    if df_results.empty:
        df_results = pd.DataFrame(columns=output_dict.keys())
    df_results.loc[len(df_results)] = output_dict

save_results('autogluon', df_results, output_pkl)
    

    # -------------------- XGBoost Classification --------------------

import xgboost as xgb

df_results = pd.DataFrame()
for test, train in tqdm(bootstrap_split(df)):
    X_test, y_test, y_binary_test = test
    X_train, y_train, y_binary_train = train



    model = xgb.XGBClassifier(
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
    
    model.fit(X_train, y_binary_train)
    y_pred_binary = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    output_dict = performance_dict(y_binary_test, y_pred_binary, y_prob)


    output_dict['y_pred_binary'] = y_pred_binary
    output_dict['y_prob'] = y_prob
    if df_results.empty:
        df_results = pd.DataFrame(columns=output_dict.keys())
    df_results.loc[len(df_results)] = output_dict

save_results('xgb', df_results, output_pkl)
    

    # -------------------- SVM VOTING ENSEMBLE CLASSIFICATION --------------------
from joblib import Parallel, delayed
from sklearn.svm import SVC

df_results = pd.DataFrame()
for test, train in tqdm(bootstrap_split(df)):
    X_test, y_test, y_binary_test = test
    X_train, y_train, y_binary_train = train


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
    output_dict = performance_dict(y_binary_test, y_pred, y_prob)


    output_dict['y_pred_binary'] = y_pred_binary
    output_dict['y_prob'] = y_prob
    if df_results.empty:
        df_results = pd.DataFrame(columns=output_dict.keys())
    df_results.loc[len(df_results)] = output_dict

save_results('svm', df_results, output_pkl)
    
    # -------------------- MLP CLASSIFICATION --------------------

from sklearn.neural_network import MLPClassifier

df_results = pd.DataFrame()
for test, train in tqdm(bootstrap_split(df)):
    X_test, y_test, y_binary_test = test
    X_train, y_train, y_binary_train = train


    arch = (8, 8, 4, 4)

    model = MLPClassifier(random_state=42, max_iter=1000, hidden_layer_sizes=arch)
    model.fit(X_train, y_binary_train)
    y_pred_binary = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    output_dict = performance_dict(y_binary_test, y_pred_binary, y_prob)


    output_dict['y_pred_binary'] = y_pred_binary
    output_dict['y_prob'] = y_prob
    if df_results.empty:
        df_results = pd.DataFrame(columns=output_dict.keys())
    df_results.loc[len(df_results)] = output_dict

save_results('mlp', df_results, output_pkl)
    
    # -------------------- RANDOM FOREST CLASSIFICATION --------------------

from sklearn.ensemble import RandomForestClassifier

df_results = pd.DataFrame()
for test, train in tqdm(bootstrap_split(df)):
    X_test, y_test, y_binary_test = test
    X_train, y_train, y_binary_train = train



    model = RandomForestClassifier()
    model.fit(X_train, y_binary_train)
    y_pred_binary = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    output_dict = performance_dict(y_binary_test, y_pred_binary, y_prob)


    output_dict['y_pred_binary'] = y_pred_binary
    output_dict['y_prob'] = y_prob
    if df_results.empty:
        df_results = pd.DataFrame(columns=output_dict.keys())
    df_results.loc[len(df_results)] = output_dict

save_results('rf', df_results, output_pkl)
    
    # -------------------- K-NEAREST NEIGHBORS CLASSIFICATION --------------------
from sklearn.neighbors import KNeighborsClassifier

df_results = pd.DataFrame()
for test, train in tqdm(bootstrap_split(df)):
    X_test, y_test, y_binary_test = test
    X_train, y_train, y_binary_train = train


    model = KNeighborsClassifier(n_neighbors=200)
    model.fit(X_train, y_binary_train)
    y_pred_binary = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    output_dict = performance_dict(y_binary_test, y_pred_binary, y_prob)


    output_dict['y_pred_binary'] = y_pred_binary
    output_dict['y_prob'] = y_prob
    if df_results.empty:
        df_results = pd.DataFrame(columns=output_dict.keys())
    df_results.loc[len(df_results)] = output_dict

save_results('knn', df_results, output_pkl)
    
    # -------------------- ASA CLASSIFICATION --------------------

df_results = pd.DataFrame()
for test, train in tqdm(bootstrap_split(df)):
    X_test, y_test, y_binary_test = test
    X_train, y_train, y_binary_train = train


    ASA_THRESHOLD = 4
    asa_idx = [len(np.unique(X_test[:, i])) for i in range(X_test.shape[1])].index(5)
    X_asa = X_test[:, asa_idx]
    
    # y_binary_test =     df[f"{TARGET}_boolean"].values
    # y_pred_binary =            (df['asa'] >= ASA_THRESHOLD).values
    # y_prob =            (df['asa'] / 6).values

    y_pred_binary =         (X_asa >= ASA_THRESHOLD).astype(float)
    y_prob =                (X_asa / 6).astype(float)

    output_dict = performance_dict(y_binary_test, y_pred_binary, y_prob)


    output_dict['y_pred_binary'] = y_pred_binary
    output_dict['y_prob'] = y_prob
    if df_results.empty:
        df_results = pd.DataFrame(columns=output_dict.keys())
    df_results.loc[len(df_results)] = output_dict

save_results('asa', df_results, output_pkl)
    