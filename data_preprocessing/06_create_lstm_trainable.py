import numpy as np
import pandas as pd
from tqdm import tqdm
from collections import Counter
from sklearn.preprocessing import StandardScaler
import torch
import random
from torch.nn.utils.rnn import pack_padded_sequence
import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

tabular_csv = '/home/server/Projects/data/base/tabular_combined.csv'
df_tabular = pd.read_csv(tabular_csv)

time_csv = '/home/server/Projects/data/Multiple-Outcomes/time_series_cleaned.csv'
df_time = pd.read_csv(time_csv)

output_file = '/home/server/Projects/data/Multiple-Outcomes/lstm_trainable.pkl'

# drops operations that are missing over half of all features
feature_mask = df_time.drop(columns=['op_id', 'chart_time']).notna().astype(int)
df_time['presence'] = feature_mask.sum(axis=1) / 24
# drops about 1/6 data
df_time = df_time[df_time['presence'] > (1/4)].drop(columns=['presence']) 
mask_flag_value = 0
df_time = df_time.fillna(mask_flag_value)
bool_cols = df_tabular.select_dtypes(include='bool').columns
df_tabular[bool_cols] = df_tabular[bool_cols].astype(float)


# pivot and pad, re-df
padded_tensors = []
pad_length = 200
op_ids = []
sequence_lengths = []
# for op_id, group in tqdm(df_time.groupby("op_id"), desc="grouping by op_ids"):
#     mat = torch.tensor(group.drop(columns=['op_id', 'chart_time']).values)
#     if mat.shape[0] < 200: # throws away about 4% of longest operations bc they would extend max pad length from 200 to like 600. Consider including all. 
#         padded_tensors.append(torch.nn.functional.pad(mat, pad=(0, 0, 0, pad_length - mat.shape[0]), value=0))
#         op_ids.append(op_id)
#         sequence_lengths.append(mat.shape[0])

for op_id, group in tqdm(df_time.groupby("op_id"), desc="grouping by op_ids"):
    mat = torch.tensor(
        group.drop(columns=["op_id", "chart_time"]).to_numpy(),
        dtype=torch.float32
    )
    padded_tensors.append(mat)          # now holds variable-length tensors
    op_ids.append(op_id)
    sequence_lengths.append(mat.shape[0])

df_time = pd.DataFrame({
    'op_id': op_ids,
    'time_tensors': padded_tensors,
    'seq_len': sequence_lengths
    })
df_combined = df_time.merge(df_tabular, on='op_id', how='inner')

df_combined.to_pickle(output_file)

# # shuffle, and then set aside first fifth for test
# df_combined = df_combined.sample(frac=1, random_state=42).reset_index(drop=True)
# test_idx_start = 0
# test_idx_end = len(df_combined) // 10
# df_test = df_combined.iloc[test_idx_start : test_idx_end]
# df_train = df_combined.iloc[test_idx_end :]

# # upsample the minority class to approximately match the count of the majority class
# df_train_majority = df_train[df_train['aki'] < 0.3]
# df_train_minority = df_train[df_train['aki'] >= 0.3]
# majority_count = df_train_majority.shape[0]
# minority_count = df_train_minority.shape[0]
# upsample_ratio = 2 * majority_count // minority_count
# df_train_minority = pd.concat([df_train_minority] * upsample_ratio, ignore_index=True)
# df_train = pd.concat([df_train_minority, df_train_majority], ignore_index=True)
# df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True) #reshuffle
# df_train_file = '/home/server/Projects/data/AKI/lstm/df_train.pkl'
# df_test_file = '/home/server/Projects/data/AKI/lstm/df_test.pkl'
# df_train.to_pickle(df_train_file)
# df_test.to_pickle(df_test_file)