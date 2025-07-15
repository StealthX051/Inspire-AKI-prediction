#!/usr/bin/env python3
import os
import time
import copy
import numpy as np
import pandas as pd
from tqdm import tqdm

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch.utils.data import TensorDataset, DataLoader

# =============================================================================
# SCRIPT CONFIGURATION
# =============================================================================
TARGET = 'aki_boolean'
RANDOM_STATE = 42
USE_BOOTSTRAPPING = True # Set to True to run the full 25-iteration cross-validation
N_BOOTSTRAP_ITERATIONS = 25

# --- I/O Configuration ---
BASE_DATA_DIR = '/home/server/Projects/data/AKI/'
LSTM_INPUT_PKL = os.path.join(BASE_DATA_DIR, 'lstm_trainable.pkl')
MLP_INPUT_CSV = os.path.join(BASE_DATA_DIR, 'tabular_preop.csv')

# Main consolidated results file
RESULTS_PKL = os.path.join(BASE_DATA_DIR, 'results/lstm_hybrid_test_optimized.pkl')

# ADDED: Additional output files for specific models
INTRAOP_RESULTS_PKL = os.path.join(BASE_DATA_DIR, 'results/tabular_intraop_test.pkl')
COMBINED_RESULTS_PKL = os.path.join(BASE_DATA_DIR, 'results/tabular_combined_test.pkl')


# --- Model Run Toggles ---
model_configs = {
    'lstm_only': True,
    'mlp_only': False,
    'hybrid': True,
}

# =============================================================================
# HYPERPARAMETER CONFIGURATION
# =============================================================================

# --- Default Hyperparameters (Used if HPO dictionaries are empty) ---
default_hyperparameters = {
    'learning_rate': 0.001,
    'epochs': 1000,
    'batch_size': 1024,
    'patience': 20,
    'es_check_interval': 5,
    'lr_scheduler_patience': 7,
    'lr_scheduler_factor': 0.1,
    'gradient_clip_value': 1.0,
    # Default architecture
    'lstm_hidden_size': 128,
    'lstm_num_layers': 2,
    'mlp_dims': [256, 128, 64],
    'dropout_rate': 0.4
}

# --- HPO RESULTS: PASTE YOUR OPTIMIZED PARAMETERS HERE ---
# After running the HPO script, copy the output dictionary and paste it here.
# If a dictionary is left empty, the script will use the default_hyperparameters above for that model.

hpo_params_lstm_only = {
    'lr': 0.000015,
    'dropout_rate': 0.363015,
    'lstm_hidden_size': 50,
    'lstm_num_layers': 4,
}

hpo_params_hybrid = {
    'mlp_dims': [120, 155],
    'lr': 0.000017,
    'dropout_rate': 0.467438,
    'lstm_hidden_size': 66,
    'lstm_num_layers': 2,
}


# Set seed for reproducibility
torch.manual_seed(RANDOM_STATE)
np.random.seed(RANDOM_STATE)

# =============================================================================
# HELPER FUNCTIONS
# =============================================================================
def performance_dict(y_true, y_pred_binary, y_prob):
    """Calculates a comprehensive dictionary of performance metrics."""
    if len(np.unique(y_true)) < 2:
        tn, fp, fn, tp = (len(y_true), 0, 0, 0) if np.all(y_true==0) else (0,0,0,len(y_true))
    else:
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred_binary).ravel()
    return {
        'roc_auc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else 0.5,
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred_binary),
        'f1': f1_score(y_true, y_pred_binary, zero_division=0),
        'recall': recall_score(y_true, y_pred_binary, zero_division=0),
        'precision': precision_score(y_true, y_pred_binary, zero_division=0),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0.0,
        'y_true': y_true,
        'y_pred_binary': y_pred_binary,
        'y_prob': y_prob,
    }

def save_results(model_name, df_results, output_pkl):
    """Saves the results DataFrame to a pickle file."""
    os.makedirs(os.path.dirname(output_pkl), exist_ok=True)
    df_collapsed = pd.DataFrame({col: [df_results[col].values] for col in df_results.columns})
    df_collapsed['model_name'] = model_name
    
    if os.path.exists(output_pkl):
        try:
            df_output = pd.read_pickle(output_pkl)
            if not df_output.empty:
                df_output = df_output[df_output['model_name'] != model_name]
            df_output = pd.concat([df_output, df_collapsed], ignore_index=True)
        except (EOFError, FileNotFoundError):
            df_output = df_collapsed
    else:
        df_output = df_collapsed
    df_output.to_pickle(output_pkl)
    print(f"Results for '{model_name}' saved to {output_pkl}")

# =============================================================================
# DATA SPLITTING CLASS
# =============================================================================
class BootstrapSplitter:
    def __init__(self, df, use_bootstrapping=False, n_iterations=25):
        self.df = df
        self.use_bootstrapping = use_bootstrapping
        self.n_iterations = n_iterations if use_bootstrapping else 1
        self.i = 0
        self.i_df = 0
        self.df_fifths = []

    def __iter__(self):
        return self

    def __next__(self):
        if self.i >= self.n_iterations:
            raise StopIteration
        if self.use_bootstrapping:
            if self.i % 5 == 0:
                self.i_df = 0
                self.df_fifths = []
                df_remainder = self.df.copy()
                for remaining_fifths in range(5, 1, -1):
                    rest_df, fold_df = train_test_split(df_remainder, test_size=(1.0/remaining_fifths), random_state=RANDOM_STATE + (self.i // 5), stratify=df_remainder[TARGET])
                    self.df_fifths.append(fold_df)
                    df_remainder = rest_df
                self.df_fifths.append(df_remainder)
            test_df = self.df_fifths[self.i_df]
            train_dfs = [df for j, df in enumerate(self.df_fifths) if j != self.i_df]
            train_df = pd.concat(train_dfs)
            self.i_df += 1
        else:
            train_df, test_df = train_test_split(self.df, test_size=0.2, random_state=RANDOM_STATE, stratify=self.df[TARGET])
        self.i += 1
        return train_df, test_df

# =============================================================================
# UNIFIED PYTORCH MODEL
# =============================================================================
class HybridModel(nn.Module):
    def __init__(self, tabular_input_size, lstm_input_size, lstm_hidden_size, lstm_num_layers, mlp_dims, dropout_rate, mode='hybrid'):
        super(HybridModel, self).__init__()
        self.mode = mode
        if self.mode in ['lstm_only', 'hybrid']:
            lstm_dropout = dropout_rate if lstm_num_layers > 1 else 0
            self.lstm = nn.LSTM(
                input_size=lstm_input_size, 
                hidden_size=lstm_hidden_size, 
                num_layers=lstm_num_layers, 
                batch_first=True,
                dropout=lstm_dropout
            )
        if self.mode in ['mlp_only', 'hybrid']:
            mlp_layers = []
            in_features = tabular_input_size
            if mlp_dims:
                for dim in mlp_dims:
                    mlp_layers.append(nn.Linear(in_features, dim))
                    mlp_layers.append(nn.ReLU())
                    mlp_layers.append(nn.Dropout(dropout_rate))
                    in_features = dim
                self.mlp = nn.Sequential(*mlp_layers)
        
        if self.mode == 'hybrid':
            classifier_input_size = lstm_hidden_size + (mlp_dims[-1] if mlp_dims else 0)
        elif self.mode == 'lstm_only':
            classifier_input_size = lstm_hidden_size
        elif self.mode == 'mlp_only':
            classifier_input_size = mlp_dims[-1] if mlp_dims else 0
        
        # Add a guard for zero-sized inputs
        if classifier_input_size > 0:
            self.classifier = nn.Sequential(
                nn.Linear(classifier_input_size, (classifier_input_size // 2)),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear((classifier_input_size // 2), 1)
            )
        else:
            # Create a dummy layer if input size is 0 to avoid errors
            self.classifier = nn.Identity()


    def forward(self, x_tab=None, x_time=None, seq_len=None):
        if self.mode == 'hybrid':
            packed_input = pack_padded_sequence(x_time, seq_len.cpu(), batch_first=True, enforce_sorted=False)
            _, (hn, _) = self.lstm(packed_input)
            lstm_out = hn[-1]
            mlp_out = self.mlp(x_tab)
            combined = torch.cat((lstm_out, mlp_out), dim=1)
        elif self.mode == 'lstm_only':
            packed_input = pack_padded_sequence(x_time, seq_len.cpu(), batch_first=True, enforce_sorted=False)
            _, (hn, _) = self.lstm(packed_input)
            combined = hn[-1]
        elif self.mode == 'mlp_only':
            combined = self.mlp(x_tab)
        return self.classifier(combined)

# =============================================================================
# TRAINING & EVALUATION FUNCTIONS
# =============================================================================

def train_evaluate_lstm_hybrid(model, train_loader, val_loader, test_loader, h_params, device):
    """
    Training loop for LSTM and Hybrid models using mini-batching (DataLoader).
    """
    optimizer = optim.Adam(model.parameters(), lr=h_params['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=h_params['lr_scheduler_factor'], patience=h_params['lr_scheduler_patience'])
    
    y_train_full = train_loader.dataset.tensors[3].cpu().numpy()
    pos_weight_val = np.sum(y_train_full == 0) / np.sum(y_train_full == 1) if np.sum(y_train_full == 1) > 0 else 1.0
    pos_weight = torch.tensor([pos_weight_val], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_loss = float('inf')
    patience_counter = 0
    best_model_state = None

    for epoch in range(h_params['epochs']):
        model.train()
        for batch in train_loader:
            x_tab_batch, x_time_batch, seq_len_batch, y_batch = [b.to(device) for b in batch]
            optimizer.zero_grad()
            outputs = model(x_tab_batch, x_time_batch, seq_len_batch)
            loss = criterion(outputs, y_batch.unsqueeze(1))
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=h_params['gradient_clip_value'])
            optimizer.step()
        
        if (epoch + 1) % h_params['es_check_interval'] == 0:
            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    x_tab_batch, x_time_batch, seq_len_batch, y_batch = [b.to(device) for b in batch]
                    val_outputs = model(x_tab_batch, x_time_batch, seq_len_batch)
                    val_loss += criterion(val_outputs, y_batch.unsqueeze(1)).item()
            
            avg_val_loss = val_loss / len(val_loader)
            scheduler.step(avg_val_loss)
            
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
            
            if patience_counter >= h_params['patience']:
                print(f"  Stopping early at epoch {epoch + 1} due to no improvement.")
                break
    
    if best_model_state:
        model.load_state_dict(best_model_state)
    
    model.eval()
    all_probs, all_true = [], []
    with torch.no_grad():
        for batch in test_loader:
            x_tab_batch, x_time_batch, seq_len_batch, y_batch = [b.to(device) for b in batch]
            test_outputs = model(x_tab_batch, x_time_batch, seq_len_batch)
            all_probs.append(torch.sigmoid(test_outputs).cpu())
            all_true.append(y_batch.cpu())
            
    y_prob = torch.cat(all_probs).numpy().flatten()
    y_true = torch.cat(all_true).numpy()
    y_pred = (y_prob >= 0.5).astype(int)
    
    return performance_dict(y_true, y_pred, y_prob)

def train_evaluate_mlp(model, train_tensors, val_tensors, test_tensors, h_params, device):
    """
    A lightweight, fast training loop for the MLP-only model using full-batch updates.
    """
    X_train_tensor, _, _, y_train_tensor = train_tensors
    X_val_tensor, _, _, y_val_tensor = val_tensors
    X_test_tensor, _, _, y_test_tensor = test_tensors
    
    # Move all data to GPU at once
    X_train_tensor = X_train_tensor.to(device)
    y_train_tensor = y_train_tensor.to(device).unsqueeze(1)
    X_val_tensor = X_val_tensor.to(device)
    X_test_tensor = X_test_tensor.to(device)

    optimizer = optim.Adam(model.parameters(), lr=h_params['learning_rate'])
    pos_weight = torch.tensor([torch.sum(y_train_tensor == 0) / torch.sum(y_train_tensor == 1)], device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    best_val_auc, patience_counter, best_model_state = 0, 0, None
    
    for epoch in range(5000): 
        model.train()
        optimizer.zero_grad()
        outputs = model(X_train_tensor)
        loss = criterion(outputs, y_train_tensor)
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_probs = torch.sigmoid(model(X_val_tensor)).cpu().numpy().flatten()
            
            current_val_auc = roc_auc_score(y_val_tensor.cpu().numpy(), val_probs)
            if current_val_auc > best_val_auc:
                best_val_auc = current_val_auc
                patience_counter = 0
                best_model_state = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
            if patience_counter >= 20: 
                print(f"  Stopping early at epoch {epoch + 1} due to no improvement in validation AUC.")
                break
    
    if best_model_state:
        model.load_state_dict(best_model_state)

    model.eval()
    with torch.no_grad():
        y_prob = torch.sigmoid(model(X_test_tensor)).cpu().numpy().flatten()
    
    y_true = y_test_tensor.cpu().numpy()
    y_pred = (y_prob >= 0.5).astype(int)

    return performance_dict(y_true, y_pred, y_prob)


# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True
    print(f"Using device: {device}")

    # --- Load Data Sources ---
    try:
        print(f"Loading time-series data from {LSTM_INPUT_PKL}...")
        df_lstm_source = pd.read_pickle(LSTM_INPUT_PKL)
        num_ts_features = df_lstm_source['time_tensors'].iloc[0].shape[1]
        print(f"--> Found {num_ts_features} time-series features.")
    except FileNotFoundError:
        print(f"ERROR: Time-series data file not found at {LSTM_INPUT_PKL}. Exiting.")
        return
        
    try:
        print(f"Loading tabular data from {MLP_INPUT_CSV}...")
        df_mlp_source = pd.read_csv(MLP_INPUT_CSV)
    except FileNotFoundError:
        print(f"ERROR: Tabular data file not found at {MLP_INPUT_CSV}. Exiting.")
        return
    
    base_results_saved = False # Flag to ensure base model is only saved once

    # --- Main Loop for Models ---
    for model_name, should_run in model_configs.items():
        if not should_run:
            print(f"\nSkipping {model_name.upper()} model.")
            continue
        
        # ADDED: try-except block for tmux reliability
        try:
            print(f"\n{'='*25} RUNNING MODEL: {model_name.upper()} {'='*25}")

            # --- Select Hyperparameters for the current model ---
            h_params = default_hyperparameters.copy()
            if model_name == 'lstm_only' and hpo_params_lstm_only:
                h_params.update(hpo_params_lstm_only)
                print("--> Using HPO parameters for LSTM_ONLY model.")
            elif model_name == 'hybrid' and hpo_params_hybrid:
                h_params.update(hpo_params_hybrid)
                print("--> Using HPO parameters for HYBRID model.")
            else:
                print("--> Using DEFAULT parameters.")


            # --- Data Preparation based on Model Type ---
            if model_name == 'mlp_only':
                df_for_run = df_mlp_source
                feature_cols_tab = [col for col in df_for_run.columns if col not in ['op_id', TARGET]]
            else:
                df_lstm_subset = df_lstm_source[['op_id', 'time_tensors', 'seq_len', TARGET]]
                df_for_run = pd.merge(df_lstm_subset, df_mlp_source.drop(columns=[TARGET], errors='ignore'), on='op_id', how='inner')
                feature_cols_tab = [col for col in df_mlp_source.columns if col not in ['op_id', TARGET]]
                print(f"For {model_name}, found {len(df_for_run)} patients common to both data sources.")

            if df_for_run.empty:
                print(f"--> ERROR: No data available for model '{model_name}'. Check for matching 'op_id's if merging. Skipping.")
                continue

            def df_to_tensors(sub_df):
                X_tab = torch.tensor(sub_df[feature_cols_tab].values, dtype=torch.float32)
                y = torch.tensor(sub_df[TARGET].values, dtype=torch.float32)
                if 'time_tensors' in sub_df.columns:
                    X_time = torch.stack([t.clone().detach() for t in sub_df['time_tensors']]).to(torch.float32)
                    seq_len = torch.tensor(sub_df['seq_len'].tolist(), dtype=torch.long)
                else:
                    n_samples = len(sub_df)
                    X_time = torch.zeros(n_samples, 1, 1, dtype=torch.float32)
                    seq_len = torch.zeros(n_samples, dtype=torch.long)
                return X_tab, X_time, seq_len, y

            df_results = pd.DataFrame()
            splitter = BootstrapSplitter(df_for_run, use_bootstrapping=USE_BOOTSTRAPPING, n_iterations=N_BOOTSTRAP_ITERATIONS)
            
            for i, (train_df, test_df) in enumerate(splitter, 1):
                print(f"--- Starting Run {i}/{splitter.n_iterations} for {model_name.upper()} ---")
                start_time = time.time()
                
                train_df, val_df = train_test_split(train_df, test_size=0.15, random_state=RANDOM_STATE + i, stratify=train_df[TARGET])

                scaler = StandardScaler()
                train_df.loc[:, feature_cols_tab] = scaler.fit_transform(train_df[feature_cols_tab])
                val_df.loc[:, feature_cols_tab] = scaler.transform(val_df[feature_cols_tab])
                test_df.loc[:, feature_cols_tab] = scaler.transform(test_df[feature_cols_tab])

                # --- Model Initialization ---
                train_tensors = df_to_tensors(train_df)
                val_tensors = df_to_tensors(val_df)
                test_tensors = df_to_tensors(test_df)

                lstm_input_size = train_tensors[1].shape[2] if train_tensors[1].numel() > 1 else 0
                
                model = HybridModel(
                    tabular_input_size=len(feature_cols_tab),
                    lstm_input_size=lstm_input_size,
                    lstm_hidden_size=h_params['lstm_hidden_size'],
                    lstm_num_layers=h_params['lstm_num_layers'],
                    mlp_dims=h_params.get('mlp_dims', []), # Use .get for safety
                    dropout_rate=h_params['dropout_rate'],
                    mode=model_name
                ).to(device)

                # --- Select appropriate training function ---
                if model_name == 'mlp_only':
                    perf = train_evaluate_mlp(model, train_tensors, val_tensors, test_tensors, h_params, device)
                else:
                    train_dataset = TensorDataset(*train_tensors)
                    val_dataset = TensorDataset(*val_tensors)
                    test_dataset = TensorDataset(*test_tensors)
                    
                    train_loader = DataLoader(train_dataset, batch_size=h_params['batch_size'], shuffle=True, pin_memory=True, num_workers=4)
                    val_loader = DataLoader(val_dataset, batch_size=h_params['batch_size'], shuffle=False, pin_memory=True, num_workers=4)
                    test_loader = DataLoader(test_dataset, batch_size=h_params['batch_size'], shuffle=False, pin_memory=True, num_workers=4)
                    
                    perf = train_evaluate_lstm_hybrid(model, train_loader, val_loader, test_loader, h_params, device)

                df_results = pd.concat([df_results, pd.DataFrame([perf])], ignore_index=True)
                
                end_time = time.time()
                print(f"--- Run {i} Finished in {end_time - start_time:.2f} seconds ---")
                print(f"    AUROC: {perf['roc_auc']:.4f}, F1: {perf['f1']:.4f}, Recall: {perf['recall']:.4f}, Precision: {perf['precision']:.4f}\n")

            if not df_results.empty:
                # Save to the main consolidated results file
                save_results(f"lstm_{model_name}", df_results, RESULTS_PKL)

                # Save results to additional files for specific models
                if model_name == 'lstm_only':
                    save_results('lstm', df_results, INTRAOP_RESULTS_PKL)
                elif model_name == 'hybrid':
                    save_results('hybrid', df_results, COMBINED_RESULTS_PKL)

                if len(df_results) > 1:
                    print(f"--- Overall {model_name.upper()} Performance Summary (Mean +/- Std Dev) ---")
                    for metric in ['roc_auc', 'f1', 'recall', 'precision']:
                        mean_val = df_results[metric].mean()
                        std_val = df_results[metric].std()
                        print(f"  {metric.capitalize()}: {mean_val:.4f} +/- {std_val:.4f}")

            # --- Save base model results after the first model run ---
            if not base_results_saved and not df_results.empty:
                print("\n--- Creating and saving 'base_54k' ground truth model ---")
                # Create a DataFrame in the format expected by the plotting script
                # It stores the true labels from the test set in the 'y_pred_binary' column
                base_df = pd.DataFrame({'y_pred_binary': df_results['y_true']})
                
                # Save this base model to all three results files
                save_results('base_54k', base_df, RESULTS_PKL)
                save_results('base_54k', base_df, INTRAOP_RESULTS_PKL)
                save_results('base_54k', base_df, COMBINED_RESULTS_PKL)
                
                base_results_saved = True # Set flag to prevent this from running again
        
        except Exception as e:
            print(f"\n{'!'*25} ERROR ENCOUNTERED {'!'*25}")
            print(f"An error occurred while running model: {model_name}")
            print(f"Error details: {e}")
            print(f"Skipping to next model.")
            print(f"{'!'*65}\n")


    print("\n\nScript finished.")

if __name__ == "__main__":
    main()
