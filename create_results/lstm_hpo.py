#!/usr/bin/env python3
import os
import time
import copy
import numpy as np
import pandas as pd
import optuna

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score, balanced_accuracy_score

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
N_TRIALS = 50  # Number of HPO trials to run for each model

# --- I/O Configuration ---
BASE_DATA_DIR = '/home/server/Projects/data/AKI/'
LSTM_INPUT_PKL = os.path.join(BASE_DATA_DIR, 'lstm_trainable.pkl')
MLP_INPUT_CSV = os.path.join(BASE_DATA_DIR, 'tabular_preop.csv')
RESULTS_FILE_PATH = os.path.join(BASE_DATA_DIR, 'results/hybrid_hpo_results.txt')

# --- Model HPO Toggles ---
hpo_configs = {
    'lstm_only': True,
    'hybrid': True,
}

# --- Search Space Definitions ---
# These define the range of hyperparameters Optuna will search over.
search_spaces = {
    'lstm_only': {
        'lr': (1e-5, 1e-2),
        'lstm_hidden_size': (16, 256),
        'lstm_num_layers': (1, 4),
        'dropout_rate': (0.1, 0.6)
    },
    'hybrid': {
        'lr': (1e-5, 1e-2),
        'lstm_hidden_size': (16, 128),
        'lstm_num_layers': (1, 3),
        'n_mlp_layers': (1, 4),
        'mlp_layer_size': (16, 256), # Range for individual MLP layer sizes
        'dropout_rate': (0.1, 0.6)
    }
}

# =============================================================================
# HYBRID MODEL DEFINITION (Copied from main training script)
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
            # Handle empty mlp_dims for lstm_only case
            if mlp_dims:
                for dim in mlp_dims:
                    mlp_layers.append(nn.Linear(in_features, dim))
                    mlp_layers.append(nn.ReLU())
                    mlp_layers.append(nn.Dropout(dropout_rate))
                    in_features = dim
                self.mlp = nn.Sequential(*mlp_layers)
        
        if self.mode == 'hybrid':
            # Guard against empty mlp_dims list
            classifier_input_size = lstm_hidden_size + (mlp_dims[-1] if mlp_dims else 0)
        elif self.mode == 'lstm_only':
            classifier_input_size = lstm_hidden_size
        elif self.mode == 'mlp_only':
            classifier_input_size = mlp_dims[-1]
        
        self.classifier = nn.Sequential(
            nn.Linear(classifier_input_size, (classifier_input_size // 2)),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear((classifier_input_size // 2), 1)
        )

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
# HPO HELPER FUNCTIONS
# =============================================================================

def check_search_space_boundaries(study, search_space):
    """Checks if the best parameters are at the boundaries of the search space."""
    warnings = []
    best_params = study.best_params

    for param, value in best_params.items():
        # Handle dynamic mlp layer sizes, which all use the 'mlp_layer_size' search space
        search_key = 'mlp_layer_size' if param.startswith('mlp_layer_') else param
        
        if search_key not in search_space:
            continue

        min_val, max_val = search_space[search_key]

        if isinstance(value, (int, float)) and (np.isclose(value, min_val) or np.isclose(value, max_val)):
            warnings.append(
                f"  - WARNING for {study.study_name}: Best value for '{param}' ({value}) is at the boundary "
                f"of its search space [{min_val}, {max_val}]. Consider expanding the range."
            )
    
    if warnings:
        print("\n" + "="*20 + " BOUNDARY WARNINGS " + "="*20)
        for warning in warnings:
            print(warning)
        print("="*63 + "\n")


def save_hpo_results_to_file(filepath, results_dict):
    """Formats and saves the final HPO results to a text file."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w') as f:
        f.write("# Hyperparameter Optimization Results for LSTM/Hybrid Models\n")
        f.write(f"# Generated on: {time.ctime()}\n\n")
        f.write("# --- COPY-PASTE A DICTIONARY BELOW INTO THE MAIN SCRIPT ---\n")
        
        for model_name, params in results_dict.items():
            f.write(f"\n# --- Best for {model_name.upper()} ---\n")
            f.write(f"hpo_params_{model_name} = {{\n")
            for key, value in params.items():
                if isinstance(value, list):
                     f.write(f"    '{key}': {value},\n")
                elif isinstance(value, float):
                     f.write(f"    '{key}': {value:.6f},\n")
                else:
                     f.write(f"    '{key}': {value},\n")
            f.write("}\n")
    print(f"\nFinal HPO results saved to: {filepath}")

def objective_builder(model_name, train_loader, val_loader, tabular_input_size, lstm_input_size, device):
    """Builds the Optuna objective function for a given model."""
    
    def objective(trial):
        # --- Suggest Hyperparameters ---
        lr = trial.suggest_float('lr', *search_spaces[model_name]['lr'], log=True)
        dropout_rate = trial.suggest_float('dropout_rate', *search_spaces[model_name]['dropout_rate'])
        
        # Model-specific architecture parameters
        if model_name == 'lstm_only':
            lstm_hidden_size = trial.suggest_int('lstm_hidden_size', *search_spaces[model_name]['lstm_hidden_size'])
            lstm_num_layers = trial.suggest_int('lstm_num_layers', *search_spaces[model_name]['lstm_num_layers'])
            mlp_dims = [] # No MLP for this model
        elif model_name == 'hybrid':
            lstm_hidden_size = trial.suggest_int('lstm_hidden_size', *search_spaces[model_name]['lstm_hidden_size'])
            lstm_num_layers = trial.suggest_int('lstm_num_layers', *search_spaces[model_name]['lstm_num_layers'])
            n_mlp_layers = trial.suggest_int('n_mlp_layers', *search_spaces[model_name]['n_mlp_layers'])
            
            mlp_dims = []
            for i in range(n_mlp_layers):
                layer_size = trial.suggest_int(f'mlp_layer_{i}_size', *search_spaces[model_name]['mlp_layer_size'])
                mlp_dims.append(layer_size)
        else:
            raise ValueError(f"Unsupported model_name for HPO: {model_name}")

        # --- Setup Model and Training ---
        model = HybridModel(
            tabular_input_size=tabular_input_size,
            lstm_input_size=lstm_input_size,
            lstm_hidden_size=lstm_hidden_size,
            lstm_num_layers=lstm_num_layers,
            mlp_dims=mlp_dims,
            dropout_rate=dropout_rate,
            mode=model_name
        ).to(device)

        optimizer = optim.Adam(model.parameters(), lr=lr)
        y_train_numpy = train_loader.dataset.tensors[3].numpy()
        pos_weight_val = np.sum(y_train_numpy == 0) / np.sum(y_train_numpy == 1) if np.sum(y_train_numpy == 1) > 0 else 1.0
        pos_weight = torch.tensor([pos_weight_val], device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        best_val_metric = 0 # Now represents balanced accuracy
        patience_counter = 0

        # --- Simplified Training Loop for HPO ---
        for epoch in range(150): # Run for a max of 150 epochs per trial
            model.train()
            for batch in train_loader:
                x_tab_batch, x_time_batch, seq_len_batch, y_batch = [b.to(device) for b in batch]
                optimizer.zero_grad()
                outputs = model(x_tab_batch, x_time_batch, seq_len_batch)
                loss = criterion(outputs, y_batch.unsqueeze(1))
                loss.backward()
                optimizer.step()

            # --- Validation and Early Stopping ---
            model.eval()
            all_val_probs = []
            with torch.no_grad():
                for batch in val_loader:
                    x_tab_batch, x_time_batch, seq_len_batch, _ = [b.to(device) for b in batch]
                    val_outputs = model(x_tab_batch, x_time_batch, seq_len_batch)
                    all_val_probs.append(torch.sigmoid(val_outputs).cpu())

            val_probs = torch.cat(all_val_probs).numpy().flatten()
            
            # --- Use Balanced Accuracy ---
            val_pred_binary = (val_probs >= 0.5).astype(int)
            current_val_metric = balanced_accuracy_score(val_loader.dataset.tensors[3].numpy(), val_pred_binary)

            if current_val_metric > best_val_metric:
                best_val_metric = current_val_metric
                patience_counter = 0
            else:
                patience_counter += 1
            
            # Pruning and early stopping
            trial.report(best_val_metric, epoch)
            if trial.should_prune() or patience_counter >= 15:
                raise optuna.exceptions.TrialPruned()

        return best_val_metric
    
    return objective

# =============================================================================
# MAIN EXECUTION BLOCK
# =============================================================================
def main():
    """Main function to run the HPO process."""
    torch.manual_seed(RANDOM_STATE)
    np.random.seed(RANDOM_STATE)
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- Load Data Sources ---
    try:
        df_lstm_source = pd.read_pickle(LSTM_INPUT_PKL)
    except FileNotFoundError:
        print(f"ERROR: LSTM data not found at {LSTM_INPUT_PKL}. Exiting.")
        return
        
    try:
        df_mlp_source = pd.read_csv(MLP_INPUT_CSV)
    except FileNotFoundError:
        print(f"ERROR: Tabular data not found at {MLP_INPUT_CSV}. Exiting.")
        return

    all_best_hpo_results = {}
    
    for model_name, should_run in hpo_configs.items():
        if not should_run:
            continue

        print(f"\n{'='*25} STARTING HPO FOR: {model_name.upper()} {'='*25}")
        
        # --- Prepare Data for the specific model ---
        if model_name == 'hybrid':
            df_lstm_subset = df_lstm_source[['op_id', 'time_tensors', 'seq_len', TARGET]]
            df_for_run = pd.merge(df_lstm_subset, df_mlp_source.drop(columns=[TARGET], errors='ignore'), on='op_id', how='inner')
        else: # lstm_only
            df_for_run = df_lstm_source
        
        feature_cols_tab = [col for col in df_mlp_source.columns if col not in ['op_id', TARGET]]

        def df_to_tensors(sub_df):
            # CORRECTED: Check if tabular columns exist before trying to create a tensor
            if feature_cols_tab and all(col in sub_df.columns for col in feature_cols_tab):
                X_tab = torch.tensor(sub_df[feature_cols_tab].values, dtype=torch.float32)
            else:
                X_tab = torch.empty(len(sub_df), 0)

            y = torch.tensor(sub_df[TARGET].values, dtype=torch.float32)
            
            # Time-series data is assumed to be present for both models in this HPO script
            X_time = torch.stack([t.clone().detach() for t in sub_df['time_tensors']]).to(torch.float32)
            seq_len = torch.tensor(sub_df['seq_len'].tolist(), dtype=torch.long)
            
            return X_tab, X_time, seq_len, y

        # --- Create a single train/validation split for the HPO process ---
        train_val_df, _ = train_test_split(df_for_run, test_size=0.2, random_state=RANDOM_STATE, stratify=df_for_run[TARGET])
        train_df, val_df = train_test_split(train_val_df, test_size=0.25, random_state=RANDOM_STATE, stratify=train_val_df[TARGET])

        scaler = StandardScaler()
        # CORRECTED: Only scale if the model is 'hybrid' and thus has the tabular columns
        if model_name == 'hybrid':
             train_df.loc[:, feature_cols_tab] = scaler.fit_transform(train_df[feature_cols_tab])
             val_df.loc[:, feature_cols_tab] = scaler.transform(val_df[feature_cols_tab])
        
        train_dataset = TensorDataset(*df_to_tensors(train_df))
        val_dataset = TensorDataset(*df_to_tensors(val_df))
        
        # MODIFIED: Batch size reduced to 1024 to prevent OOM errors
        train_loader = DataLoader(train_dataset, batch_size=512, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=512)

        # --- Run Optuna Study ---
        study = optuna.create_study(direction='maximize', study_name=f"{model_name}_hpo")
        objective_func = objective_builder(
            model_name, train_loader, val_loader, 
            tabular_input_size=len(feature_cols_tab) if model_name == 'hybrid' else 0,
            lstm_input_size=train_dataset.tensors[1].shape[2],
            device=device
        )
        study.optimize(objective_func, n_trials=N_TRIALS, show_progress_bar=True)

        print(f"Best trial for {model_name.upper()}: Balanced Accuracy = {study.best_value:.4f}")
        
        # --- Console printout and boundary check ---
        print("\n--- Best Parameters Found ---")
        best_params = study.best_params
        # Re-create mlp_dims for clean output if necessary
        if 'n_mlp_layers' in best_params:
            mlp_dims = [best_params[f'mlp_layer_{i}_size'] for i in range(best_params['n_mlp_layers'])]
            final_params = {'mlp_dims': mlp_dims}
            # Add other non-layer-specific params
            for key, val in best_params.items():
                if not key.startswith('mlp_layer_') and key != 'n_mlp_layers':
                    final_params[key] = val
            all_best_hpo_results[model_name] = final_params
        else:
             all_best_hpo_results[model_name] = best_params

        # Print final formatted parameters
        for key, value in all_best_hpo_results[model_name].items():
            print(f"  - {key}: {value}")
        
        check_search_space_boundaries(study, search_spaces[model_name])


    # --- Save Final Results ---
    if all_best_hpo_results:
        save_hpo_results_to_file(RESULTS_FILE_PATH, all_best_hpo_results)

if __name__ == "__main__":
    main()
