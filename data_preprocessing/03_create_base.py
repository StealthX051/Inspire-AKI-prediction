## TODO: Think about our missing data imputation more deeply, specifically the sentinel value of -99. 
# Currently we normalize before KNN/sentinel imputation, which is probably methadologically unsound. Consider readding the boolean missing values. 

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer
import os
from pathlib import Path
import sys

def main():
    """
    Main function to execute the data preprocessing pipeline.
    """
    # =============================================================================
    # Configuration and Paths
    # =============================================================================
    print("--- 1. Configuring Paths ---")
    
    # Define base directories
    # Using Path for cross-platform compatibility
    INPUT_DATA_DIR = Path("/home/server/Projects/data/Multiple-Outcomes/")
    OUTPUT_DATA_DIR = Path("/home/server/Projects/data/base/")

    # Create the output directory if it doesn't exist to prevent errors
    try:
        os.makedirs(OUTPUT_DATA_DIR, exist_ok=True)
        print(f"Output directory '{OUTPUT_DATA_DIR}' is ready.")
    except OSError as e:
        print(f"Error: Could not create directory {OUTPUT_DATA_DIR}. {e}", file=sys.stderr)
        sys.exit(1)

    # Define full file paths
    preop_csv_path = INPUT_DATA_DIR / "preop_data.csv"
    intraop_csv_path = INPUT_DATA_DIR / "feature_engineered.csv"
    
    norm_csv_path = OUTPUT_DATA_DIR / "normalization_stats.csv"
    combined_csv_path = OUTPUT_DATA_DIR / "tabular_combined.csv"
    preop_out_path = OUTPUT_DATA_DIR / "tabular_preop.csv"
    intraop_out_path = OUTPUT_DATA_DIR / "tabular_intraop.csv"

    # =============================================================================
    # Load and Merge Data
    # =============================================================================
    print("\n--- 2. Loading and Merging Data ---")
    try:
        print(f"Loading pre-operative data from: {preop_csv_path}")
        df_preop = pd.read_csv(preop_csv_path)
        
        print(f"Loading intra-operative data from: {intraop_csv_path}")
        df_intraop = pd.read_csv(intraop_csv_path)

        # Merge based on the corrected understanding of the data identifiers
        # This assumes 'subject_id' in preop maps to 'op_id' in intraop for a given procedure
        # If the relationship is different, this merge key needs to be adjusted.
        # NOTE: Based on our prior conversation, this merge is likely incorrect if one subject
        # can have multiple operations. A linking file is needed. For now, this addresses the KeyError.
        if 'op_id' not in df_preop.columns and 'subject_id' in df_preop.columns:
             print("Merging with left_on='subject_id' and right_on='op_id'")
             df = pd.merge(df_preop, df_intraop, left_on='subject_id', right_on='op_id', how='inner')
        else:
             print("Merging on 'op_id'")
             df = pd.merge(df_preop, df_intraop, on='op_id', how='inner')

        if df.empty:
            print("Warning: DataFrame is empty after merging. Check merge keys and input files.", file=sys.stderr)
            sys.exit(1)

    except FileNotFoundError as e:
        print(f"Error: Input file not found. {e}", file=sys.stderr)
        sys.exit(1)
    except KeyError as e:
        print(f"Error: Merge key not found in one of the DataFrames. {e}", file=sys.stderr)
        print("Please ensure 'op_id' or 'subject_id' exists in the respective files.", file=sys.stderr)
        sys.exit(1)

    # =============================================================================
    # Initial Data Cleaning
    # =============================================================================
    print("\n--- 3. Cleaning Data ---")
    
    # Replace infinite values with NaN for consistent missing value handling
    df.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Define columns to remove (potential data leaks or identifiers)
    cols_to_pop = [
        "postop_creatinine", "subject_id", "opstart_time", 
        "opend_time", "inhosp_death_time", "allcause_death_time"
    ]
    
    # More robustly find and remove columns that actually exist in the DataFrame
    existing_cols_to_pop = [col for col in cols_to_pop if col in df.columns]
    df.drop(columns=existing_cols_to_pop, inplace=True)
    print(f"Removed columns: {existing_cols_to_pop}")

    # =============================================================================
    # Outlier and Feature Type Handling
    # =============================================================================
    print("\n--- 4. Handling Outliers and Feature Types ---")
    
    # Convert integer columns to float to prevent type errors during calculations
    int_columns = df.select_dtypes(include=['int']).columns
    df[int_columns] = df[int_columns].astype(float)
    
    # Set a seed for reproducibility of random outlier replacement
    np.random.seed(42)

    print("beginnign dbugging")
    # Define columns to ignore during outlier handling and normalization
    ignore_cols = ['op_id', 'age', 'emop', 'num_card_events', 'antype', 'sex', 'asa']
    # Dynamically add other columns to ignore based on naming patterns
    for col in df.columns:
        if ('aki' in col):
            print("AKI BEING IGNORED")
        if ('department' in col) or ('_isna' in col) or ('aki' in col):
            ignore_cols.append(col)
    
    print("complete")
    exit(0)
    # Use a set for faster lookups
    ignore_cols = set(ignore_cols) 

    # Replace outliers in numerical columns not in the ignore list
    for col in df.columns:
        if col not in ignore_cols and pd.api.types.is_numeric_dtype(df[col]):
            lower_1 = df[col].quantile(0.01)
            upper_1 = df[col].quantile(0.99)
            
            # Continue only if the quantiles are valid numbers
            if pd.notna(lower_1) and pd.notna(upper_1):
                lower_05 = df[col].quantile(0.005)
                lower_5 = df[col].quantile(0.05)
                upper_95 = df[col].quantile(0.95)
                upper_995 = df[col].quantile(0.995)

                mask_lower = df[col] < lower_1
                mask_upper = df[col] > upper_1
                
                # Replace with random values from a plausible range
                df.loc[mask_lower, col] = np.random.uniform(lower_05, lower_5, size=mask_lower.sum())
                df.loc[mask_upper, col] = np.random.uniform(upper_95, upper_995, size=mask_upper.sum())

    # =============================================================================
    # Normalization
    # =============================================================================
    print("\n--- 5. Normalizing Numerical Features ---")
    
    # Identify columns to normalize
    cols_to_norm = [col for col in df.columns if col not in ignore_cols and pd.api.types.is_numeric_dtype(df[col])]
    
    scaler = StandardScaler()
    
    # Fit the scaler and store normalization statistics
    scaler.fit(df[cols_to_norm])
    df_stats = pd.DataFrame({'mean': scaler.mean_, 'var': scaler.var_}, index=cols_to_norm)

    # Apply the normalization
    df[cols_to_norm] = scaler.transform(df[cols_to_norm])
    print(f"Normalized {len(cols_to_norm)} columns.")

    # =============================================================================
    # Missing Value Imputation
    # =============================================================================
    print("\n--- 6. Imputing Missing Values ---")
    
    nan_percentage = (df.isna().mean() * 100)
    
    # Strategy 1: For high-missingness columns, fill with a sentinel value (-99)
    high_missing_cols = nan_percentage[nan_percentage >= 10].index.tolist()
    print(f"{len(high_missing_cols)} columns to be flagged with -99.")
    df[high_missing_cols] = df[high_missing_cols].fillna(-99)
    
    # Strategy 2: For low-missingness columns, use KNNImputer
    # Note: KNNImputer can be slow on large datasets.
    low_missing_cols = nan_percentage[(nan_percentage > 0) & (nan_percentage < 10)].index.tolist()
    print(f"{len(low_missing_cols)} columns to be imputed with KNNImputer.")
    if low_missing_cols:
        imputer = KNNImputer(n_neighbors=5)
        # Impute only on the subset of columns for efficiency
        df[low_missing_cols] = imputer.fit_transform(df[low_missing_cols])

    # =============================================================================
    # Save Outputs
    # =============================================================================
    print("\n--- 7. Saving Processed Data ---")
    try:
        # Identify original preop and intraop columns for separate file saving
        preop_cols_final = [col for col in df.columns if col in df_preop.columns and col != 'subject_id']
        intraop_cols_final = [col for col in df.columns if col in df_intraop.columns]
        
        # Add op_id to both for potential future merges
        if 'op_id' not in preop_cols_final: preop_cols_final.insert(0, 'op_id')
        if 'op_id' not in intraop_cols_final: intraop_cols_final.insert(0, 'op_id')

        # Save the combined dataset
        print(f"Saving combined data to: {combined_csv_path}")
        df.to_csv(combined_csv_path, index=False)
        
        # Save the preop subset
        print(f"Saving pre-operative subset to: {preop_out_path}")
        df[list(dict.fromkeys(preop_cols_final))].to_csv(preop_out_path, index=False)

        # Save the intraop subset
        print(f"Saving intra-operative subset to: {intraop_out_path}")
        df[list(dict.fromkeys(intraop_cols_final))].to_csv(intraop_out_path, index=False)

        # Save the normalization statistics for later use (e.g., in SHAP analysis)
        print(f"Saving normalization stats to: {norm_csv_path}")
        df_stats.to_csv(norm_csv_path)

        print("\n--- Script Finished Successfully ---")

    except Exception as e:
        print(f"An error occurred during file saving: {e}", file=sys.stderr)
        sys.exit(1)

if __name__ == "__main__":
    main()
