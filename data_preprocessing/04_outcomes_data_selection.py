import pandas as pd
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths (edit as needed)
# ----------------------------------------------------------------------------
# Input data paths
ops_path = Path("/home/server/Projects/data/INSPIRE/physionet.org/files/inspire/1.3/operations.csv")
diag_path = Path("/home/server/Projects/data/INSPIRE/physionet.org/files/inspire/1.3/diagnosis.csv")
vitals_path = Path("/home/server/Projects/data/INSPIRE/physionet.org/files/inspire/1.3/ward_vitals.csv")

# Base feature sets paths
base_path = Path("/home/server/Projects/data/base/")
base_combined_csv = base_path / 'tabular_combined.csv'
base_preop_csv = base_path / 'tabular_preop.csv'
base_intraop_csv = base_path / 'tabular_intraop.csv'

# Output directory for the augmented feature sets
output_path = Path("/home/server/Projects/data/Multiple-Outcomes/")
output_path.mkdir(parents=True, exist_ok=True) # Ensure output directory exists

# Output file paths
outcomes_combined_csv = output_path / "outcomes_combined.csv"
outcomes_preop_csv = output_path / "outcomes_preop.csv"
outcomes_intraop_csv = output_path / "outcomes_intraop.csv"


# ----------------------------------------------------------------------------
# Central Configuration
# ----------------------------------------------------------------------------
# This dictionary holds key parameters for defining clinical outcomes.
# Modifying these values will change the definitions across the script.
CONFIG = {
    "ENABLE_PROCEDURE_FILTER": False,      # Master switch for procedure filtering. Set to False to include all operations.
    "TARGET_PCS_CODES": ["02VW3DZ", "04V00DZ"], # Target ICD-10-PCS codes for filtering.
    "ICD_OUTCOME_WINDOW_DAYS": 30,      # Window for ICD-based outcomes (e.g., MACCE, PNA, PE)
    "MORTALITY_WINDOW_DAYS": 30,        # Window for 30-day all-cause mortality
    "PRF_WINDOW_HOURS": 48,             # Window for Postoperative Respiratory Failure (PRF)
    "EXTENDED_LOS_PERCENTILE": 0.75     # Percentile to define extended Length of Stay (LOS)
}


# ----------------------------------------------------------------------------
# Define Outcome ICD-10 Codes
# ----------------------------------------------------------------------------
MACCE_CODES = ("I21", "I63", "I20", "I50", "I46")
PNA_CODES = tuple(f"J1{i}" for i in range(1, 9))
PE_CODES = ("I26",)


# ----------------------------------------------------------------------------
# Function to Process ICD-10 Code Based Outcomes
# ----------------------------------------------------------------------------
def process_icd_outcome(df_ops, df_diag, icd_codes, outcome_name):
    """Identifies operations followed by a specific clinical outcome within a defined window based on ICD-10 codes."""
    print(f"Processing ICD-10 outcome: {outcome_name.upper()}...")
    df_diag_filtered = df_diag[df_diag["icd10_cm"].str.startswith(icd_codes, na=False)].copy()
    
    df_merge = pd.merge(df_ops, df_diag_filtered, on="subject_id", how="left")
    
    # Calculate time window from the central configuration
    time_window_in_minutes = CONFIG["ICD_OUTCOME_WINDOW_DAYS"] * 24 * 60

    print(f"  - Time window for {outcome_name}: {time_window_in_minutes} minutes")
    
    df_merge_in_window = df_merge[
        (df_merge["chart_time"] >= df_merge["opend_time"]) &
        (df_merge["chart_time"] <= df_merge["opend_time"] + time_window_in_minutes)
    ].copy()

    df_grouped = (
        df_merge_in_window
        .groupby("op_id")["icd10_cm"]
        .apply(lambda codes: ";".join(sorted(set(codes.dropna()))))
        .reset_index()
        .rename(columns={"icd10_cm": f"{outcome_name}_event"})
    )

    if not df_grouped.empty:
        df_grouped[outcome_name] = True
    else:
        df_grouped[outcome_name] = pd.Series(dtype=bool)
    
    return df_grouped

# ----------------------------------------------------------------------------
# Function to Process Postoperative Respiratory Failure (PRF)
# ----------------------------------------------------------------------------
def process_prf(df_ops, df_vitals):
    """Identifies operations followed by PRF, defined as ventilation > X hours after surgery end."""
    print("Processing time-series outcome: PRF...")
    df_vent = df_vitals[(df_vitals['item_name'] == 'vent') & (pd.to_numeric(df_vitals['value'], errors='coerce') == 1)].copy()
    df_merge = pd.merge(df_ops, df_vent, on="subject_id", how="left")
    
    # Calculate time window from the central configuration
    prf_window_in_minutes = CONFIG["PRF_WINDOW_HOURS"] * 60

    df_prf_events = df_merge[df_merge["chart_time"] > (df_merge["opend_time"] + prf_window_in_minutes)].copy()
    
    prf_op_ids = df_prf_events["op_id"].unique()
    df_prf = pd.DataFrame({'op_id': prf_op_ids})
    if not df_prf.empty:
        df_prf['prf'] = True
    
    return df_prf

# ----------------------------------------------------------------------------
# Function to Process Extended Hospital Length of Stay (LOS)
# ----------------------------------------------------------------------------
def process_extended_los(df_ops):
    """Identifies operations with a postoperative hospital stay > Xth percentile."""
    print("Processing outcome: Extended LOS...")
    
    # Work on a copy to avoid SettingWithCopyWarning
    df_ops_copy = df_ops[['op_id', 'opend_time', 'discharge_time']].copy()

    # Calculate postoperative LOS using 'discharge_time' from the operations table
    df_ops_copy.dropna(subset=['opend_time', 'discharge_time'], inplace=True)
    df_ops_copy['postop_los'] = df_ops_copy['discharge_time'] - df_ops_copy['opend_time']
    
    # Ensure LOS is non-negative
    df_ops_copy = df_ops_copy[df_ops_copy['postop_los'] >= 0].copy()

    # Calculate the threshold from the central configuration
    los_threshold = df_ops_copy['postop_los'].quantile(CONFIG["EXTENDED_LOS_PERCENTILE"])
    print(f"  - Extended LOS threshold ({CONFIG['EXTENDED_LOS_PERCENTILE']:.0%} percentile): {los_threshold / (60*24):.2f} days")

    # Flag operations with LOS > threshold
    df_ops_copy['extended_los'] = df_ops_copy['postop_los'] > los_threshold
    
    return df_ops_copy[['op_id', 'extended_los']]

# ----------------------------------------------------------------------------
# Function to Process Postoperative ICU Admission
# ----------------------------------------------------------------------------
def process_postop_icu(df_ops):
    """Identifies operations followed by an ICU admission."""
    print("Processing outcome: Postoperative ICU Admission...")
    df_ops_copy = df_ops[['op_id', 'opend_time', 'icuin_time']].copy()
    df_ops_copy['postop_icu_admission'] = (
        df_ops_copy['icuin_time'].notna() & 
        (df_ops_copy['icuin_time'] > df_ops_copy['opend_time'])
    )
    return df_ops_copy[['op_id', 'postop_icu_admission']]

# ----------------------------------------------------------------------------
# Function to Process 30-Day All-Cause Mortality
# ----------------------------------------------------------------------------
def process_30day_mortality(df_ops):
    """Identifies operations followed by all-cause death within a defined window."""
    print("Processing outcome: 30-Day Mortality...")
    df_ops_copy = df_ops[['op_id', 'opend_time', 'allcause_death_time']].copy()
    df_ops_copy.dropna(subset=['opend_time', 'allcause_death_time'], inplace=True)
    
    # Calculate time window from the central configuration
    mortality_window_in_minutes = CONFIG["MORTALITY_WINDOW_DAYS"] * 24 * 60
    
    df_ops_copy['mortality_30day'] = (
        (df_ops_copy['allcause_death_time'] > df_ops_copy['opend_time']) &
        (df_ops_copy['allcause_death_time'] <= (df_ops_copy['opend_time'] + mortality_window_in_minutes))
    )
    df_mortality = df_ops_copy[df_ops_copy['mortality_30day'] == True]
    return df_mortality[['op_id', 'mortality_30day']]

# ----------------------------------------------------------------------------
# Main Execution
# ----------------------------------------------------------------------------
# 1. Read in the source data
print("Reading source data...")
df_ops = pd.read_csv(ops_path)
df_diag = pd.read_csv(diag_path)
df_vitals = pd.read_csv(vitals_path)


for name, df in [("combined", pd.read_csv(base_combined_csv)),
                 ("preop", pd.read_csv(base_preop_csv)),
                 ("intraop", pd.read_csv(base_intraop_csv))]:
    print(name, "rows:", len(df), "unique op_id:", df['op_id'].nunique(),
          "intersection with ops:", len(set(df['op_id']).intersection(set(df_ops['op_id']))))

# 2. Prepare data types for merging and comparison
for df in [df_ops, df_diag, df_vitals]:
    time_cols = [col for col in df.columns if 'time' in col]
    for col in time_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')

df_diag["icd10_cm"] = df_diag["icd10_cm"].astype(str)
# Ensure icd10_pcs exists and is a string for filtering
if 'icd10_pcs' in df_ops.columns:
    print("icd10_pcs column found in operations.csv")
    df_ops['icd10_pcs'] = df_ops['icd10_pcs'].astype(str).fillna('')

# 2.5 Filter for Specific Operations (if enabled)
if CONFIG["ENABLE_PROCEDURE_FILTER"]:
    print("Procedure filtering is ENABLED.")
    if 'icd10_pcs' not in df_ops.columns:
        print("  - WARNING: 'icd10_pcs' column not found in operations.csv. Skipping filter.")
    else:
        initial_op_count = len(df_ops)
        
        # Create a regex pattern to find any of the target codes.
        # This will match if the string contains any of the codes in the list.
        pcs_filter_pattern = '|'.join(CONFIG["TARGET_PCS_CODES"])
        
        # Filter the operations dataframe
        df_ops = df_ops[df_ops['icd10_pcs'].str.contains(pcs_filter_pattern, na=False)].copy()
        
        print(f"  - Filtered {initial_op_count} operations down to {len(df_ops)} based on {len(CONFIG['TARGET_PCS_CODES'])} PCS codes.")
else:
    print("Procedure filtering is DISABLED.")


# 3. Process each clinical outcome individually
df_macce = process_icd_outcome(df_ops, df_diag, MACCE_CODES, "macce")
df_pna = process_icd_outcome(df_ops, df_diag, PNA_CODES, "pna")
df_pe = process_icd_outcome(df_ops, df_diag, PE_CODES, "pe")
df_prf = process_prf(df_ops, df_vitals)
df_los = process_extended_los(df_ops)
df_icu = process_postop_icu(df_ops)
df_mortality = process_30day_mortality(df_ops)

# 4. Sequentially merge all outcomes into a final DataFrame
print("Merging all outcomes...")
df_final = df_ops[["op_id"]].drop_duplicates().copy()

# Merge each outcome result. A left merge ensures all operations are kept.
df_final = pd.merge(df_final, df_macce, on="op_id", how="left")
df_final = pd.merge(df_final, df_pna, on="op_id", how="left")
df_final = pd.merge(df_final, df_pe, on="op_id", how="left")
df_final = pd.merge(df_final, df_prf, on="op_id", how="left")
df_final = pd.merge(df_final, df_los, on="op_id", how="left")
df_final = pd.merge(df_final, df_icu, on="op_id", how="left")
df_final = pd.merge(df_final, df_mortality, on="op_id", how="left")

# 5. Clean up the final DataFrame
# not including prf
outcomes_to_clean = ["macce", "pna", "pe", "prf", "extended_los", "postop_icu_admission", "mortality_30day"]
for outcome in outcomes_to_clean:
    df_final[outcome] = df_final[outcome].fillna(False)
    event_col = f"{outcome}_event"
    if event_col in df_final.columns:
        df_final[event_col] = df_final[event_col].fillna("")

# 6. Merge outcomes with base feature sets and save
print("Merging outcomes with base feature sets and saving files...")
outcome_cols_to_merge = ['op_id', 'macce', 'pna', 'pe', 'prf', 'extended_los', 'postop_icu_admission', 'mortality_30day']

# Combined dataset
df_combined = pd.read_csv(base_combined_csv)
df_combined_outcomes = df_combined.merge(df_final[outcome_cols_to_merge], on='op_id', how='inner')
df_combined_outcomes.to_csv(outcomes_combined_csv, index=False)
print(f"Saved combined data with outcomes to {outcomes_combined_csv}")

# Pre-operative dataset
df_preop = pd.read_csv(base_preop_csv)
df_preop_outcomes = df_preop.merge(df_final[outcome_cols_to_merge], on='op_id', how='inner')
df_preop_outcomes.to_csv(outcomes_preop_csv, index=False)
print(f"Saved pre-op data with outcomes to {outcomes_preop_csv}")

# Intra-operative dataset
df_intraop = pd.read_csv(base_intraop_csv)
df_intraop_outcomes = df_intraop.merge(df_final[outcome_cols_to_merge], on='op_id', how='inner')
df_intraop_outcomes.to_csv(outcomes_intraop_csv, index=False)
print(f"Saved intra-op data with outcomes to {outcomes_intraop_csv}")

print("\nScript finished successfully.")
print(f"Final shape of combined data: {df_combined_outcomes.shape}")
print(f"Outcome counts in combined data:\n{df_combined_outcomes[outcome_cols_to_merge[1:]].sum()}")
