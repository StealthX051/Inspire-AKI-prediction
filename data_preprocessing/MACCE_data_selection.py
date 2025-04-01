import pandas as pd
from pathlib import Path

# ----------------------------------------------------------------------------
# Paths (edit as needed)
# ----------------------------------------------------------------------------
ops_path = Path("/home/server/Projects/data/INSPIRE/physionet.org/files/inspire/1.3/operations.csv")
diag_path = Path("/home/server/Projects/data/INSPIRE/physionet.org/files/inspire/1.3/diagnosis.csv")
macce_output_file = "/home/server/Projects/data/MACCE/ops_macce_events.csv"

# ----------------------------------------------------------------------------
# Read in data
# ----------------------------------------------------------------------------
df_ops = pd.read_csv(ops_path)
df_diag = pd.read_csv(diag_path)

# Convert chart_time to float if needed
df_diag["chart_time"] = df_diag["chart_time"].astype(float)

# ----------------------------------------------------------------------------
# Define MACCE codes & filter
# ----------------------------------------------------------------------------
MACCE_CODES = ["I21", "I63", "I20", "I50", "I46"]
df_diag_macce = df_diag[df_diag["icd10_cm"].isin(MACCE_CODES)]

# ----------------------------------------------------------------------------
# Merge ops + macce diagnoses
# ----------------------------------------------------------------------------
df_merge = pd.merge(
    df_ops,
    df_diag_macce,
    on="subject_id",
    how="left",
    suffixes=("", "_diag")
)

# Keep only diagnoses within 30 days of opend_time
thirty_days_in_minutes = 30 * 24 * 60
df_merge = df_merge[
    (df_merge["chart_time"] >= df_merge["opend_time"]) &
    (df_merge["chart_time"] <= df_merge["opend_time"] + thirty_days_in_minutes)
]

# Now, group by op_id to find all ICD-10 codes that occurred within that window
df_grouped = (
    df_merge
    .groupby("op_id")["icd10_cm"]
    .apply(lambda codes: ";".join(sorted(set(codes))))
    .reset_index()
    .rename(columns={"icd10_cm": "macce_event"})
)

# Add a macce = 1 indicator
df_grouped["macce"] = 1

# ----------------------------------------------------------------------------
# Merge back to the ops DataFrame
# ----------------------------------------------------------------------------
df_final = pd.merge(
    df_ops[["op_id"]],          # only keep op_id from ops
    df_grouped[["op_id", "macce_event", "macce"]],
    on="op_id",
    how="left"
)

# If an op_id wasn’t found in the 30-day MACCE group, fill as no MACCE
df_final["macce"] = df_final["macce"].fillna(0)
df_final["macce_event"] = df_final["macce_event"].fillna("")

# ----------------------------------------------------------------------------
# Keep only the desired columns
# ----------------------------------------------------------------------------
df_final = df_final[["op_id", "macce", "macce_event"]]

# ----------------------------------------------------------------------------
# Save to CSV
# ----------------------------------------------------------------------------
df_final.to_csv(macce_output_file, index=False)

print(df_final.shape)
print(len(df_final["op_id"].unique()))
print(f"Saved to {macce_output_file}")
