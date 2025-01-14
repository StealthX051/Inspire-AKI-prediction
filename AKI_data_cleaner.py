from pathlib import Path
import pandas as pd
from tqdm import tqdm

# Define paths
inspire_path = Path("/home/server/Projects/data/INSPIRE/physionet.org/files/inspire/1.3")
output_path = Path("/home/server/Projects/data/AKI")
vitals_path = inspire_path / "vitals.csv"

# Output file for combined data
output_file = output_path / "combined_vitals.parquet"


# Load in vitals
print(f"Loading vitals from {vitals_path}")
df_vitals = pd.read_csv(vitals_path.as_posix())


# Group by both 'op_id' and 'item_name'
print("Grouping data...")
grouped = df_vitals.groupby(['op_id', 'item_name'])

# Process and store structured data
structured_data = []

# Process each group
for (op_id, item_name), group in tqdm(grouped, desc="Processing Groups"):
    group = group.sort_values(by='chart_time')  # Sort by chart_time
    structured_data.append(group[['op_id', 'item_name', 'chart_time', 'value']])

# Combine all groups into a single DataFrame
structured_df = pd.concat(structured_data, ignore_index=True)

# Save the combined data as a Parquet file
structured_df.to_parquet(output_file, index=False)

print(f"Combined data saved to {output_file}")
