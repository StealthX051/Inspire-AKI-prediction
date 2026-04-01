import os
import numpy as np
import pandas as pd
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm\

from pathlib import Path


class TimeSeriesDataset(Dataset):
    def __init__(self, vitals_file, preopdata_file):
        """
        Custom Dataset for loading time series data grouped by op_id directly from vitals.csv.

        Args:
            vitals_file (str or Path): Path to the CSV file containing all time series data.
            preopdata_file (str or Path): Path to the CSV file containing preopdata for each op_id.
        """
        # Load preop data
        self.preopdata = pd.read_csv(preopdata_file)

        # Load and group vitals data
        print(f"Loading vitals from {vitals_file}")
        df_vitals = pd.read_csv(vitals_file)

        print("Grouping vitals data by op_id...")
        self.grouped = {
            op_id: group.groupby("item_name")
            for op_id, group in tqdm(df_vitals.groupby("op_id"), desc="Processing op_id Groups")
        }
        # Create a dictionary to store which op_id has which vital labels
        # op_id_vital_map = {
        #     op_id: list(group.groups.keys()) for op_id, group in self.grouped.items()
        # }

        # # Add indicator columns for intraoperative data categories
        # # 18 column preop_data -> 94 column preop_data
        # # e.g. 'has_hr' indicates whether the operation has longitudinal heart rate data

        # # Build a DataFrame to track presence of vital labels
        # vital_presence = pd.DataFrame(0, index=preopdata.index, columns=['has_' + label for label in 
        #                                 set(v for labels in op_id_vital_map.values() for v in labels)])

        # # Populate the DataFrame
        # for op_id, vital_labels in op_id_vital_map.items():
        #     mask = preopdata['op_id'] == op_id
        #     for vital_label in vital_labels:
        #         vital_presence.loc[mask, 'has_' + vital_label] = 1

        # # Combine with the original DataFrame
        # preopdata = pd.concat([preopdata, vital_presence], axis=1)

    def __len__(self):
        return len(self.grouped)

    def __getitem__(self, idx):
        """
        Retrieve all time series data and corresponding preop data for a specific op_id.

        Args:
            idx (int): Index of the operation.

        Returns:
            dict: Contains 'op_id', 'time_series', and 'preopdata'.
        """
        op_id = list(self.grouped.keys())[idx]
        item_groups = self.grouped[op_id]

        # Extract all time series for this op_id
        time_series = {
            item_name: {
                "chart_time": group["chart_time"].values,
                "value": group["value"].values,
            }
            for item_name, group in item_groups
        }

        # Fetch preopdata for the current op_id
        preopdata_row = self.preopdata[self.preopdata['op_id'] == op_id]
        preopdata = preopdata_row.iloc[0].to_dict() if not preopdata_row.empty else {}

        return {"op_id": op_id, "time_series": time_series, "preopdata": preopdata}


# Custom collate_fn for handling variable-length time series
def custom_collate_fn(batch):
    """
    Custom collate function for variable-length time series data.

    Args:
        batch: List of dictionaries from the Dataset.

    Returns:
        dict: Batched data grouped into lists for each key.
    """
    batch_data = {
        "op_id": [],
        "time_series": [],
        "preopdata": [],
    }

    for item in batch:
        batch_data["op_id"].append(item["op_id"])
        batch_data["time_series"].append(item["time_series"])
        batch_data["preopdata"].append(item["preopdata"])

    return batch_data

# Define paths
inspire_path = Path("/home/server/Projects/data/INSPIRE/physionet.org/files/inspire/1.3")
vitals_file = inspire_path / "vitals.csv"
preopdata_file = "/home/server/Projects/data/AKI/preop_data.csv"

# Initialize the dataset
time_series_dataset = TimeSeriesDataset(vitals_file.as_posix(), preopdata_file)

# Create the DataLoader
time_series_loader = DataLoader(
    time_series_dataset,
    batch_size=1,
    shuffle=True,
    num_workers=os.cpu_count(),  # Use all available CPU cores
    collate_fn=custom_collate_fn,  # Use the custom collate function
)

# Example usage
for batch in tqdm(time_series_loader, desc="Loading Batches"):
    print(batch)
    break
