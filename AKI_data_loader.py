import os

import torch
from torch.utils.data import Dataset, DataLoader
from pathlib import Path
import pandas as pd
from multiprocessing import Pool
from tqdm import tqdm


class TimeSeriesDataset(Dataset):
    def __init__(self, data_dir, preopdata_file):
        """
        Custom Dataset for preloading all time series and preopdata.

        Args:
            data_dir (str or Path): The root directory containing per-op_id directories of time series files.
            preopdata_file (str or Path): The CSV file containing preopdata for each op_id.
        """
        self.data_dir = Path(data_dir)
        self.preopdata = pd.read_csv(preopdata_file)
        self.data = self._load_all_data_parallel()

    def _load_single_op(self, op_dir):
        """
        Loads data for a single op_id.

        Args:
            op_dir (Path): Path to the operation directory.

        Returns:
            tuple: (op_id, dict containing preopdata and time series data)
        """
        op_id = int(op_dir.name.split("_")[1])  # Extract op_id from directory name

        # Fetch preopdata for the current op_id
        preopdata_row = self.preopdata[self.preopdata['op_id'] == op_id]
        if preopdata_row.empty:
            return None  # Skip if no preopdata
        preopdata_row = preopdata_row.iloc[0].to_dict()

        # Load all time series files in the directory
        time_series = {}
        for file in op_dir.glob("*.csv"):
            item_name = file.stem  # Get item_name from file name
            df = pd.read_csv(file)
            time_series[item_name] = {
                "chart_time": df["chart_time"].values,
                "value": df["value"].values,
            }

        return op_id, {"preopdata": preopdata_row, "time_series": time_series}

    def _load_all_data_parallel(self):
        """
        Preloads all time series data into memory in parallel.

        Returns:
            dict: A dictionary where keys are op_id and values are dicts with preopdata and time series data.
        """
        op_dirs = list(self.data_dir.glob("op_*"))  # All directories starting with "op_"

        # Use multiprocessing to load data in parallel
        with Pool() as pool:
            results = list(
                tqdm(pool.imap(self._load_single_op, op_dirs), total=len(op_dirs), desc="Loading Time Series Data")
            )

        # Filter out None results and convert to dictionary
        data = {op_id: content for op_id, content in results if content is not None}
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        """
        Retrieve the preloaded data for a specific op_id.

        Args:
            idx (int): Index of the operation.

        Returns:
            dict: Contains 'op_id', 'preopdata', and 'time_series' for the operation.
        """
        op_id = list(self.data.keys())[idx]
        return {"op_id": op_id, **self.data[op_id]}


# Define paths
data_dir = "/home/server/Projects/data/AKI/intraop_data"
preopdata_file = "/home/server/Projects/data/AKI/preop_data.csv"

# Initialize the dataset
time_series_dataset = TimeSeriesDataset(data_dir, preopdata_file)

# Create the DataLoader
time_series_loader = DataLoader(time_series_dataset, batch_size=8, shuffle=True, num_workers=os.cpu_count())

# Example usage
for batch in tqdm(time_series_loader, desc="Loading Batches"):
    print(batch)
    break
