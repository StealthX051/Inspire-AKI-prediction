from pathlib import Path
import pandas as pd
import numpy as np

inspire_path = Path("/home/server/Projects/data/INSPIRE/physionet.org/files/inspire/1.3")
med_path = inspire_path / "medications.csv"
ops_path = inspire_path / "operations.csv"
labs_path = inspire_path / "labs.csv"
vitals_path = inspire_path / "vitals.csv"

# df_med = pd.read_csv(med_path.as_posix())
# df_ops = pd.read_csv(ops_path.as_posix())
# df_labs = pd.read_csv(labs_path.as_posix())
df_vitals = pd.read_csv(vitals_path.as_posix())
# print(df_med)
# print(df_ops)

subject_id_filtered = df_vitals[df_vitals['item_name'] == 'sti']
print(subject_id_filtered)