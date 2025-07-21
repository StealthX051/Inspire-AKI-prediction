# test file EXCLUSIVELY FOR TESTING OUTSIDE OF IPYNB ENV







import pandas as pd
import numpy as np
from IPython.display import display, HTML
import html

# --- Configuration ---
# Define the path to your input data file.
NORMALIZATION_STATS_PATH = '/home/server/Projects/data/base/normalization_stats.csv'
INPUT_CSV_PATH = '/home/server/Projects/data/AKI/tabular_combined_unnormalized.csv'
# Define the path for the output HTML file.
OUTPUT_HTML_PATH = 'descriptive_table_test.html'

def denormalize_data(df: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """
    Denormalizes the DataFrame using the provided statistics.

    Args:
        df: A DataFrame containing normalized data.
        stats: A DataFrame containing mean and standard deviation for each feature.

    Returns:
        A DataFrame with denormalized values.
    """
    if 'std' not in stats.columns:
        stats['std'] = np.sqrt(stats['var'])
    common_cols = df.columns.intersection(stats.index)

    means = stats.loc[common_cols, 'mean']
    stds = stats.loc[common_cols, 'std']
    
    df_denorm = df.copy()
    df_denorm[common_cols] = df[common_cols] * stds + means
    return df_denorm

def generate_descriptive_table_html(df: pd.DataFrame) -> str:
    """
    Generates a formatted descriptive statistics table as an HTML string.

    The output is a self-contained HTML file that can be opened in a browser,
    making it easy to copy and paste into rich text editors like Google Docs.

    Args:
        df: A pre-processed DataFrame with one row per subject.

    Returns:
        A string containing the full HTML of the descriptive statistics table.
    """
    # This list will hold the HTML `<tr>` strings for our table.
    table_rows_html = []
    total_population = len(df)

    # --- Helper function for formatting ---
    def add_row(characteristic: str, finding: str, is_section_header: bool = False, is_indented: bool = False):
        """Adds a formatted row to the HTML table list."""
        if is_section_header:
            # Section headers span both columns and have a border
            row_html = f'<tr><td colspan="2" style="font-weight: bold; padding-top: 10px; border: 1px solid black;">{characteristic}</td></tr>'
        else:
            # Regular rows have borders on each cell
            char_style = 'padding-left: 25px;' if is_indented else ''
            row_html = f'<tr><td style="{char_style} border: 1px solid black;">{characteristic}</td><td style="border: 1px solid black;">{finding}</td></tr>'
        table_rows_html.append(row_html)

    # --- Population Demographics ---
    continuous_vars = {
        'Age, y': 'age',
        'Weight, kg': 'weight',
        'Height, cm': 'height',
        'BMI, kg/m^2': 'BMI',
        'ASA': 'asa',
        'Number of Preexisting Cardiac Diagnoses': 'num_card_events',
        'Booking Case Length, min': 'booking_case_length'
    }
    for desc, col in continuous_vars.items():
        if col in df.columns:
            mean = df[col].mean()
            std = df[col].std()
            add_row(f'{desc}, mean (SD)', f'{mean:.2f} ± {std:.2f}')

    if 'sex' in df.columns:
        female_count = df['sex'].value_counts().get(False, 0)
        percentage = (female_count / total_population) * 100
        add_row('Female sex, n (%)', f'{female_count} ({percentage:.2f}%)')
    
    # --- ASA Classification ---
    if 'asa' in df.columns:
        add_row('ASA classification, n (%)', '', is_section_header=True)
        asa_counts = df['asa'].value_counts().sort_index()
        for level, count in asa_counts.items():
            percentage = (count / total_population) * 100
            add_row(f'{level}', f'{count} ({percentage:.3f}%)', is_indented=True)

    # --- Postoperative AKI ---
    if 'aki_boolean' in df.columns:
        aki_count = df['aki_boolean'].value_counts().get(True, 0)
        percentage = (aki_count / total_population) * 100
        add_row('Postoperative AKI, n (%)', f'{aki_count} ({percentage:.3f}%)')

    # --- Department Surgery Type ---
    # Dictionary to map abbreviations to full department names
    dept_mapping = {
        'UR': 'Urology',
        'RO': 'Radiation Oncology',
        'RAD': 'Radiology',
        'PS': 'Plastic Surgery',
        'PED': 'Pediatric Surgery',
        'OT': 'Orthopedic Surgery',
        'OS': 'Oral Surgery',
        'OL': 'Otorhinolaryngology',
        'OG': 'Obstetrics and Gynecology',
        'NS': 'Neurosurgery',
        'IM': 'Internal Medicine',
        'GS': 'General Surgery',
        'EM': 'Emergency Medicine',
        'DM': 'Dermatology',
        'CTS': 'Cardiothoracic Surgery',
        'AN': 'Anesthesiology'
    }
    
    dept_cols = sorted([col for col in df.columns if 'department' in col])
    if dept_cols:
        add_row('Department Surgery type, n (%)', '', is_section_header=True)
        for col in dept_cols:
            # Extract abbreviation from column name (e.g., 'department_GS' -> 'GS')
            abbreviation = col.replace('department_', '').upper()
            # Look up the full name, defaulting to the abbreviation if not found
            dept_name = dept_mapping.get(abbreviation, abbreviation)
            
            count = df[col].sum()
            percentage = (count / total_population) * 100
            add_row(f'{dept_name}', f'{count} ({percentage:.4f}%)', is_indented=True)

    # --- Assemble the final HTML string ---
    all_rows_str = "\n".join(table_rows_html)
    html_template = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <title>Descriptive Statistics</title>
        <style>
            body {{ font-family: 'Times New Roman', Times, serif; margin: 20px; }}
            table {{ border-collapse: collapse; width: 600px; border: 1px solid black; }}
            th, td {{ padding: 8px 12px; text-align: left; border: 1px solid black; }}
            th {{ background-color: #f2f2f2; font-weight: bold; }}
            .title {{ font-size: 1.2em; font-weight: bold; margin-bottom: 10px; }}
        </style>
    </head>
    <body>
        <div class="title">Table 1. Characteristics of Cohort</div>
        <table>
            <thead>
                <tr>
                    <th>Characteristic</th>
                    <th>Finding (N={total_population})</th>
                </tr>
            </thead>
            <tbody>
                {all_rows_str}
            </tbody>
        </table>
    </body>
    </html>
    """
    return html_template


print("Starting script...")
try:
    df_norm = pd.read_csv(INPUT_CSV_PATH)
    df_stats = pd.read_csv(NORMALIZATION_STATS_PATH, index_col='Unnamed: 0')
    df = denormalize_data(df_norm, df_stats)
    
    # Pre-processing: one row per subject, not operation
    df_subject = df.groupby('subject_id').agg('first').reset_index()
    df_subject = df_subject.drop(columns=['op_id'], errors='ignore')

    print(f"Loaded and processed {len(df_subject)} unique subjects.")

    # Generate the descriptive table as an HTML string
    descriptive_table_html = generate_descriptive_table_html(df_subject)

    # --- Save the clean HTML to a file ---
    with open(OUTPUT_HTML_PATH, 'w', encoding='utf-8') as f:
        f.write(descriptive_table_html)

    print("\n" + "="*50)
    print("SUCCESS!")
    print(f"A clean version of the rich text table has also been saved to '{OUTPUT_HTML_PATH}'")
    print("="*50 + "\n")

except FileNotFoundError:
    print(f"Error: Input file not found at '{INPUT_CSV_PATH}'")
except ImportError:
    print("\nNOTE: To display the table in the notebook, you must run this script in a Jupyter environment.")
except Exception as e:
    print(f"An unexpected error occurred: {e}")






# import pandas as pd
# import numpy as np

# TABULAR_COMBINED_PATH = '/home/server/Projects/data/AKI/tabular_combined.csv'
# NORMALIZATION_STATS_PATH = '/home/server/Projects/data/base/normalization_stats.csv'

# df_normalized = pd.read_csv(TABULAR_COMBINED_PATH)
# df_stats = pd.read_csv(NORMALIZATION_STATS_PATH)
# df_stats = df_stats.set_index('Unnamed: 0')
# if 'std' not in df_stats.columns:
#     df_stats['std'] = np.sqrt(df_stats['var'])
# display = print

# common_cols = df_normalized.columns.intersection(df_stats.index)

# # Only process those
# means = df_stats.loc[common_cols, 'mean']
# stds = df_stats.loc[common_cols, 'std']

# # Unnormalize
# df_unnormalized = df_normalized.copy()
# df_unnormalized[common_cols] = df_normalized[common_cols] * stds + means





# import pandas as pd
# from pathlib import Path
# import sys

# base_path = Path("/home/server/Projects/data/base/")
# aki_path = Path("/home/server/Projects/data/AKI/")

# base_combined_csv =      base_path / "tabular_combined.csv"
# aki_combined_csv =      aki_path / "tabular_combined.csv"
# temp_combined_csv =      aki_path / "temp_tabular_combined.csv"



# base_df = pd.read_csv(base_combined_csv)
# df = pd.read_csv(aki_combined_csv)
# aki_df = pd.read_csv(temp_combined_csv)

# print(len(aki_df))

# # operations that are in consort but not in aki
# aki_op_ids = [422310768, 471156425, 490727246]
# # operations that are in aki but not in consort
# op_ids = [435191458.0]
# OP_ID = 435191458.0
# SUBJECT_ID = 150497664
# TEST_i = 0

# for op_id in (set(df['op_id'].unique()) - set(aki_df['op_id'].unique())):
#     print(op_id)
# print('za')
# for op_id in (set(aki_df['op_id'].unique()) - set(df['op_id'].unique())):
#     print(op_id)

# def test(df, string=None):
#     global TEST_i

#     if string is not None:
#         print('test', TEST_i, string)
#     else:
#         print('test', TEST_i)
#     TEST_i += 1
#     if OP_ID in df['op_id'].values:
#         print(OP_ID, " found in DataFrame\n")
#     else:
#         print("Not found in DataFrame\n")
    




# from pathlib import Path
# import pandas as pd
# import numpy as np
# from collections import Counter
# from tqdm import tqdm

# def nprint(string):
#     print("="*25, string, "="*25)

# # All the input paths
# inspire_path = Path("/home/server/Projects/data/INSPIRE/physionet.org/files/inspire/1.3")
# labs_path = inspire_path / "labs.csv"
# vitals_path = inspire_path / "vitals.csv"
# ops_path = inspire_path / "operations.csv"
# diagnosis_path = inspire_path / "diagnosis.csv"
# ward_vitals_path = inspire_path / "ward_vitals.csv"


# # move from /aki to /base later
# base_path = Path("/home/server/Projects/data/base/")

# # if you change output_csv, also change in aki data selection and create base
# output_csv = "/home/server/Projects/data/AKI/preop_data_test.csv"

# nprint("starting")
# df_labs = pd.read_csv(labs_path)
# df_labs["chart_time"] = df_labs["chart_time"].astype(float)
# df_ward = pd.read_csv(ward_vitals_path)
# df_ward["chart_time"] = df_ward["chart_time"].astype(float)
# df_vitals = pd.read_csv(vitals_path)
# df_ops = pd.read_csv(ops_path)
# df_diags = pd.read_csv(diagnosis_path.as_posix())

# nprint("finished reading csvs")

# pd.set_option('display.max_rows', None)
# pd.set_option('display.max_columns', None)

# df = df_labs.loc[df_labs['item_name'] == 'creatinine']
# print(df[df['subject_id'].isin([SUBJECT_ID])])


# ### code from AKI_data_cleaner.ipynb
# df_preop = df_ops.copy()
# # Keeping only the necessary columns
# desired_columns = [
#     'op_id', 'subject_id', 'age', 'sex', 'height', 'weight', 
#     'asa', 'emop', 'opstart_time', 'opend_time', 
#     'inhosp_death_time', 'allcause_death_time', 'orin_time', 'orout_time',
# ]
# # Select only the desired columns
# df_preop = df_preop[desired_columns]
# # Add BSA (Body Surface Area) and BMI (Body Mass Index) columns
# # Ensure height and weight are valid (not NaN) for calculations
# valid_mask = df_preop['height'].notna() & df_preop['weight'].notna()
# # Initialize BSA and BMI with NaN
# df_preop['BSA'] = np.nan
# df_preop['BMI'] = np.nan
# # Calculate BSA and BMI only for valid rows
# df_preop.loc[valid_mask, 'BSA'] = np.sqrt((df_preop.loc[valid_mask, 'height'] * df_preop.loc[valid_mask, 'weight']) / 3600)
# df_preop.loc[valid_mask, 'BMI'] = df_preop.loc[valid_mask, 'weight'] / ((df_preop.loc[valid_mask, 'height'] / 100) ** 2)
# # Add Booking Case Length column
# valid_mask = df_preop['orin_time'].notna() & df_preop['orout_time'].notna()
# df_preop['booking_case_length'] = np.nan
# df_preop.loc[valid_mask, 'booking_case_length'] = df_preop.loc[valid_mask, 'orout_time'] - df_preop.loc[valid_mask, 'orin_time']
# # Remove orin_time and orout_time columns
# df_preop = df_preop.drop(columns=['orin_time', 'orout_time'])
# # Filter cardiovascular diagnoses (ICD-10-CM codes starting with 'I')
# df_diags_cvd = df_diags[df_diags['icd10_cm'].str.startswith('I', na=False)]
# # Merge operations and cardiovascular diagnoses on subject_id
# merged = pd.merge(
#     df_preop[['op_id', 'subject_id', 'opstart_time']],
#     df_diags_cvd[['subject_id', 'chart_time']],
#     on='subject_id',
#     how='inner'
# )
# # Filter diagnoses where chart_time < opstart_time
# merged = merged[merged['chart_time'] < merged['opstart_time']]
# # Count the number of diagnoses for each operation
# num_card_events = merged.groupby('op_id').size().reset_index(name='num_card_events')
# # Merge the counts back into the operations DataFrame
# df_preop = pd.merge(
#     df_preop,
#     num_card_events,
#     on='op_id',
#     how='left'
# )
# # Fill NaN values with 0 for operations with no past cardiovascular diagnoses
# df_preop['num_card_events'] = df_preop['num_card_events'].fillna(0).astype(int)
# ### end of code from AKI_data_cleaner.ipynb


# test(df_preop, 'df_preop after initial processing')
# df_preop = df_preop[df_preop["asa"] < 6]
# df_preop = df_preop[df_preop["age"] >= 18]
# df_preop = df_preop.dropna(subset="opend_time")
# df_preop = df_preop.dropna(subset="opstart_time")
# df_preop["op_len"] = df_preop["opend_time"] - df_preop["opstart_time"]

# # encode gender and remove rows with missing height/weight
# df_preop["sex"] = df_preop["sex"] == "M"
# df_preop = df_preop[~(df_preop['weight'].isna() | df_preop['height'].isna())]
# df_preop = df_preop[(df_preop['weight'] != 0) & (df_preop['height'] != 0)] #& (df['op_id'] != 435191458)] # prejudicial

# # Replace antypes with numbers, after removing rows with regional set as antype
# df_ops = df_ops.drop(df_ops[df_ops['antype'] == 'Regional'].index)
# df_ops.loc[df_ops['antype'] == 'General', 'antype'] = 0     
# df_ops.loc[df_ops['antype'] == 'MAC', 'antype'] = 1
# df_ops.loc[df_ops['antype'] == 'Neuraxial', 'antype'] = 1
# test(df_ops, 'df_ops after replacing antypes')
# # Replace departments with one-hot encodings
# df_ops = df_ops[df_ops['department'] != 'PED']
# df_ops = pd.get_dummies(df_ops, columns=['department'])
# cols_to_keep = ['op_id', 'subject_id', 'antype']
# for col in df_ops.columns:
#     if 'department_' in col:
#         cols_to_keep.append(col)
# df_preop = pd.merge(df_preop, df_ops[cols_to_keep], on=['op_id', 'subject_id'], how='inner')

# nprint("finished basic filtering")

# preop_item_names = [
#     "total_protein",
#     "sodium",
#     "potassium",
#     "platelet",
#     "glucose",
#     "wbc",
#     "alt",
#     "chloride",
#     "lymphocyte",
#     "phosphorus",
#     "albumin",
#     "fibrinogen",
#     "creatinine",
#     "ptinr",
#     "total_bilirubin",
#     "alp",
#     "aptt",
#     "calcium",
#     "bun",
#     "ast",
#     "crp",
#     "hb",
#     "hct",
#     "seg"
# ]
# test(df_preop, 'df_preop before merging preop labs')
# for item_name in preop_item_names:
#     df_preop = pd.merge_asof(df_preop.sort_values('opstart_time'), 
#                     df_labs.loc[df_labs['item_name'] == item_name].sort_values('chart_time'), # grab rows w the item name we want and sort by chart_time
#                     left_on='opstart_time', right_on='chart_time', by='subject_id',           # chooses row in df_labs w greatest chart_time that is still less than opstart_time and matches subject_id
#                     tolerance=90 * 24 * 60, suffixes=('', '_'))                               # 90 day tolerance
#     df_preop.drop(columns=['chart_time', 'item_name'], inplace=True)
#     df_preop.rename(columns={'value':f'preop_{item_name}'}, inplace=True)
# nprint("finished getting preop data from time series")
# test(df_preop, 'df_preop after merging preop labs')





# # --- Configuration Constants ---
# # Centralize clinical definitions and magic values for easy updates.
# # NOTE: Time windows are now in minutes to match the source data format.
# CONFIG = {
#     "PRE_OP_WINDOW_MINUTES": 90 * 24 * 60,  # 90 days in minutes
#     "POST_OP_WINDOW_MINUTES": 48 * 60,   # 48 hours in minutes
#     "MAX_PRE_OP_SCR": 4.5,
#     "CREATININE_ITEM_NAME": "creatinine",
#     "DIALYSIS_ITEM_NAME": "crrt",
#     "EXCLUSION_PREFIXES": ["10", "0TY", "B50", "B51"],
#     "EPSILON": 1e-9, # Small constant to prevent division by zero
#     "REQUIRED_OP_COLS": [
#         'op_id', 'subject_id', 'age', 'antype', 'asa', 'icd10_pcs',
#         'height', 'weight', 'opstart_time', 'opend_time', 'department'
#     ]
# }

# def load_and_validate_data():
#     """Loads and validates the necessary CSV files."""
#     try:
#         # Define the base paths for the data files
#         inspire_path = Path("/home/server/Projects/data/INSPIRE/physionet.org/files/inspire/1.3")
#         aki_path = Path("/home/server/Projects/data/AKI/") # Path for intraop data
        
#         # Define full paths for each required file
#         ops_path = inspire_path / "operations.csv"
#         labs_path = inspire_path / "labs.csv"
#         ward_vitals_path = inspire_path / "ward_vitals.csv"
#         intraop_path = aki_path / "feature_engineered.csv"

#         # Load the dataframes from the specified paths
#         df_operations = pd.read_csv(ops_path)
#         df_labs = pd.read_csv(labs_path)
#         df_ward = pd.read_csv(ward_vitals_path)
#         print(f"Loading intraoperative data from: {intraop_path}")
#         # Only load op_id from the feature_engineered file for the intra-op data check
#         df_intraop = pd.read_csv(intraop_path, usecols=['op_id'])

#     except FileNotFoundError as e:
#         print(f"Error: {e}. Data file not found.", file=sys.stderr)
#         return None, None, None, None

#     if not all(col in df_operations.columns for col in CONFIG["REQUIRED_OP_COLS"]):
#         print("Error: 'operations.csv' is missing required columns.", file=sys.stderr)
#         return None, None, None, None

#     # IMPORTANT: Ensure time columns are treated as numeric (minutes), not datetime objects.
#     for df, col in [(df_operations, 'opstart_time'), (df_operations, 'opend_time'), (df_labs, 'chart_time')]:
#         df[col] = pd.to_numeric(df[col], errors='coerce')

#     return df_operations, df_labs, df_ward, df_intraop

# def process_labs(df_ops, df_labs):
#     """
#     Pre-processes lab data to efficiently find required creatinine values using numerical minute-based time.
#     """
#     creatinine_labs = df_labs[df_labs['item_name'] == CONFIG["CREATININE_ITEM_NAME"]].dropna(subset=['value', 'chart_time'])
#     merged_labs = pd.merge(df_ops[['op_id', 'subject_id', 'opstart_time', 'opend_time']], creatinine_labs, on='subject_id')
#     test(merged_labs, 'merged labs')


#     pd.set_option('display.max_rows', None)
#     pd.set_option('display.max_columns', None)

#     print(merged_labs[merged_labs['op_id'].isin([OP_ID])])

#     # Pre-operative window: Find labs within 90 days (in minutes) before opstart_time
#     pre_op_mask = (merged_labs['chart_time'] < merged_labs['opstart_time']) & \
#                   (merged_labs['chart_time'] >= merged_labs['opstart_time'] - CONFIG["PRE_OP_WINDOW_MINUTES"])
#     last_pre_op_scr = merged_labs[pre_op_mask].sort_values('chart_time').groupby('op_id').tail(1)[['op_id', 'value']]
#     last_pre_op_scr = last_pre_op_scr.rename(columns={'value': 'preop_creatinine'})
#     test(last_pre_op_scr, 'last pre op scr')

#     # Calculate max creatinine for 2-day and 7-day windows for AKI staging
#     post_op_2d_mask = (merged_labs['chart_time'] > merged_labs['opend_time']) & \
#                       (merged_labs['chart_time'] <= merged_labs['opend_time'] + (2 * 24 * 60))
#     post_op_7d_mask = (merged_labs['chart_time'] > merged_labs['opend_time']) & \
#                       (merged_labs['chart_time'] <= merged_labs['opend_time'] + (7 * 24 * 60))

#     max_postop_2d = merged_labs[post_op_2d_mask].groupby('op_id')['value'].max().rename('postop_creatinine_2_days')
#     max_postop_7d = merged_labs[post_op_7d_mask].groupby('op_id')['value'].max().rename('postop_creatinine_7_days')
    
#     # Use 'left' merge to keep all patients with a preop_creatinine, even if they lack post-op values.
#     # This ensures they are carried forward to the filtering step where NaNs will be evaluated.
#     processed_labs = pd.merge(last_pre_op_scr, max_postop_2d, on='op_id', how='left')
#     processed_labs = pd.merge(processed_labs, max_postop_7d, on='op_id', how='left')
#     test(processed_labs, 'inner proc labs')
    
#     return processed_labs

# def apply_filters(df_operations, processed_labs_df, df_intraop, df_ward):
#     """Applies the sequential filtering logic to the cohort."""
#     cohort_counts = []
#     df_cohort = df_operations.copy()

#     def record_step(description, df, reason=""):
#         count = len(df['op_id'].unique())
#         cohort_counts.append({"desc": description, "count": count, "reason": reason})
#         if df.empty:
#             print(f"Warning: Cohort empty after: '{description}'. Halting.", file=sys.stderr)
#         return df.empty

#     # --- Filtering Steps ---
#     if record_step('Total operations recorded', df_cohort): return None, cohort_counts
    
#     df_cohort.dropna(subset=['opstart_time', 'opend_time'], inplace=True)
#     df_cohort = df_cohort[df_cohort['asa'] < 6] # Exclude organ donors
#     df_cohort = df_cohort[df_cohort['department'] != 'PED'] # Exclude pediatrics
#     if record_step('Operations after excluding unrecorded start/end time', df_cohort, "Unrecorded start/end time"): return None, cohort_counts

#     df_cohort.dropna(subset=['height', 'weight'], inplace=True)
#     df_cohort = df_cohort[(df_cohort['height'] > 0) & (df_cohort['weight'] > 0)]
#     if record_step('Operations after excluding unrecorded height/weight', df_cohort, "Unrecorded height/weight"): return None, cohort_counts

#     df_cohort = df_cohort[df_cohort['antype'] != 'Regional']
#     if record_step('Operations after excluding Regional antype', df_cohort, "Regional antype"): return None, cohort_counts

#     df_cohort = df_cohort[~df_cohort['icd10_pcs'].astype(str).str.startswith(tuple(CONFIG["EXCLUSION_PREFIXES"]), na=False)]
#     if record_step('Operations after excluding specific procedures', df_cohort, "Obstetric, Kidney Donor/Recipient, and AV Fistula procedures"): return None, cohort_counts
    
#     df_cohort = pd.merge(df_cohort, df_intraop[['op_id']].drop_duplicates(), on='op_id', how='inner')
#     if record_step('Operations after excluding missing intraoperative variables', df_cohort, "No recorded intraoperative variables"): return None, cohort_counts

#     # Merge pre-processed lab data (contains pre-op and post-op creatinine values)
#     df_cohort = pd.merge(df_cohort, processed_labs_df, on='op_id', how='inner')
#     if record_step('Operations after excluding unrecorded preoperative creatinine', df_cohort, "Unrecorded preoperative creatinine"): return None, cohort_counts
    
#     # --- Permissive post-operative outcome filter ---
#     # Get dialysis (CRRT) data
#     df_dialysis = df_ward[df_ward['item_name'] == CONFIG["DIALYSIS_ITEM_NAME"]]
#     df_dialysis = df_dialysis[['subject_id', 'value']].rename(columns={'value': 'dialysis'}).drop_duplicates('subject_id')
    
#     # Merge dialysis data into the cohort to use for filtering
#     df_cohort = pd.merge(df_cohort, df_dialysis, on='subject_id', how='left')

#     # The filter condition: keep if ANY of the three post-op outcome columns are not NaN.
#     filter_condition = df_cohort[['postop_creatinine_2_days', 'postop_creatinine_7_days', 'dialysis']].notna().any(axis=1)
#     df_cohort = df_cohort[filter_condition]

#     # Drop the temporary dialysis column. It will be re-added and properly filled in calculate_aki_stages.
#     df_cohort.drop(columns=['dialysis'], inplace=True)
    
#     if record_step('Operations after excluding missing post-op outcome', df_cohort, "No post-op outcome (Cr at 2d/7d or Dialysis)"): return None, cohort_counts
    
#     # --- NOTE ON PREOPERATIVE CREATININE > 4.5 FILTER ---
#     # The following filter is part of the original CONSORT diagram's logic.
#     # However, it is NOT part of the final data pipeline that generates the training dataset of 60,824 patients.
#     # It is commented out here to ensure this script's output matches the final training data.
#     # To replicate the CONSORT diagram exactly, you can uncomment these two lines.
#     #
#     # df_cohort = df_cohort[df_cohort['preop_creatinine'] <= CONFIG["MAX_PRE_OP_SCR"]]
#     # if record_step('Operations after excluding preoperative creatinine > 4.5', df_cohort, "Preoperative creatinine > 4.5"): return None, cohort_counts

#     return df_cohort, cohort_counts

# def calculate_aki_stages(df_final_cohort, df_ward):
#     """Calculates AKI stages for the final cohort based on KDIGO criteria."""
#     df_aki = df_final_cohort.copy()
    
#     df_dialysis = df_ward[df_ward['item_name'] == CONFIG["DIALYSIS_ITEM_NAME"]]
#     df_dialysis = df_dialysis[['subject_id', 'value']].rename(columns={'value': 'dialysis'}).drop_duplicates('subject_id')
#     df_aki = pd.merge(df_aki, df_dialysis, on='subject_id', how='left').fillna({'dialysis': 0})

#     df_aki['crt_7_day_ratio'] = df_aki["postop_creatinine_7_days"] / (df_aki["preop_creatinine"] + CONFIG["EPSILON"])
    
#     # --- KDIGO Stage Definitions ---
#     aki_stage1 = ((df_aki['postop_creatinine_2_days'] - df_aki['preop_creatinine']) >= 0.3) | \
#                  ((df_aki['crt_7_day_ratio'] >= 1.5) & (df_aki['crt_7_day_ratio'] < 2))
#     aki_stage2 = (df_aki['crt_7_day_ratio'] >= 2) & (df_aki['crt_7_day_ratio'] < 3)
#     aki_stage3 = (df_aki['crt_7_day_ratio'] >= 3) | (df_aki["postop_creatinine_7_days"] >= 4) | (df_aki['dialysis'] > 0)
    
#     # Using Stage 2 or 3 for the final boolean outcome
#     df_aki['aki_boolean'] = aki_stage2 | aki_stage3
#     return df_aki


# """Main execution function to run the cohort selection pipeline."""
# df_ops, df_labs, df_ward, df_intraop = load_and_validate_data()

# processed_labs = process_labs(df_ops, df_labs)
# test(processed_labs, 'proc lab')

# # # Pass df_ward to apply_filters for the dialysis check
# # final_cohort, counts = apply_filters(df_ops, processed_labs, df_intraop, df_ward)

# # aki_df = calculate_aki_stages(final_cohort, df_ward)


# import pandas as pd
# from pathlib import Path
# import sys

# # --- Configuration Constants ---
# # Centralize clinical definitions and magic values for easy updates.
# # NOTE: Time windows are now in minutes to match the source data format.
# CONFIG = {
#     "PRE_OP_WINDOW_MINUTES": 90 * 24 * 60,  # 90 days in minutes
#     "POST_OP_WINDOW_MINUTES": 48 * 60,   # 48 hours in minutes
#     "MAX_PRE_OP_SCR": 4.5,
#     "CREATININE_ITEM_NAME": "creatinine",
#     "DIALYSIS_ITEM_NAME": "crrt",
#     "EXCLUSION_PREFIXES": ["10", "0TY", "B50", "B51"],
#     "EPSILON": 1e-9, # Small constant to prevent division by zero
#     "REQUIRED_OP_COLS": [
#         'op_id', 'subject_id', 'age', 'antype', 'asa', 'icd10_pcs',
#         'height', 'weight', 'opstart_time', 'opend_time', 'department'
#     ]
# }

# def load_and_validate_data():
#     """Loads and validates the necessary CSV files."""
#     try:
#         # Define the base paths for the data files
#         inspire_path = Path("/home/server/Projects/data/INSPIRE/physionet.org/files/inspire/1.3")
#         aki_path = Path("/home/server/Projects/data/AKI/") # Path for intraop data
        
#         # Define full paths for each required file
#         ops_path = inspire_path / "operations.csv"
#         labs_path = inspire_path / "labs.csv"
#         ward_vitals_path = inspire_path / "ward_vitals.csv"
#         intraop_path = aki_path / "feature_engineered.csv"

#         # Load the dataframes from the specified paths
#         df_operations = pd.read_csv(ops_path)
#         df_labs = pd.read_csv(labs_path)
#         df_ward = pd.read_csv(ward_vitals_path)
#         print(f"Loading intraoperative data from: {intraop_path}")
#         # Only load op_id from the feature_engineered file for the intra-op data check
#         df_intraop = pd.read_csv(intraop_path, usecols=['op_id'])

#     except FileNotFoundError as e:
#         print(f"Error: {e}. Data file not found.", file=sys.stderr)
#         return None, None, None, None

#     if not all(col in df_operations.columns for col in CONFIG["REQUIRED_OP_COLS"]):
#         print("Error: 'operations.csv' is missing required columns.", file=sys.stderr)
#         return None, None, None, None

#     # IMPORTANT: Ensure time columns are treated as numeric (minutes), not datetime objects.
#     for df, col in [(df_operations, 'opstart_time'), (df_operations, 'opend_time'), (df_labs, 'chart_time')]:
#         df[col] = pd.to_numeric(df[col], errors='coerce')

#     return df_operations, df_labs, df_ward, df_intraop

# def process_labs(df_ops, df_labs):
#     """
#     Pre-processes lab data to efficiently find required creatinine values using numerical minute-based time.
#     """
#     creatinine_labs = df_labs[df_labs['item_name'] == CONFIG["CREATININE_ITEM_NAME"]].dropna(subset=['value', 'chart_time'])
#     merged_labs = pd.merge(df_ops[['op_id', 'subject_id', 'opstart_time', 'opend_time']], creatinine_labs, on='subject_id')

#     # Pre-operative window: Find labs within 90 days (in minutes) before opstart_time
#     pre_op_mask = (merged_labs['chart_time'] <= merged_labs['opstart_time']) & \
#                   (merged_labs['chart_time'] >= merged_labs['opstart_time'] - CONFIG["PRE_OP_WINDOW_MINUTES"])
#     last_pre_op_scr = merged_labs[pre_op_mask].sort_values('chart_time').groupby('op_id').tail(1)[['op_id', 'value']]
#     last_pre_op_scr = last_pre_op_scr.rename(columns={'value': 'preop_creatinine'})
    
#     # Calculate max creatinine for 2-day and 7-day windows for AKI staging
#     post_op_2d_mask = (merged_labs['chart_time'] > merged_labs['opend_time']) & \
#                       (merged_labs['chart_time'] <= merged_labs['opend_time'] + (2 * 24 * 60))
#     post_op_7d_mask = (merged_labs['chart_time'] > merged_labs['opend_time']) & \
#                       (merged_labs['chart_time'] <= merged_labs['opend_time'] + (7 * 24 * 60))

#     max_postop_2d = merged_labs[post_op_2d_mask].groupby('op_id')['value'].max().rename('postop_creatinine_2_days')
#     max_postop_7d = merged_labs[post_op_7d_mask].groupby('op_id')['value'].max().rename('postop_creatinine_7_days')
    
#     # Use 'left' merge to keep all patients with a preop_creatinine, even if they lack post-op values.
#     # This ensures they are carried forward to the filtering step where NaNs will be evaluated.
#     processed_labs = pd.merge(last_pre_op_scr, max_postop_2d, on='op_id', how='left')
#     processed_labs = pd.merge(processed_labs, max_postop_7d, on='op_id', how='left')
    
#     return processed_labs

# def apply_filters(df_operations, processed_labs_df, df_intraop, df_ward):
#     """Applies the sequential filtering logic to the cohort."""
#     cohort_counts = []
#     df_cohort = df_operations.copy()

#     def record_step(description, df, reason=""):
#         count = len(df['op_id'].unique())
#         cohort_counts.append({"desc": description, "count": count, "reason": reason})
#         if df.empty:
#             print(f"Warning: Cohort empty after: '{description}'. Halting.", file=sys.stderr)
#         return df.empty

#     # --- Filtering Steps ---
#     if record_step('Total operations recorded', df_cohort): return None, cohort_counts
    
#     df_cohort.dropna(subset=['opstart_time', 'opend_time'], inplace=True)
#     df_cohort = df_cohort[df_cohort['asa'] < 6] # Exclude organ donors
#     df_cohort = df_cohort[df_cohort['department'] != 'PED'] # Exclude pediatrics
#     if record_step('Operations after excluding unrecorded start/end time', df_cohort, "Unrecorded start/end time"): return None, cohort_counts

#     df_cohort.dropna(subset=['height', 'weight'], inplace=True)
#     df_cohort = df_cohort[(df_cohort['height'] > 0) & (df_cohort['weight'] > 0)]
#     if record_step('Operations after excluding unrecorded height/weight', df_cohort, "Unrecorded height/weight"): return None, cohort_counts

#     df_cohort = df_cohort[df_cohort['antype'] != 'Regional']
#     if record_step('Operations after excluding Regional antype', df_cohort, "Regional antype"): return None, cohort_counts

#     df_cohort = df_cohort[~df_cohort['icd10_pcs'].astype(str).str.startswith(tuple(CONFIG["EXCLUSION_PREFIXES"]), na=False)]
#     if record_step('Operations after excluding specific procedures', df_cohort, "Obstetric, Kidney Donor/Recipient, and AV Fistula procedures"): return None, cohort_counts
    
#     df_cohort = pd.merge(df_cohort, df_intraop[['op_id']].drop_duplicates(), on='op_id', how='inner')
#     if record_step('Operations after excluding missing intraoperative variables', df_cohort, "No recorded intraoperative variables"): return None, cohort_counts

#     # Merge pre-processed lab data (contains pre-op and post-op creatinine values)
#     df_cohort = pd.merge(df_cohort, processed_labs_df, on='op_id', how='inner')
#     if record_step('Operations after excluding unrecorded preoperative creatinine', df_cohort, "Unrecorded preoperative creatinine"): return None, cohort_counts
    
#     # --- Permissive post-operative outcome filter ---
#     # Get dialysis (CRRT) data
#     df_dialysis = df_ward[df_ward['item_name'] == CONFIG["DIALYSIS_ITEM_NAME"]]
#     df_dialysis = df_dialysis[['subject_id', 'value']].rename(columns={'value': 'dialysis'}).drop_duplicates('subject_id')
    
#     # Merge dialysis data into the cohort to use for filtering
#     df_cohort = pd.merge(df_cohort, df_dialysis, on='subject_id', how='left')

#     # The filter condition: keep if ANY of the three post-op outcome columns are not NaN.
#     filter_condition = df_cohort[['postop_creatinine_2_days', 'postop_creatinine_7_days', 'dialysis']].notna().any(axis=1)
#     df_cohort = df_cohort[filter_condition]

#     # Drop the temporary dialysis column. It will be re-added and properly filled in calculate_aki_stages.
#     df_cohort.drop(columns=['dialysis'], inplace=True)
    
#     if record_step('Operations after excluding missing post-op outcome', df_cohort, "No post-op outcome (Cr at 2d/7d or Dialysis)"): return None, cohort_counts
    
#     # --- NOTE ON PREOPERATIVE CREATININE > 4.5 FILTER ---
#     # The following filter is part of the original CONSORT diagram's logic.
#     # However, it is NOT part of the final data pipeline that generates the training dataset of 60,824 patients.
#     # It is commented out here to ensure this script's output matches the final training data.
#     # To replicate the CONSORT diagram exactly, you can uncomment these two lines.
#     #
#     # df_cohort = df_cohort[df_cohort['preop_creatinine'] <= CONFIG["MAX_PRE_OP_SCR"]]
#     # if record_step('Operations after excluding preoperative creatinine > 4.5', df_cohort, "Preoperative creatinine > 4.5"): return None, cohort_counts

#     return df_cohort, cohort_counts

# def calculate_aki_stages(df_final_cohort, df_ward):
#     """Calculates AKI stages for the final cohort based on KDIGO criteria."""
#     df_aki = df_final_cohort.copy()
    
#     df_dialysis = df_ward[df_ward['item_name'] == CONFIG["DIALYSIS_ITEM_NAME"]]
#     df_dialysis = df_dialysis[['subject_id', 'value']].rename(columns={'value': 'dialysis'}).drop_duplicates('subject_id')
#     df_aki = pd.merge(df_aki, df_dialysis, on='subject_id', how='left').fillna({'dialysis': 0})

#     df_aki['crt_7_day_ratio'] = df_aki["postop_creatinine_7_days"] / (df_aki["preop_creatinine"] + CONFIG["EPSILON"])
    
#     # --- KDIGO Stage Definitions ---
#     aki_stage1 = ((df_aki['postop_creatinine_2_days'] - df_aki['preop_creatinine']) >= 0.3) | \
#                  ((df_aki['crt_7_day_ratio'] >= 1.5) & (df_aki['crt_7_day_ratio'] < 2))
#     aki_stage2 = (df_aki['crt_7_day_ratio'] >= 2) & (df_aki['crt_7_day_ratio'] < 3)
#     aki_stage3 = (df_aki['crt_7_day_ratio'] >= 3) | (df_aki["postop_creatinine_7_days"] >= 4) | (df_aki['dialysis'] > 0)
    
#     # Using Stage 2 or 3 for the final boolean outcome
#     df_aki['aki_boolean'] = aki_stage2 | aki_stage3
#     return df_aki


# """Main execution function to run the cohort selection pipeline."""
# df_ops, df_labs, df_ward, df_intraop = load_and_validate_data()


# processed_labs = process_labs(df_ops, df_labs)

# # Pass df_ward to apply_filters for the dialysis check
# final_cohort, counts = apply_filters(df_ops, processed_labs, df_intraop, df_ward)


# aki_df = calculate_aki_stages(final_cohort, df_ward)
# counts.append({"desc": "Final Cohort: Negative AKI Cases", "count": (~aki_df['aki_boolean']).sum()})
# counts.append({"desc": "Final Cohort: Positive AKI Cases", "count": aki_df['aki_boolean'].sum()})

# print("=" * 70, "\nCOHORT SELECTION RESULTS\n", "=" * 70)
# previous_count = 0
# for i, step in enumerate(counts):
#     # Special handling for the first step to not show exclusion
#     if i == 0:
#         print(f"{step['desc']:<60} n = {step['count']:,}")
#         previous_count = step['count']
#         continue

#     print(f"{step['desc']:<60} n = {step['count']:,}")
#     if "Final Cohort:" not in step['desc'] and step.get("reason"):
#         excluded = previous_count - step['count']
#         if excluded > 0:
#             print(f"   └─ Excluded: {excluded:,} ({step['reason']})")
#     previous_count = step['count']
# print("=" * 70)
