import pandas as pd
from sklearn.preprocessing import StandardScaler

df = pd.read_csv('/home/server/Projects/data/AKI/aki_data_trainable.csv')
cols_to_norm = ['age', 'height', 'weight', 'BSA', 'BMI', 'booking_case_length', 'num_card_events',  'last_preop_scr',
       'min_preop_scr', 'preop_total_protein', 'preop_sodium',
       'preop_potassium', 'preop_platelet', 'preop_glucose', 'preop_wbc',
       'preop_alt', 'preop_chloride', 'preop_lymphocyte', 'preop_phosphorus',
       'preop_albumin', 'preop_fibrinogen', 'preop_creatinine', 'preop_ptinr',
       'preop_total_bilirubin', 'preop_alp', 'preop_aptt', 'preop_calcium',
       'preop_bun', 'preop_ast', 'preop_crp', 'preop_hb', 'preop_hct',
       'preop_seg', 'op_len']

scaler = StandardScaler()
df[cols_to_norm] = scaler.fit_transform(df[cols_to_norm])

df.to_csv("/home/server/Projects/data/AKI/aki_data_normalized.csv", index=False)
