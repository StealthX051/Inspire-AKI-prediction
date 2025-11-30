# REQUIRES preopdata_file FROM EXTRACT_PREOP.PY
# EXTRACTS INTRAOP DATA TO BE COMBINED WITH PREOP DATA IN CREATE_BASE.PY

import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import entropy, kurtosis, skew
from tqdm import tqdm

inspire_path = Path("/home/server/Projects/data/INSPIRE/physionet.org/files/inspire/1.3")
vitals_file = inspire_path / "vitals.csv"
preop_csv = "/home/server/Projects/data/Multiple-Outcomes/preop_data.csv"
output_csv = '/home/server/Projects/data/Multiple-Outcomes/feature_engineered.csv'

# INTENDED TO HEREAFTER BE COMBINED WITH PREOP DATA AND CLEANED WITH CREATE_AKI_TRAINABLE.PY

# Define statistical trend and energy
def trend(y):
    x = np.arange(len(y)).T
    x = np.vstack((np.ones(len(x)), x)).T
    y = y.T
    return (np.linalg.pinv(x) @ y)[1]
def energy(x):
    return np.inner(x, x)

# Load data from CSVs
print(f"Loading Data")
df_vitals = pd.read_csv(vitals_file)
df_preop = pd.read_csv(preop_csv)

# Cut down df_vitals to only include op_ids included in df_preop
df_vitals = df_vitals[df_vitals['op_id'].isin(df_preop['op_id'].unique())]


# REGULAR data summarized with eight statistical metrics
print("Summarizing regular longitudinal data")
# Cut down df_vitals to only include item_names of high-frequency vitals
high_frequency_labels = ["rr", "hr", "spo2", "fio2", "pmean", "etco2", "peep", 
"pip", "art_mbp", "cpat", "vt", "art_sbp", "art_dbp", 
"minvol", "pplat", "bt", "etgas", "cvp"]
medium_frequency_labels = ["pap_mbp", "pap_sbp", "pap_dbp", "nibp_mbp", "nibp_dbp", "nibp_sbp"]
regular_labels = high_frequency_labels + medium_frequency_labels
df_regular = df_vitals.loc[df_vitals['item_name'].isin(regular_labels), ['op_id', 'item_name', 'value']]
# Generate a table with 24 vitals X 8 statistical metrics = 192 columns of data
print(f"Calculating Pivot Table")
df_regular = df_regular.pivot_table(
    index='op_id', 
    columns='item_name', 
    values='value', 
    aggfunc=['mean', 'max', 'min', entropy, kurtosis, skew, trend, energy]
).reset_index()
df_regular.columns = [f"{feature}_{vital}" for feature, vital in df_regular.columns]
df_regular.columns.values[0] = 'op_id'

# Generically summed data
print("Aggregating summed variables")
cross_sec_avg_labels = ["bis", "ci", "rfti", "dobui", "mlni", "ppfi", "o2", "air", "cbro2", "ntgi"]
df_cs_average = df_vitals.loc[df_vitals['item_name'].isin(cross_sec_avg_labels), ['op_id', 'item_name', 'value']]
df_cs_average = df_cs_average.pivot_table(
    index='op_id', 
    columns='item_name', 
    values='value', 
    aggfunc=['mean']
).reset_index()
df_cs_average.columns = [f"{feature}_{vital}" for feature, vital in df_cs_average.columns]
df_cs_average.columns.values[0] = 'op_id'


# TIME and WEIGHT adjusted drugs aggregated by SUM per operation
print("Aggregating time/weight adjusted variables")
wt_adjusted_labels = ["eph", "mdz", "ppf", "sft"]
df_wt_adjusted = df_vitals.loc[df_vitals['item_name'].isin(wt_adjusted_labels), ['op_id', 'item_name', 'value']]
df_wt_adjusted = df_wt_adjusted.merge(df_preop[['op_id', 'weight', 'op_len']], on='op_id', how='inner')
df_wt_adjusted['value'] = df_wt_adjusted['value'] / (df_wt_adjusted['weight'] * df_wt_adjusted['op_len'])
df_wt_adjusted = df_wt_adjusted[['op_id', 'item_name', 'value']].pivot_table(
    index='op_id', 
    columns='item_name', 
    values='value', 
    aggfunc=['sum']
).reset_index()
# df_wt_adjusted.fillna(0, inplace=True)
df_wt_adjusted.columns = [f"{feature}_{vital}" for feature, vital in df_wt_adjusted.columns]
df_wt_adjusted.columns.values[0] = 'op_id'

# TIME adjusted measures aggregated by SUM per operation
print("Aggregating time adjusted variables")
time_adjusted_labels = ["n2o", "ebl", "rbc", "uo", "ftn", "ffp", "pc", "cryo", "pheresis"]
df_time_adjusted = df_vitals.loc[df_vitals['item_name'].isin(time_adjusted_labels), ['op_id', 'item_name', 'value']]
df_time_adjusted = df_time_adjusted.merge(df_preop[['op_id', 'op_len']], on='op_id', how='inner')
df_time_adjusted['value'] = df_time_adjusted['value'] / df_time_adjusted['op_len']
df_time_adjusted = df_time_adjusted[['op_id', 'item_name', 'value']].pivot_table(
    index='op_id', 
    columns='item_name', 
    values='value', 
    aggfunc=['sum']
).reset_index()
# df_time_adjusted.fillna(0, inplace=True)
df_time_adjusted.columns = [f"{feature}_{vital}" for feature, vital in df_time_adjusted.columns]
df_time_adjusted.columns.values[0] = 'op_id'
# "n2o" L/min ??

# TIME adjusted total fluid input (summed across 10 different types)
print("Aggregating total fluid input")
fluids_agg_labels = ["d5w", "hes", "psa", "hs", "ns", "hns", "alb20", "alb5", "d10w", "d50w"]
df_fluids_agg = df_vitals.loc[df_vitals['item_name'].isin(fluids_agg_labels), ['op_id', 'item_name', 'value']]
df_fluids_agg = df_fluids_agg.groupby("op_id")['value'].sum().reset_index()
df_fluids_agg = df_fluids_agg.merge(df_preop[['op_id', 'op_len']], on='op_id', how='inner')
df_fluids_agg['fluids_agg'] = df_fluids_agg.pop('value') / df_fluids_agg.pop('op_len')


# # Desflurane and Sevoflurane interpolated by forward fill, MAC equivalent found, and then summed across operation.
print("Aggregating MAC equivalents for anesthetics")
anesthetic_labels = ['etdes', 'etsevo']
df_anesthetic = df_vitals.loc[df_vitals['item_name'].isin(anesthetic_labels), ['op_id', 'item_name', 'value', 'chart_time']]
anesth_op_ids = []
anesth_means = []
for op_id, df in tqdm(df_anesthetic.groupby('op_id')):
    end = df['chart_time'].max()
    start = df['chart_time'].min()
    times = pd.DataFrame({'chart_time': np.arange(start, end + 5, 5)})

    df_complete = pd.merge(times, df.loc[df['item_name'] == 'etdes', ['chart_time', 'value']], on='chart_time', how='left')
    df_complete = pd.merge(df_complete, df.loc[df['item_name'] == 'etsevo', ['chart_time', 'value']], on='chart_time', how='left')

    # df_complete.interpolate(''inplace=True)
    df_complete.ffill(inplace=True)
    df_complete.fillna(0, inplace=True)
    df_complete['equiv_MAC'] = (df_complete['value_x'] / 6) + (df_complete['value_y'] / 2)

    anesth_op_ids.append(op_id)
    anesth_means.append(df_complete['equiv_MAC'].mean())
df_anesthetic = pd.DataFrame({'op_id':anesth_op_ids, 'equiv_MAC_totals': anesth_means})


df_final = pd.DataFrame({'op_id': sorted(df_vitals['op_id'].unique())})
df_list = [df_regular, df_cs_average, df_wt_adjusted, df_time_adjusted, df_fluids_agg, df_anesthetic]
for df in df_list:
    df_final = df_final.merge(df, on='op_id', how='left')


df_final.to_csv(output_csv, index=False)

# REQUIRES preopdata_file FROM EXTRACT_PREOP.PY
# EXTRACTS INTRAOP DATA TO BE COMBINED WITH PREOP DATA IN CREATE_BASE.PY
#!/usr/bin/env python3
# Memory-safe, checkpointed feature engineering with automatic resume.

# import os
# import sys
# import gc
# import json
# import math
# import numpy as np
# import pandas as pd
# from pathlib import Path
# from tqdm import tqdm
# from scipy.stats import entropy as sp_entropy, kurtosis, skew

# # ----------------- Paths & config -----------------
# inspire_path = Path("/home/server/Projects/data/INSPIRE/physionet.org/files/inspire/1.3")
# vitals_file = inspire_path / "vitals.csv"
# preop_csv   = Path("/home/server/Projects/data/Multiple-Outcomes/preop_data.csv")
# out_csv     = Path("/home/server/Projects/data/Multiple-Outcomes/feature_engineered.csv")

# WORK_DIR    = Path("/home/server/Projects/data/Multiple-Outcomes/tmp_features")
# WORK_DIR.mkdir(parents=True, exist_ok=True)
# META_FILE   = WORK_DIR / "meta.json"   # stores bins and bookkeeping across runs

# # Tunables
# CHUNKSIZE   = 1_000_000            # rows per read_csv chunk
# N_BUCKETS   = 256                  # shard count to split by op_id for small working sets
# BINS        = 20                   # histogram bins for entropy
# DTYPE_MAP   = {"op_id": "int64", "item_name": "category", "value": "float32", "chart_time": "float64"}

# # ----------------- Labels -----------------
# high_freq = ["rr","hr","spo2","fio2","pmean","etco2","peep","pip","art_mbp","cpat","vt",
#              "art_sbp","art_dbp","minvol","pplat","bt","etgas","cvp"]
# med_freq  = ["pap_mbp","pap_sbp","pap_dbp","nibp_mbp","nibp_dbp","nibp_sbp"]
# regular_labels = set(high_freq + med_freq)

# cross_sec_avg_labels = ["bis","ci","rfti","dobui","mlni","ppfi","o2","air","cbro2","ntgi"]
# wt_adjusted_labels   = ["eph","mdz","ppf","sft"]
# time_adjusted_labels = ["n2o","ebl","rbc","uo","ftn","ffp","pc","cryo","pheresis"]
# fluids_agg_labels    = ["d5w","hes","psa","hs","ns","hns","alb20","alb5","d10w","d50w"]
# anesthetic_labels    = ["etdes","etsevo"]

# # ----------------- Helpers -----------------
# def log(msg): print(msg, flush=True)

# def slope_from_time(y, t):
#     y = np.asarray(y, dtype=np.float64)
#     t = np.asarray(t, dtype=np.float64)
#     n = y.size
#     if n < 2: return np.nan
#     t_sum = t.sum(); y_sum = y.sum()
#     tt_sum = np.dot(t,t); ty_sum = np.dot(t,y)
#     denom = n*tt_sum - t_sum*t_sum
#     if denom == 0: return np.nan
#     return (n*ty_sum - t_sum*y_sum) / denom

# def fast_entropy(y, edges):
#     y = np.asarray(y, dtype=float)
#     y = y[~np.isnan(y)]
#     if y.size == 0: return np.nan
#     # fixed edges -> histogram counts
#     hist, _ = np.histogram(y, bins=edges)
#     p = hist.astype(float)
#     p = p[p > 0]
#     if p.size == 0: return np.nan
#     p /= p.sum()
#     return sp_entropy(p)

# def vector_energy(y):
#     y = np.asarray(pd.Series(y).fillna(0.0), dtype=float)
#     return float(np.inner(y, y))

# def bucket_dir(name):
#     d = WORK_DIR / name
#     d.mkdir(exist_ok=True)
#     return d

# def shard_id(op_id):  # stable shard assignment
#     return int(op_id) % N_BUCKETS

# def save_meta(meta):
#     with open(META_FILE, "w") as f:
#         json.dump(meta, f)

# def load_meta():
#     if META_FILE.exists():
#         with open(META_FILE) as f:
#             return json.load(f)
#     return {}

# # ----------------- Stage 0: Preflight & bins (pilot pass for entropy) -----------------
# def stage0_prepare_bins_and_shards():
#     """
#     1) Load preop to filter op_ids.
#     2) Pilot pass (small fraction of file) to get per-item_name min/max for entropy bins.
#     3) Save item->edges into meta.
#     """
#     meta = load_meta()
#     if meta.get("bins_ready"):
#         log("Stage0: bins already prepared. Skipping.")
#         return

#     log("Loading preop...")
#     df_preop = pd.read_csv(preop_csv, usecols=["op_id","weight","op_len"])
#     op_set = set(df_preop["op_id"].astype("int64").tolist())
#     meta["preop_present"] = True
#     save_meta(meta)

#     log("Pilot pass to infer entropy bin edges (per item_name)...")
#     mins = {}
#     maxs = {}

#     # Only regular labels need entropy
#     needed_items = regular_labels

#     # Read first ~5M rows or whole file if smaller
#     rows_read = 0
#     ROW_LIMIT = 5_000_000
#     for chunk in pd.read_csv(
#         vitals_file,
#         usecols=["op_id","item_name","value"],
#         chunksize=CHUNKSIZE,
#         dtype={"op_id":"int64","item_name":"category","value":"float64"}
#     ):
#         # filter to relevant ops and items
#         chunk = chunk[chunk["op_id"].isin(op_set)]
#         chunk = chunk[chunk["item_name"].isin(needed_items)]
#         if chunk.empty:
#             rows_read += len(chunk)
#             if rows_read >= ROW_LIMIT: break
#             continue
#         # update mins/maxs
#         for name, g in chunk.groupby("item_name"):
#             v = g["value"].to_numpy()
#             v = v[~np.isnan(v)]
#             if v.size == 0: continue
#             mn, mx = float(np.min(v)), float(np.max(v))
#             if name not in mins: mins[name] = mn
#             else: mins[name] = min(mins[name], mn)
#             if name not in maxs: maxs[name] = mx
#             else: maxs[name] = max(maxs[name], mx)

#         rows_read += len(chunk)
#         if rows_read >= ROW_LIMIT:
#             break

#     # define edges (guard degenerate ranges)
#     item_edges = {}
#     for name in needed_items:
#         mn = mins.get(name, -1.0)
#         mx = maxs.get(name,  1.0)
#         if not np.isfinite(mn) or not np.isfinite(mx) or mn == mx:
#             mn, mx = (mn if np.isfinite(mn) else -1.0), (mx if np.isfinite(mx) else 1.0)
#             mx = mn + 1.0 if mn == mx else mx
#         item_edges[name] = list(np.linspace(mn, mx, BINS+1).astype(float))

#     meta["item_edges"] = item_edges
#     meta["bins_ready"] = True
#     meta["regular_done"] = False
#     meta["cs_done"] = False
#     meta["wt_done"] = False
#     meta["time_done"] = False
#     meta["fluids_done"] = False
#     meta["mac_done"] = False
#     save_meta(meta)
#     log("Stage0 complete: saved bin edges for entropy.")

# # ----------------- Stage 1: Shard raw vitals by block to disk -----------------
# def stage1_shard_raw():
#     """
#     Shard rows by block into bucketed parquet *part files* for small working sets.
#     No append: each write creates a new file, e.g., bucket_007_part000123.parquet
#     """
#     meta = load_meta()
#     if meta.get("raw_sharded"):
#         log("Stage1: raw shards already exist. Skipping.")
#         return

#     df_preop = pd.read_csv(preop_csv, usecols=["op_id"])
#     op_set = set(df_preop["op_id"].astype("int64").tolist())

#     # Create shard dirs
#     dirs = {
#         "regular": bucket_dir("shard_regular"),
#         "cs": bucket_dir("shard_cs"),
#         "wt": bucket_dir("shard_wt"),
#         "time": bucket_dir("shard_time"),
#         "fluids": bucket_dir("shard_fluids"),
#         "mac": bucket_dir("shard_mac"),
#     }

#     log("Sharding vitals.csv into on-disk buckets (no appends; writing parts)...")
#     part_id = 0  # monotonically increasing part counter across all buckets

#     for chunk in tqdm(pd.read_csv(
#         vitals_file,
#         usecols=["op_id","item_name","value","chart_time"],
#         chunksize=CHUNKSIZE,
#         dtype=DTYPE_MAP
#     )):
#         chunk = chunk[chunk["op_id"].isin(op_set)]
#         if chunk.empty:
#             continue

#         # helper to write a subframe into a uniquely named part file
#         def _write_parts(df_sub, shard_key):
#             nonlocal part_id
#             if df_sub.empty:
#                 return
#             # group by bucket id and write a unique part per (bucket, this call)
#             for b, sub in df_sub.groupby(df_sub["op_id"].map(shard_id), sort=False):
#                 fname = dirs[shard_key] / f"bucket_{b:03d}_part{part_id:06d}.parquet"
#                 sub.to_parquet(fname, index=False, engine="pyarrow", compression="snappy")
#                 part_id += 1

#         # REGULAR shards
#         _write_parts(chunk[chunk["item_name"].isin(regular_labels)][["op_id","item_name","value","chart_time"]], "regular")

#         # CS avg
#         _write_parts(chunk[chunk["item_name"].isin(cross_sec_avg_labels)][["op_id","item_name","value"]], "cs")

#         # WT adj
#         _write_parts(chunk[chunk["item_name"].isin(wt_adjusted_labels)][["op_id","item_name","value"]], "wt")

#         # TIME adj
#         _write_parts(chunk[chunk["item_name"].isin(time_adjusted_labels)][["op_id","item_name","value"]], "time")

#         # FLUIDS
#         _write_parts(chunk[chunk["item_name"].isin(fluids_agg_labels)][["op_id","value"]], "fluids")

#         # MAC
#         _write_parts(chunk[chunk["item_name"].isin(anesthetic_labels)][["op_id","item_name","value","chart_time"]], "mac")

#         del chunk
#         gc.collect()

#     meta["raw_sharded"] = True
#     save_meta(meta)
#     log("Stage1 complete.")


# # ----------------- Stage 2: Compute each block per bucket (resume-safe) -----------------
# def compute_regular():
#     meta = load_meta()
#     if meta.get("regular_done"):
#         log("Regular stats already complete. Skipping.")
#         return

#     edges = {k: np.array(v, dtype=float) for k, v in meta["item_edges"].items()}
#     out_dir = bucket_dir("regular_out")

#     # For each bucket file, compute stats in-memory (small file), write partial parquet
#     shard_dir = bucket_dir("shard_regular")
#     files = sorted(shard_dir.glob("bucket_*.parquet"))
#     if not files:
#         # no data for regular block
#         (out_dir / "regular_all.parquet").touch()
#         meta["regular_done"] = True
#         save_meta(meta)
#         return

#     for f in tqdm(files, desc="regular buckets"):
#         outf = out_dir / f"{f.stem}_stats.parquet"
#         if outf.exists():
#             continue
#         df = pd.read_parquet(f, engine="pyarrow")
#         if df.empty:
#             outf.touch(); continue

#         # group and compute stats for each (op_id,item)
#         parts = []
#         for (op, name), g in df.groupby(["op_id","item_name"], sort=False):
#             v = g["value"].to_numpy()
#             t = g["chart_time"].to_numpy()
#             # stats
#             m = np.nanmean(v) if v.size else np.nan
#             mx = np.nanmax(v) if v.size else np.nan
#             mn = np.nanmin(v) if v.size else np.nan
#             sk = skew(v, bias=False, nan_policy='omit') if v.size >= 3 else np.nan
#             ku = kurtosis(v, fisher=True, bias=False, nan_policy='omit') if v.size >= 4 else np.nan
#             en = vector_energy(v)
#             tr = slope_from_time(v, t)
#             ent = fast_entropy(v, edges.get(name, np.linspace(np.nanmin(v), np.nanmax(v)+1e-9, BINS+1))) if v.size else np.nan

#             parts.append({
#                 "op_id": op,
#                 "item_name": str(name),
#                 "mean": m,
#                 "max": mx,
#                 "min": mn,
#                 "skew": sk,
#                 "kurtosis": ku,
#                 "entropy": ent,
#                 "trend": tr,
#                 "energy": en
#             })
#         if parts:
#             pd.DataFrame(parts).to_parquet(outf, index=False, engine="pyarrow", compression="snappy")
#         else:
#             outf.touch()
#         del df, parts
#         gc.collect()

#     # stitch all partials, then unstack to wide
#     log("Stitching regular partials...")
#     frames = []
#     for f in sorted(out_dir.glob("bucket_*_stats.parquet")):
#         if f.stat().st_size == 0: continue
#         frames.append(pd.read_parquet(f, engine="pyarrow"))
#     if frames:
#         REG = pd.concat(frames, ignore_index=True)
#         REG = REG.pivot_table(index="op_id", columns="item_name",
#                               values=["mean","max","min","skew","kurtosis","entropy","trend","energy"])
#         REG.columns = [f"{feat}_{vital}" for feat, vital in REG.columns]
#         REG = REG.reset_index()
#         REG.to_parquet(out_dir / "regular_all.parquet", index=False, engine="pyarrow", compression="snappy")
#     else:
#         pd.DataFrame({"op_id":[]}).to_parquet(out_dir / "regular_all.parquet", index=False, engine="pyarrow")

#     meta["regular_done"] = True
#     save_meta(meta)

# def compute_cs_average():
#     meta = load_meta()
#     if meta.get("cs_done"):
#         log("CS averages already complete. Skipping.")
#         return

#     out_dir = bucket_dir("cs_out")
#     shard_dir = bucket_dir("shard_cs")

#     for f in tqdm(sorted(shard_dir.glob("bucket_*.parquet")), desc="cs buckets"):
#         outf = out_dir / f"{f.stem}_cs.parquet"
#         if outf.exists(): continue
#         df = pd.read_parquet(f, engine="pyarrow")
#         if df.empty: outf.touch(); continue
#         g = df.groupby(["op_id","item_name"])["value"].mean().unstack("item_name")
#         g = g.add_prefix("mean_").reset_index()
#         g.to_parquet(outf, index=False, engine="pyarrow", compression="snappy")
#         del df, g; gc.collect()

#     # stitch
#     frames = [pd.read_parquet(f, engine="pyarrow") for f in sorted(out_dir.glob("bucket_*_cs.parquet")) if f.stat().st_size>0]
#     if frames:
#         CS = pd.concat(frames, ignore_index=True).groupby("op_id", as_index=False).first()
#         CS.to_parquet(out_dir / "cs_all.parquet", index=False, engine="pyarrow", compression="snappy")
#     else:
#         pd.DataFrame({"op_id":[]}).to_parquet(out_dir / "cs_all.parquet", index=False, engine="pyarrow")

#     meta["cs_done"] = True
#     save_meta(meta)

# def compute_wt_adjusted():
#     meta = load_meta()
#     if meta.get("wt_done"):
#         log("WT adjusted already complete. Skipping.")
#         return

#     out_dir = bucket_dir("wt_out")
#     shard_dir = bucket_dir("shard_wt")

#     df_pre = pd.read_csv(preop_csv, usecols=["op_id","weight","op_len"])
#     for f in tqdm(sorted(shard_dir.glob("bucket_*.parquet")), desc="wt buckets"):
#         outf = out_dir / f"{f.stem}_wt.parquet"
#         if outf.exists(): continue
#         df = pd.read_parquet(f, engine="pyarrow")
#         if df.empty: outf.touch(); continue
#         g = df.groupby(["op_id","item_name"])["value"].sum().unstack("item_name").reset_index()
#         g = g.merge(df_pre, on="op_id", how="inner")
#         scale = (g["weight"] * g["op_len"]).replace(0, np.nan)
#         for col in [c for c in g.columns if c in wt_adjusted_labels]:
#             g[col] = g[col] / scale
#         WT = g.drop(columns=["weight","op_len"]).add_prefix("sum_")
#         WT.rename(columns={"sum_op_id":"op_id"}, inplace=True)
#         WT.to_parquet(outf, index=False, engine="pyarrow", compression="snappy")
#         del df, g, WT; gc.collect()

#     frames = [pd.read_parquet(f, engine="pyarrow") for f in sorted(out_dir.glob("bucket_*_wt.parquet")) if f.stat().st_size>0]
#     if frames:
#         WTALL = pd.concat(frames, ignore_index=True).groupby("op_id", as_index=False).first()
#         WTALL.to_parquet(out_dir / "wt_all.parquet", index=False, engine="pyarrow", compression="snappy")
#     else:
#         pd.DataFrame({"op_id":[]}).to_parquet(out_dir / "wt_all.parquet", index=False, engine="pyarrow")

#     meta["wt_done"] = True
#     save_meta(meta)

# def compute_time_adjusted():
#     meta = load_meta()
#     if meta.get("time_done"):
#         log("TIME adjusted already complete. Skipping.")
#         return

#     out_dir = bucket_dir("time_out")
#     shard_dir = bucket_dir("shard_time")

#     df_pre = pd.read_csv(preop_csv, usecols=["op_id","op_len"])
#     for f in tqdm(sorted(shard_dir.glob("bucket_*.parquet")), desc="time buckets"):
#         outf = out_dir / f"{f.stem}_time.parquet"
#         if outf.exists(): continue
#         df = pd.read_parquet(f, engine="pyarrow")
#         if df.empty: outf.touch(); continue
#         g = df.groupby(["op_id","item_name"])["value"].sum().unstack("item_name").reset_index()
#         g = g.merge(df_pre, on="op_id", how="inner")
#         op_len = g["op_len"].replace(0, np.nan)
#         for col in [c for c in g.columns if c in time_adjusted_labels]:
#             g[col] = g[col] / op_len
#         TA = g.drop(columns=["op_len"]).add_prefix("sum_")
#         TA.rename(columns={"sum_op_id":"op_id"}, inplace=True)
#         TA.to_parquet(outf, index=False, engine="pyarrow", compression="snappy")
#         del df, g, TA; gc.collect()

#     frames = [pd.read_parquet(f, engine="pyarrow") for f in sorted(out_dir.glob("bucket_*_time.parquet")) if f.stat().st_size>0]
#     if frames:
#         TAALL = pd.concat(frames, ignore_index=True).groupby("op_id", as_index=False).first()
#         TAALL.to_parquet(out_dir / "time_all.parquet", index=False, engine="pyarrow", compression="snappy")
#     else:
#         pd.DataFrame({"op_id":[]}).to_parquet(out_dir / "time_all.parquet", index=False, engine="pyarrow")

#     meta["time_done"] = True
#     save_meta(meta)

# def compute_fluids():
#     meta = load_meta()
#     if meta.get("fluids_done"):
#         log("Fluids agg already complete. Skipping.")
#         return

#     out_dir = bucket_dir("fluids_out")
#     shard_dir = bucket_dir("shard_fluids")

#     df_pre = pd.read_csv(preop_csv, usecols=["op_id","op_len"])
#     for f in tqdm(sorted(shard_dir.glob("bucket_*.parquet")), desc="fluids buckets"):
#         outf = out_dir / f"{f.stem}_fluids.parquet"
#         if outf.exists(): continue
#         df = pd.read_parquet(f, engine="pyarrow")
#         if df.empty: outf.touch(); continue
#         g = df.groupby("op_id")["value"].sum().reset_index()
#         g = g.merge(df_pre, on="op_id", how="inner")
#         FA = pd.DataFrame({"op_id": g["op_id"], "fluids_agg": g["value"] / g["op_len"].replace(0, np.nan)})
#         FA.to_parquet(outf, index=False, engine="pyarrow", compression="snappy")
#         del df, g, FA; gc.collect()

#     frames = [pd.read_parquet(f, engine="pyarrow") for f in sorted(out_dir.glob("bucket_*_fluids.parquet")) if f.stat().st_size>0]
#     if frames:
#         FAALL = pd.concat(frames, ignore_index=True).groupby("op_id", as_index=False).first()
#         FAALL.to_parquet(out_dir / "fluids_all.parquet", index=False, engine="pyarrow", compression="snappy")
#     else:
#         pd.DataFrame({"op_id":[]}).to_parquet(out_dir / "fluids_all.parquet", index=False, engine="pyarrow")

#     meta["fluids_done"] = True
#     save_meta(meta)

# def compute_mac():
#     meta = load_meta()
#     if meta.get("mac_done"):
#         log("MAC already complete. Skipping.")
#         return

#     out_dir = bucket_dir("mac_out")
#     shard_dir = bucket_dir("shard_mac")

#     for f in tqdm(sorted(shard_dir.glob("bucket_*.parquet")), desc="mac buckets"):
#         outf = out_dir / f"{f.stem}_mac.parquet"
#         if outf.exists(): continue
#         df = pd.read_parquet(f, engine="pyarrow")
#         if df.empty: outf.touch(); continue
#         # pivot by time within bucket, ffill by op_id
#         df = (df.sort_values(["op_id","chart_time"])
#                 .pivot_table(index=["op_id","chart_time"], columns="item_name", values="value", aggfunc="last")
#                 .reset_index())
#         for col in anesthetic_labels:
#             if col not in df.columns: df[col] = np.nan
#         df[anesthetic_labels] = df.groupby("op_id", group_keys=False)[anesthetic_labels].apply(lambda d: d.ffill())
#         # interval dt
#         df["dt"] = df.groupby("op_id")["chart_time"].shift(-1) - df["chart_time"]
#         df = df[df["dt"].notna()]
#         etdes = df["etdes"].fillna(0).to_numpy()
#         etsevo = df["etsevo"].fillna(0).to_numpy()
#         df["equiv_MAC"] = (etdes/6.0) + (etsevo/2.0)
#         MAC = (df.groupby("op_id")
#                  .apply(lambda d: (d["equiv_MAC"]*d["dt"]).sum() / d["dt"].sum())
#                  .rename("equiv_MAC_totals").reset_index())
#         MAC.to_parquet(outf, index=False, engine="pyarrow", compression="snappy")
#         del df, MAC; gc.collect()

#     frames = [pd.read_parquet(f, engine="pyarrow") for f in sorted(out_dir.glob("bucket_*_mac.parquet")) if f.stat().st_size>0]
#     if frames:
#         MACALL = pd.concat(frames, ignore_index=True).groupby("op_id", as_index=False).first()
#         MACALL.to_parquet(out_dir / "mac_all.parquet", index=False, engine="pyarrow", compression="snappy")
#     else:
#         pd.DataFrame({"op_id":[]}).to_parquet(out_dir / "mac_all.parquet", index=False, engine="pyarrow")

#     meta["mac_done"] = True
#     save_meta(meta)

# # ----------------- Stage 3: Merge all blocks and write CSV -----------------
# def stage3_merge_and_save():
#     log("Merging blocks...")
#     # start with all op_ids from preop filter
#     base_ops = pd.read_csv(preop_csv, usecols=["op_id"]).drop_duplicates()
#     base_ops["op_id"] = base_ops["op_id"].astype("int64")
#     df_final = base_ops.copy()

#     # load block results if present
#     paths = {
#         "regular": WORK_DIR / "regular_out" / "regular_all.parquet",
#         "cs":      WORK_DIR / "cs_out"      / "cs_all.parquet",
#         "wt":      WORK_DIR / "wt_out"      / "wt_all.parquet",
#         "time":    WORK_DIR / "time_out"    / "time_all.parquet",
#         "fluids":  WORK_DIR / "fluids_out"  / "fluids_all.parquet",
#         "mac":     WORK_DIR / "mac_out"     / "mac_all.parquet",
#     }
#     for name, p in paths.items():
#         if p.exists() and p.stat().st_size > 0:
#             log(f"  + merging {name}")
#             dfp = pd.read_parquet(p, engine="pyarrow")
#             df_final = df_final.merge(dfp, on="op_id", how="left", copy=False)

#     # Save final CSV
#     df_final.to_csv(out_csv, index=False)
#     log(f"Wrote features to {out_csv}")

# # ----------------- Main -----------------
# def main():
#     log("=== Feature engineering with checkpoints ===")
#     stage0_prepare_bins_and_shards()  # bins + meta
#     stage1_shard_raw()                # one-time sharding
#     compute_regular()
#     compute_cs_average()
#     compute_wt_adjusted()
#     compute_time_adjusted()
#     compute_fluids()
#     compute_mac()
#     stage3_merge_and_save()
#     log("=== Done ===")

# if __name__ == "__main__":
#     main()
