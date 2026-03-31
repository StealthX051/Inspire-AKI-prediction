"""
Imputation-SHAP Cross-Reference Analysis
=========================================
Addresses Reviewer 2, Comment 2:
  "The authors should verify that -99-imputed features are not among the
   dominant SHAP contributors."

This script:
  1. Loads fill rates (from the preprocessing pipeline output)
  2. Loads SHAP importance rankings (from batch SHAP pkl files)
  3. Flags features that are both high-missing (>=10%) AND high-SHAP
  4. Outputs a formatted summary table

Usage:
  python imputation_shap_cross_reference.py

Paths assume the AKI data server layout; adjust BASE_DATA_DIR as needed.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

HIGH_MISSING_THRESHOLD_PCT = 10.0   # matches config: high_missing_threshold_pct
TOP_N_SHAP = 20                     # number of top SHAP features to inspect

BASE_DATA_DIR = Path("/home/server/Projects/data/AKI/")
RESULTS_DIR = BASE_DATA_DIR / "results" / "batch_shap_analysis"
FILL_RATES_CSV = BASE_DATA_DIR / "fill_rates.csv"   # produced by pipeline preprocess step

# Models to analyse (must match the pkl naming in batch SHAP notebook)
SHAP_JOBS = [
    "XGBoost_Combined",
    "XGBoost_Preop",
    "XGBoost_Intraop",
    "RandomForest_Combined",
    "RandomForest_Preop",
    "RandomForest_Intraop",
]

# ---------------------------------------------------------------------------
# Hardcoded fill rates from fill_rate_table.html (fallback if CSV unavailable)
# These are the actual fill rates from the published analysis.
# ---------------------------------------------------------------------------

FILL_RATES_HARDCODED = {
    "preop_creatinine": 100.00, "last_preop_scr": 100.00, "preop_scr": 100.00,
    "hr": 99.72, "preop_potassium": 98.04, "preop_sodium": 98.04,
    "preop_hct": 97.96, "preop_hb": 97.46, "preop_wbc": 97.33,
    "preop_platelet": 97.18, "preop_chloride": 97.14, "fluids_agg": 96.83,
    "preop_bun": 96.79, "preop_calcium": 96.62, "preop_albumin": 96.51,
    "etco2": 96.34, "preop_ast": 96.25, "preop_alt": 96.23,
    "preop_total_protein": 96.16, "preop_alp": 96.08, "preop_ptinr": 95.83,
    "preop_aptt": 95.46, "preop_phosphorus": 95.13, "preop_glucose": 94.14,
    "preop_total_bilirubin": 93.30, "spo2": 92.78, "preop_lymphocyte": 92.48,
    "nibp_mbp": 92.48, "preop_seg": 92.47, "nibp_dbp": 92.40,
    "nibp_sbp": 92.27, "o2": 91.90, "rr": 90.77,
    # --- >10% missing below: these receive -99 sentinel imputation ---
    "fio2": 85.96, "preop_fibrinogen": 84.46, "bt": 83.64,
    "pip": 82.77, "vt": 82.33, "minvol": 80.11, "air": 79.16,
    "preop_crp": 78.45, "ebl": 73.42, "uo": 68.43, "pmean": 67.99,
    "art_mbp": 64.02, "art_sbp": 63.95, "art_dbp": 63.94,
    "bis": 62.12, "peep": 59.37, "pplat": 47.40, "cpat": 44.66,
    "equiv_MAC_totals": 44.33, "eph": 41.68, "etgas": 39.13,
    "ppf": 38.41, "rfti": 24.81, "ftn": 16.74, "ppfi": 16.06,
    "rbc": 12.76, "ci": 11.46,
    # The following have very high missingness (>90% missing)
    "cvp": 8.33, "mdz": 6.74, "pap_mbp": 4.53, "pap_sbp": 4.46,
    "pap_dbp": 4.43, "ffp": 4.23, "ntgi": 3.40, "sft": 2.81,
    "dobui": 2.59, "pc": 2.36, "n2o": 0.99, "cryo": 0.95,
    "mlni": 0.47, "cbro2": 0.30, "pheresis": 0.15,
}

# Derived intraop feature base names that map to the fill rates above
# (intraop features are stat aggregations like mean_art_mbp, max_uo, etc.)
INTRAOP_BASE_MAP = {
    base: fill_rate
    for base, fill_rate in FILL_RATES_HARDCODED.items()
    if fill_rate < (100.0 - HIGH_MISSING_THRESHOLD_PCT)
}


def get_base_feature(feature_name: str, fill_rates: dict[str, float]) -> str | None:
    """
    For an intraop derived feature like 'mean_art_mbp', return the base
    variable name 'art_mbp' if it exists in fill_rates, else return None.
    For preop features like 'preop_albumin', return as-is if in fill_rates.
    """
    if feature_name in fill_rates:
        return feature_name
    # Try stripping common stat prefixes
    for prefix in ("mean_", "max_", "min_", "entropy_", "kurtosis_",
                   "skew_", "trend_", "energy_", "sum_", "std_"):
        if feature_name.startswith(prefix):
            base = feature_name[len(prefix):]
            if base in fill_rates:
                return base
    return None


def load_fill_rates() -> pd.DataFrame:
    if FILL_RATES_CSV.exists():
        fr = pd.read_csv(FILL_RATES_CSV)
        fr["missing_pct"] = (1 - fr["fill_rate"]) * 100
        fr["imputed_with_neg99"] = fr["missing_pct"] >= HIGH_MISSING_THRESHOLD_PCT
        return fr
    # Fallback to hardcoded values
    rows = [
        {"feature": k, "fill_rate_pct": v, "missing_pct": 100 - v,
         "imputed_with_neg99": (100 - v) >= HIGH_MISSING_THRESHOLD_PCT}
        for k, v in FILL_RATES_HARDCODED.items()
    ]
    return pd.DataFrame(rows)


def load_shap_importance(job_name: str) -> pd.DataFrame | None:
    pkl_path = RESULTS_DIR / f"{job_name}_shap_explanation.pkl"
    if not pkl_path.exists():
        print(f"  [SKIP] {pkl_path} not found.")
        return None
    with open(pkl_path, "rb") as f:
        explanation = pickle.load(f)
    # shap.Explanation object
    shap_values = np.abs(explanation.values)  # shape: (n_samples, n_features)
    mean_abs_shap = shap_values.mean(axis=0)
    importance = pd.DataFrame({
        "feature": explanation.feature_names,
        "mean_abs_shap": mean_abs_shap,
    }).sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    importance["rank"] = importance.index + 1
    return importance


def build_cross_reference_table(
    importance: pd.DataFrame,
    fill_rates: pd.DataFrame,
    top_n: int = TOP_N_SHAP,
) -> pd.DataFrame:
    top_features = importance.head(top_n).copy()
    fr_lookup = dict(zip(fill_rates["feature"], fill_rates["fill_rate_pct"]
                         if "fill_rate_pct" in fill_rates.columns
                         else fill_rates["fill_rate"] * 100))
    imputed_lookup = dict(zip(fill_rates["feature"], fill_rates["imputed_with_neg99"]))

    rows = []
    for _, row in top_features.iterrows():
        fname = row["feature"]
        base = get_base_feature(fname, fr_lookup)
        fill_rate = fr_lookup.get(base, float("nan")) if base else float("nan")
        imputed = imputed_lookup.get(base, False) if base else False
        rows.append({
            "rank": row["rank"],
            "feature": fname,
            "mean_abs_shap": round(row["mean_abs_shap"], 5),
            "base_variable": base or "—",
            "fill_rate_pct": round(fill_rate, 1) if not np.isnan(fill_rate) else "—",
            "missing_pct": round(100 - fill_rate, 1) if not np.isnan(fill_rate) else "—",
            "imputed_with_neg99": "YES ⚠" if imputed else "no",
        })
    return pd.DataFrame(rows)


def main() -> None:
    print("=" * 70)
    print("Imputation-SHAP Cross-Reference Analysis")
    print(f"Threshold for -99 imputation: >{HIGH_MISSING_THRESHOLD_PCT}% missing")
    print("=" * 70)

    fill_rates = load_fill_rates()
    neg99_features = fill_rates[fill_rates["imputed_with_neg99"]]["feature"].tolist()
    print(f"\nFeatures receiving -99 sentinel imputation ({len(neg99_features)} total):")
    for f in sorted(neg99_features):
        fr = fill_rates.loc[fill_rates["feature"] == f, "fill_rate_pct"].values[0] \
            if "fill_rate_pct" in fill_rates.columns \
            else (1 - fill_rates.loc[fill_rates["feature"] == f, "fill_rate"].values[0]) * 100
        print(f"  {f:30s}  fill={100 - fr:.1f}% missing")

    all_results = []
    for job in SHAP_JOBS:
        print(f"\n{'=' * 60}")
        print(f"Job: {job}")
        print(f"{'=' * 60}")
        importance = load_shap_importance(job)
        if importance is None:
            continue
        table = build_cross_reference_table(importance, fill_rates, top_n=TOP_N_SHAP)
        print(table.to_string(index=False))
        flagged = table[table["imputed_with_neg99"] == "YES ⚠"]
        if flagged.empty:
            print(f"\n✓ No -99-imputed features in top {TOP_N_SHAP} SHAP for {job}.")
        else:
            print(f"\n⚠ {len(flagged)} -99-imputed feature(s) in top {TOP_N_SHAP} SHAP:")
            for _, r in flagged.iterrows():
                print(f"  Rank {r['rank']:2d}: {r['feature']}  "
                      f"(fill={r['fill_rate_pct']}%, SHAP={r['mean_abs_shap']})")
        table["job"] = job
        all_results.append(table)

    if all_results:
        combined = pd.concat(all_results, ignore_index=True)
        out_path = RESULTS_DIR / "imputation_shap_crossref.csv"
        combined.to_csv(out_path, index=False)
        print(f"\nFull cross-reference saved to: {out_path}")

        # Summary: which -99-imputed features appear across models?
        flagged_all = combined[combined["imputed_with_neg99"] == "YES ⚠"]
        if not flagged_all.empty:
            print("\n--- Summary: -99-imputed features appearing in top SHAP ---")
            summary = (
                flagged_all.groupby("feature")
                .agg(
                    times_in_top20=("rank", "count"),
                    best_rank=("rank", "min"),
                    fill_rate_pct=("fill_rate_pct", "first"),
                )
                .sort_values("best_rank")
                .reset_index()
            )
            print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
