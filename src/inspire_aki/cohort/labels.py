from __future__ import annotations

from pathlib import Path

import pandas as pd

from inspire_aki.cohort.audit import record_count


def derive_aki_labels(
    *,
    config: dict,
    raw_inspire_dir: Path,
    preop_df: pd.DataFrame,
    tabular_combined_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    cohort_cfg = config["cohort"]
    labs_path = raw_inspire_dir / "labs.csv"
    ward_vitals_path = raw_inspire_dir / "ward_vitals.csv"

    df_ward = pd.read_csv(ward_vitals_path)
    df_dialysis = df_ward[df_ward["item_name"] == cohort_cfg["dialysis_item_name"]][["subject_id", "value"]]
    df_dialysis = df_dialysis.rename(columns={"value": "dialysis"})
    df_dialysis = df_dialysis.drop_duplicates(subset="subject_id", keep="first").reset_index(drop=True)

    cols_to_keep = ["op_id", "subject_id", "preop_creatinine", "opend_time"]
    df_preop = preop_df[cols_to_keep].copy()
    df_labs = pd.read_csv(labs_path)
    base_combined = tabular_combined_df[["op_id"]].copy()
    df_preop = df_preop.merge(base_combined, on="op_id", how="inner")

    audit: list[dict] = []
    record_count(audit, "tabular_ops_before_labels", df_preop)
    df_preop = df_preop[df_preop["preop_creatinine"].notna()]
    record_count(audit, "has_preop_creatinine", df_preop)
    df_preop = df_preop[df_preop["preop_creatinine"] < cohort_cfg["max_preop_creatinine"]]
    record_count(audit, "preop_creatinine_lt_threshold", df_preop)

    df_creatinine = df_labs[df_labs["item_name"] == cohort_cfg["creatinine_item_name"]]
    df_merge = pd.merge(df_preop, df_creatinine, on="subject_id")

    df_aki = df_preop.copy().merge(df_dialysis, on="subject_id", how="left")
    for n_days in cohort_cfg["postop_windows_days"]:
        n_minutes = n_days * 24 * 60
        df_merge_filtered = df_merge[
            (df_merge["chart_time"] > df_merge["opend_time"])
            & (df_merge["chart_time"] <= (df_merge["opend_time"] + n_minutes))
        ]
        max_creatinine = (
            df_merge_filtered.groupby("op_id")["value"]
            .max()
            .reset_index()
            .rename(columns={"value": f"postop_creatinine_{n_days}_days"})
        )
        df_aki = pd.merge(df_aki, max_creatinine, on="op_id", how="outer")

    cols = ["postop_creatinine_2_days", "postop_creatinine_7_days", "dialysis"]
    df_aki = df_aki[~df_aki[cols].isna().all(axis=1)]
    df_aki = df_aki.fillna({"dialysis": 0})
    record_count(audit, "has_postop_creatinine_or_dialysis", df_aki)

    df_aki["crt_7_day_ratio"] = df_aki["postop_creatinine_7_days"] / df_aki["preop_creatinine"]
    df_aki["aki_1"] = (
        ((df_aki["crt_7_day_ratio"] > 1.5) & (df_aki["crt_7_day_ratio"] < 2))
        | ((df_aki["postop_creatinine_2_days"] - df_aki["preop_creatinine"]) > 0.3)
    )
    df_aki["aki_2"] = (df_aki["crt_7_day_ratio"] >= 2) & (df_aki["crt_7_day_ratio"] < 3)
    df_aki["aki_3"] = (
        (df_aki["crt_7_day_ratio"] >= 3)
        | (df_aki["postop_creatinine_7_days"] > 4)
        | (df_aki["dialysis"] > 0)
    )
    df_aki["aki_boolean"] = df_aki[["aki_2", "aki_3"]].any(axis=1).astype(bool)

    labels = df_aki[["op_id", "aki_boolean"]].copy()
    record_count(audit, "final_labeled_ops", labels)
    return labels, pd.DataFrame(audit)
