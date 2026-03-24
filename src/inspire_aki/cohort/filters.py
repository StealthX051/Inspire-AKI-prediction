from __future__ import annotations

import pandas as pd

from inspire_aki.cohort.audit import record_count


def apply_preop_filters(
    df_preop: pd.DataFrame,
    df_ops_for_merge: pd.DataFrame,
    config: dict,
    audit: list[dict],
) -> tuple[pd.DataFrame, pd.DataFrame, list[dict]]:
    cohort_cfg = config["cohort"]
    feature_cfg = config["features"]

    df_preop = df_preop[df_preop["asa"] < cohort_cfg["max_asa_exclusive"]]
    audit = record_count(audit, "asa_lt_6", df_preop)

    df_preop = df_preop[df_preop["age"] >= cohort_cfg["min_age"]]
    audit = record_count(audit, "adult_only", df_preop)

    df_preop = df_preop.dropna(subset=["opend_time"])
    audit = record_count(audit, "has_opend_time", df_preop)
    df_preop = df_preop.dropna(subset=["opstart_time"])
    audit = record_count(audit, "has_opstart_time", df_preop)

    df_preop["op_len"] = df_preop["opend_time"] - df_preop["opstart_time"]
    df_preop = df_preop[df_preop["op_len"] > 0]
    audit = record_count(audit, "positive_op_len_only", df_preop)
    df_preop["sex"] = df_preop["sex"] == "M"
    df_preop = df_preop[~(df_preop["weight"].isna() | df_preop["height"].isna())]
    audit = record_count(audit, "has_height_weight", df_preop)
    df_preop = df_preop[(df_preop["weight"] != 0) & (df_preop["height"] != 0)]
    audit = record_count(audit, "nonzero_height_weight", df_preop)

    df_ops_for_merge = df_ops_for_merge.drop(df_ops_for_merge[df_ops_for_merge["antype"].isin(cohort_cfg["exclude_antype"])].index)
    antype_map = {"General": 0, "MAC": 1, "Neuraxial": 1}
    df_ops_for_merge["antype"] = df_ops_for_merge["antype"].map(antype_map).astype(float)
    df_ops_for_merge = df_ops_for_merge[~df_ops_for_merge["department"].isin(cohort_cfg["department_exclude"])]
    df_ops_for_merge = pd.get_dummies(df_ops_for_merge, columns=["department"])

    ignore_cols = set(feature_cfg["base_ignore_cols"])
    ignore_cols.update(col for col in df_preop.columns if "department_" in col)
    return df_preop, df_ops_for_merge, audit
