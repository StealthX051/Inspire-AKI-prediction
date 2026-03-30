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

    df_preop = df_preop[df_preop["asa"] < cohort_cfg["max_asa_exclusive"]]
    audit = record_count(audit, "asa_lt_6", df_preop)

    df_preop = df_preop[df_preop["age"] >= cohort_cfg["min_age"]]
    audit = record_count(audit, "adult_only", df_preop)

    df_preop = df_preop.dropna(subset=["opend_time"])
    audit = record_count(audit, "has_opend_time", df_preop)
    df_preop = df_preop.dropna(subset=["opstart_time"])
    audit = record_count(audit, "has_opstart_time", df_preop)

    df_preop["op_len"] = df_preop["opend_time"] - df_preop["opstart_time"]
    if bool(cohort_cfg.get("require_positive_op_len", True)):
        df_preop = df_preop[df_preop["op_len"] > 0]
    audit = record_count(audit, "positive_op_len_only", df_preop)
    df_preop["sex"] = df_preop["sex"] == "M"
    if bool(cohort_cfg.get("require_height_weight", True)):
        df_preop = df_preop[~(df_preop["weight"].isna() | df_preop["height"].isna())]
    audit = record_count(audit, "has_height_weight", df_preop)
    if bool(cohort_cfg.get("require_height_weight", True)):
        df_preop = df_preop[(df_preop["weight"] != 0) & (df_preop["height"] != 0)]
    audit = record_count(audit, "nonzero_height_weight", df_preop)

    exclude_antype = cohort_cfg.get("exclude_antype", [])
    if exclude_antype:
        df_ops_for_merge = df_ops_for_merge.drop(df_ops_for_merge[df_ops_for_merge["antype"].isin(exclude_antype)].index)
    antype_map = {"General": 0, "MAC": 1, "Neuraxial": 1}
    df_ops_for_merge["antype"] = df_ops_for_merge["antype"].map(antype_map).astype(float)
    department_include = cohort_cfg.get("department_include", [])
    if department_include:
        df_ops_for_merge = df_ops_for_merge[df_ops_for_merge["department"].isin(department_include)]
    department_exclude = cohort_cfg.get("department_exclude", [])
    if department_exclude:
        df_ops_for_merge = df_ops_for_merge[~df_ops_for_merge["department"].isin(department_exclude)]
    df_ops_for_merge = pd.get_dummies(df_ops_for_merge, columns=["department"])
    return df_preop, df_ops_for_merge, audit
