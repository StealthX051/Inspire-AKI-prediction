from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest
import yaml


def _build_operations(n_ops: int) -> pd.DataFrame:
    rows = []
    for idx in range(n_ops):
        op_id = idx + 1
        subject_id = 1000 + op_id
        opstart = 10_000 + idx * 500
        opend = opstart + 120
        rows.append(
            {
                "op_id": op_id,
                "subject_id": subject_id,
                "age": 45 + idx,
                "sex": "M" if idx % 2 == 0 else "F",
                "height": 170 + (idx % 3),
                "weight": 70 + idx,
                "asa": 3 if idx < (n_ops // 2) else 4,
                "emop": idx % 2,
                "opstart_time": opstart,
                "opend_time": opend,
                "inhosp_death_time": None,
                "allcause_death_time": None,
                "orin_time": opstart - 20,
                "orout_time": opend + 20,
                "antype": "General",
                "department": "GS",
                "icd10_pcs": "0AB",
            }
        )
    return pd.DataFrame(rows)


def _build_diagnosis(operations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, row in operations.iterrows():
        rows.append(
            {
                "subject_id": row["subject_id"],
                "chart_time": row["opstart_time"] - 200,
                "icd10_cm": "I10" if idx % 3 == 0 else "J20",
            }
        )
    return pd.DataFrame(rows)


def _build_labs(operations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    positive_ops = {operations.iloc[-1]["op_id"], operations.iloc[-2]["op_id"], operations.iloc[-3]["op_id"], operations.iloc[-4]["op_id"]}
    dialysis_only_op = operations.iloc[-1]["op_id"]
    for idx, row in operations.iterrows():
        subject_id = row["subject_id"]
        rows.append(
            {
                "subject_id": subject_id,
                "chart_time": row["opstart_time"] - 60,
                "item_name": "creatinine",
                "value": 1.0 + (idx % 2) * 0.05,
            }
        )
        rows.append(
            {
                "subject_id": subject_id,
                "chart_time": row["opstart_time"] - 45,
                "item_name": "sodium",
                "value": 137 + idx,
            }
        )
        if row["op_id"] != dialysis_only_op:
            rows.append(
                {
                    "subject_id": subject_id,
                    "chart_time": row["opend_time"] + 60,
                    "item_name": "creatinine",
                    "value": 2.4 if row["op_id"] in positive_ops else 1.1,
                }
            )
    return pd.DataFrame(rows)


def _build_ward_vitals(operations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    dialysis_only_subject = operations.iloc[-1]["subject_id"]
    for _, row in operations.iterrows():
        rows.append(
            {
                "subject_id": row["subject_id"],
                "chart_time": row["opstart_time"] - 30,
                "item_name": "spo2",
                "value": 96 + (row["op_id"] % 3),
            }
        )
        rows.append(
            {
                "subject_id": row["subject_id"],
                "chart_time": row["opend_time"] + 30,
                "item_name": "crrt",
                "value": 1 if row["subject_id"] == dialysis_only_subject else 0,
            }
        )
    return pd.DataFrame(rows)


def _build_vitals(operations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for idx, row in operations.iterrows():
        op_id = row["op_id"]
        positive_shift = 10 if op_id > (len(operations) - 4) else 0
        for offset in [0, 5, 10, 15, 20]:
            rows.extend(
                [
                    {"op_id": op_id, "chart_time": offset, "item_name": "hr", "value": 60 + idx + offset * 0.2 + positive_shift},
                    {"op_id": op_id, "chart_time": offset, "item_name": "rr", "value": 12 + idx * 0.1 + offset * 0.1},
                    {"op_id": op_id, "chart_time": offset, "item_name": "nibp_sbp", "value": 110 + idx + offset * 0.3},
                    {"op_id": op_id, "chart_time": offset, "item_name": "etdes", "value": 3.0 + 0.1 * offset},
                    {"op_id": op_id, "chart_time": offset, "item_name": "etsevo", "value": 1.0 + 0.05 * offset},
                ]
            )
        rows.extend(
            [
                {"op_id": op_id, "chart_time": 10, "item_name": "o2", "value": 2.0 + idx * 0.1},
                {"op_id": op_id, "chart_time": 5, "item_name": "eph", "value": 0.05 + idx * 0.001},
                {"op_id": op_id, "chart_time": 15, "item_name": "eph", "value": 0.04 + idx * 0.001},
                {"op_id": op_id, "chart_time": 20, "item_name": "uo", "value": 40 + idx},
                {"op_id": op_id, "chart_time": 20, "item_name": "rbc", "value": 0 if idx % 4 else 1},
                {"op_id": op_id, "chart_time": 20, "item_name": "ns", "value": 500 + idx * 10},
            ]
        )
    return pd.DataFrame(rows)


def _write_workspace(base_path: Path) -> Path:
    raw_dir = base_path / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    operations = _build_operations(12)
    operations.to_csv(raw_dir / "operations.csv", index=False)
    _build_diagnosis(operations).to_csv(raw_dir / "diagnosis.csv", index=False)
    _build_labs(operations).to_csv(raw_dir / "labs.csv", index=False)
    _build_ward_vitals(operations).to_csv(raw_dir / "ward_vitals.csv", index=False)
    _build_vitals(operations).to_csv(raw_dir / "vitals.csv", index=False)

    config = {
        "paths": {
            "artifacts_dir": str(base_path / "artifacts"),
            "raw_inspire_dir": str(raw_dir),
            "compat_aki_dir": str(base_path / "compat_aki"),
            "compat_base_dir": str(base_path / "compat_base"),
            "compat_results_dir": str(base_path / "compat_results"),
        },
        "features": {
            "preop_lab_items": ["creatinine", "sodium"],
            "ward_items": ["spo2"],
            "high_frequency_labels": ["hr", "rr"],
            "medium_frequency_labels": ["nibp_sbp"],
            "cross_sec_avg_labels": ["o2"],
            "wt_adjusted_labels": ["eph"],
            "time_adjusted_labels": ["uo", "rbc"],
            "fluids_agg_labels": ["ns"],
            "anesthetic_labels": ["etdes", "etsevo"],
        },
        "sequence": {
            "pad_length": 10,
            "presence_threshold": 0.01,
        },
        "splits": {
            "use_bootstrapping": True,
            "n_bootstrap_iterations": 4,
            "n_cv_folds": 2,
            "holdout_fraction": 0.25,
            "hpo_validation_fraction_within_train": 0.25,
        },
        "models": {
            "tabular_enabled": ["log_reg", "asa_rule"],
            "sequence_enabled": [],
            "tabular_hpo_enabled": [],
            "sequence_hpo_enabled": [],
        },
        "calibration": {
            "cv_folds": 3,
            "threshold_steps": 25,
        },
        "evaluation": {
            "bootstrap_reps": 20,
        },
        "reports": {
            "batch_shap_jobs": [],
        },
        "runtime": {
            "profile": "balanced",
            "progress_interval_seconds": 60,
            "orchestration": {
                "mode": "serial",
            },
            "cpu_reserve_fraction": 0.125,
            "cpu_reserve_min": 4,
            "ram_reserve_fraction": 0.15,
            "ram_reserve_gb_min": 16,
        },
    }
    config_path = base_path / "test_config.yaml"
    config_path.write_text(yaml.safe_dump(config), encoding="utf-8")
    return config_path


@pytest.fixture()
def synthetic_config(tmp_path: Path) -> Path:
    return _write_workspace(tmp_path)


@pytest.fixture(scope="session")
def shared_synthetic_config(tmp_path_factory: pytest.TempPathFactory) -> Path:
    return _write_workspace(tmp_path_factory.mktemp("shared_synthetic"))


@pytest.fixture()
def loaded_synthetic_config(synthetic_config: Path) -> dict:
    from inspire_aki.config import load_config

    return load_config(synthetic_config)


@pytest.fixture(scope="session")
def completed_pipeline(shared_synthetic_config: Path) -> dict:
    from inspire_aki.config import load_config
    from inspire_aki.pipelines.evaluate import run_calibration, run_dca, run_delong, run_metrics
    from inspire_aki.pipelines.evaluate_generate import run_evaluate_generate
    from inspire_aki.pipelines.preprocess import run_intraop, run_labels, run_preop, run_sequence, run_tabular, run_timeseries
    from inspire_aki.pipelines.report import run_consort, run_curves, run_tables
    from inspire_aki.pipelines.train import run_train_tabular

    config = load_config(shared_synthetic_config)
    run_preop(config)
    run_intraop(config)
    run_tabular(config)
    run_labels(config)
    run_timeseries(config)
    run_sequence(config)
    if config.get("evaluation_mode", "legacy_repeated_cv") != "legacy_repeated_cv":
        run_evaluate_generate(config)
    run_train_tabular(config)
    run_calibration(config)
    run_metrics(config)
    run_delong(config)
    run_dca(config)
    run_consort(config)
    run_tables(config)
    run_curves(config)
    return config
