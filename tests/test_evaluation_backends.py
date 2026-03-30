from __future__ import annotations

import pandas as pd

from inspire_aki.evaluation.backends import build_evaluation_backend


def _backend_df() -> pd.DataFrame:
    rows = []
    op_id = 1
    for patient_id, labels in {
        11: [0, 0],
        12: [1, 0],
        13: [0, 0],
        14: [1, 1],
        15: [0, 1],
        16: [0, 0],
        17: [1, 1],
        18: [0, 0],
        19: [1, 0],
        20: [0, 0],
    }.items():
        for label in labels:
            rows.append({"op_id": op_id, "patient_id": patient_id, "aki_boolean": label})
            op_id += 1
    return pd.DataFrame(rows)


def test_backend_factory_builds_legacy_manifest(loaded_synthetic_config) -> None:
    config = dict(loaded_synthetic_config)
    config["evaluation_mode"] = "legacy_repeated_cv"
    backend = build_evaluation_backend(config)

    result = backend.build(_backend_df(), target="aki_boolean", dataset_family="tabular_common")

    assert backend.mode == "legacy_repeated_cv"
    assert set(result.manifest["evaluation_mode"]) == {"legacy_repeated_cv"}
    assert not result.overlap_audit.empty


def test_backend_factory_builds_grouped_holdout_manifest(loaded_synthetic_config) -> None:
    config = dict(loaded_synthetic_config)
    config["evaluation_mode"] = "grouped_holdout"
    config["splits"]["n_cv_folds"] = 3
    backend = build_evaluation_backend(config)

    result = backend.build(_backend_df(), target="aki_boolean", dataset_family="tabular_common")
    outer = result.manifest[result.manifest["split_scope"] == "outer"]

    assert backend.mode == "grouped_holdout"
    assert set(result.manifest["evaluation_mode"]) == {"grouped_holdout"}
    assert set(outer["split_name"]) == {"train", "test"}
    assert not set(outer.loc[outer["split_name"] == "train", "patient_id"]) & set(outer.loc[outer["split_name"] == "test", "patient_id"])


def test_backend_factory_builds_grouped_nested_manifest(loaded_synthetic_config) -> None:
    config = dict(loaded_synthetic_config)
    config["evaluation_mode"] = "grouped_nested_cv"
    config["splits"]["n_cv_folds"] = 5
    backend = build_evaluation_backend(config)

    result = backend.build(_backend_df(), target="aki_boolean", dataset_family="tabular_common")
    outer_test = result.manifest[(result.manifest["split_scope"] == "outer") & (result.manifest["split_name"] == "test")]

    assert backend.mode == "grouped_nested_cv"
    assert set(result.manifest["evaluation_mode"]) == {"grouped_nested_cv"}
    assert outer_test["op_id"].nunique() == _backend_df()["op_id"].nunique()
