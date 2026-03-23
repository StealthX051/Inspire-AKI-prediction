from __future__ import annotations

import pandas as pd
import pytest

from inspire_aki.config import config_hash, load_config, validate_config
from inspire_aki.io.artifacts import ArtifactManager


def test_load_config_merges_override_and_preserves_defaults(synthetic_config) -> None:
    config = load_config(synthetic_config)
    assert config["paths"]["artifacts_dir"].endswith("artifacts")
    assert config["features"]["preop_lab_items"] == ["creatinine", "sodium"]
    assert config["cohort"]["min_age"] == 18
    assert config["models"]["target"] == "aki_boolean"


def test_config_hash_changes_with_override(synthetic_config) -> None:
    config_a = load_config(synthetic_config)
    config_b = load_config(synthetic_config)
    config_b["splits"]["random_state"] = 999
    assert config_hash(config_a) == config_hash(load_config(synthetic_config))
    assert config_hash(config_a) != config_hash(config_b)


def test_load_config_normalizes_legacy_shap_key_and_removes_dead_compat(synthetic_config) -> None:
    config = load_config(synthetic_config)
    assert "batch_shap_jobs" not in config["reports"]
    assert config["reports"]["shap_jobs"] == []
    assert config["reports"]["manuscript_sections"] == ["consort", "tables", "curves", "shap"]
    assert "compat" not in config or "export_legacy_aliases" not in config.get("compat", {})


def test_default_config_validates() -> None:
    config = load_config()
    assert config["reports"]["shap_jobs"]
    assert config["reports"]["manuscript_sections"] == ["consort", "tables", "curves", "shap"]


def test_validate_config_rejects_unsupported_shap_jobs(loaded_synthetic_config) -> None:
    loaded_synthetic_config["reports"]["shap_jobs"] = [{"dataset_regime": "combined", "model_key": "svm"}]
    with pytest.raises(ValueError, match="Unsupported SHAP model_key"):
        validate_config(loaded_synthetic_config)


def test_validate_config_rejects_invalid_bootstrap_ratio(loaded_synthetic_config) -> None:
    loaded_synthetic_config["splits"]["n_bootstrap_iterations"] = 5
    loaded_synthetic_config["splits"]["n_cv_folds"] = 2
    with pytest.raises(ValueError, match="n_bootstrap_iterations"):
        validate_config(loaded_synthetic_config)


def test_artifact_manager_round_trips_and_writes_manifest(loaded_synthetic_config) -> None:
    artifacts = ArtifactManager(loaded_synthetic_config)
    frame = pd.DataFrame({"op_id": [1, 2], "value": [0.1, 0.2]})
    csv_path = artifacts.write_dataframe(frame, "unit", "frame.csv")
    parquet_path = artifacts.write_dataframe(frame, "unit", "frame.parquet")
    json_path = artifacts.write_json({"stage": "unit"}, "unit", "payload.json")
    pickle_path = artifacts.write_pickle({"numbers": [1, 2, 3]}, "unit", "payload.pkl")
    manifest_path = artifacts.write_manifest(
        "unit_test",
        ["unit", "manifest.json"],
        inputs=["tests/input.csv"],
        outputs=[artifacts.relative(csv_path)],
        metadata={"rows": len(frame)},
    )

    pd.testing.assert_frame_equal(artifacts.read_dataframe("unit", "frame.csv"), frame)
    pd.testing.assert_frame_equal(artifacts.read_dataframe("unit", "frame.parquet"), frame)
    assert artifacts.read_json("unit", "payload.json") == {"stage": "unit"}
    assert artifacts.read_pickle("unit", "payload.pkl") == {"numbers": [1, 2, 3]}
    manifest = artifacts.read_json("unit", "manifest.json")
    assert manifest["stage"] == "unit_test"
    assert manifest["metadata"]["rows"] == 2
    assert csv_path.exists() and parquet_path.exists() and json_path.exists() and pickle_path.exists() and manifest_path.exists()
