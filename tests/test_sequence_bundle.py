from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from inspire_aki.models.sequence import fit_sequence_model, load_sequence_bundle, predict_sequence_bundle


def test_sequence_bundle_round_trip(loaded_synthetic_config, tmp_path: Path) -> None:
    pytest.importorskip("torch")

    sequence_rows = []
    for idx in range(8):
        sequence_rows.append(
            {
                "op_id": idx + 1,
                "aki_boolean": idx % 2,
                "static_a": float(idx),
                "static_b": float(idx) / 10.0,
                "seq_len": 4,
                "time_tensors": np.full((4, 2), fill_value=idx + 1, dtype=np.float32),
            }
        )
    train_df = pd.DataFrame(sequence_rows)
    feature_cols_tab = ["static_a", "static_b"]
    params = {
        "learning_rate": 0.01,
        "epochs": 2,
        "batch_size": 2,
        "patience": 2,
        "es_check_interval": 1,
        "lr_scheduler_patience": 1,
        "lr_scheduler_factor": 0.5,
        "gradient_clip_value": 1.0,
        "lstm_hidden_size": 4,
        "lstm_num_layers": 1,
        "mlp_dims": [],
        "dropout_rate": 0.1,
    }

    bundle = fit_sequence_model(
        model_key="lstm_only",
        train_df=train_df,
        feature_cols_tab=feature_cols_tab,
        target="aki_boolean",
        params=params,
        config=loaded_synthetic_config,
        model_output_dir=tmp_path / "sequence_model",
        seed=42,
    )
    loaded_bundle = load_sequence_bundle(tmp_path / "sequence_model")

    y_pred_original, y_prob_original = predict_sequence_bundle(bundle, train_df.iloc[:4].copy())
    y_pred_loaded, y_prob_loaded = predict_sequence_bundle(loaded_bundle, train_df.iloc[:4].copy())

    np.testing.assert_array_equal(y_pred_original, y_pred_loaded)
    np.testing.assert_allclose(y_prob_original, y_prob_loaded, rtol=1e-5, atol=1e-6)
    assert loaded_bundle.time_input_size == 2
    assert loaded_bundle.lstm_hidden_size == 4


def test_sequence_bundle_uses_dynamic_target_metadata(loaded_synthetic_config, tmp_path: Path) -> None:
    pytest.importorskip("torch")

    sequence_rows = []
    for idx in range(8):
        sequence_rows.append(
            {
                "op_id": idx + 1,
                "macce": idx % 2,
                "static_a": float(idx),
                "static_b": float(idx) / 10.0,
                "seq_len": 4,
                "time_tensors": np.full((4, 2), fill_value=idx + 1, dtype=np.float32),
            }
        )
    train_df = pd.DataFrame(sequence_rows)
    params = {
        "learning_rate": 0.01,
        "epochs": 2,
        "batch_size": 2,
        "patience": 2,
        "es_check_interval": 1,
        "lr_scheduler_patience": 1,
        "lr_scheduler_factor": 0.5,
        "gradient_clip_value": 1.0,
        "lstm_hidden_size": 4,
        "lstm_num_layers": 1,
        "mlp_dims": [],
        "dropout_rate": 0.1,
    }

    bundle = fit_sequence_model(
        model_key="lstm_only",
        train_df=train_df,
        feature_cols_tab=["static_a", "static_b"],
        target="macce",
        params=params,
        config=loaded_synthetic_config,
        model_output_dir=tmp_path / "sequence_model_macce",
        seed=7,
    )
    loaded_bundle = load_sequence_bundle(tmp_path / "sequence_model_macce")

    y_pred, y_prob = predict_sequence_bundle(loaded_bundle, train_df.iloc[:4].copy())

    assert loaded_bundle.metadata["target"] == "macce"
    assert len(y_pred) == 4
    assert len(y_prob) == 4
