from __future__ import annotations

import copy
from pathlib import Path

import pandas as pd

from inspire_aki.cohort.labels import derive_active_labels, derive_aki_labels
from inspire_aki.config import load_config


def test_dialysis_positive_case_is_labeled(synthetic_config) -> None:
    config = load_config(synthetic_config)
    raw_dir = synthetic_config.parent / "raw"
    preop_df = pd.read_csv(synthetic_config.parent / "artifacts" / "features" / "preop" / "preop_features.csv") if (synthetic_config.parent / "artifacts" / "features" / "preop" / "preop_features.csv").exists() else None
    if preop_df is None:
        from inspire_aki.cohort.preop import build_preop_features
        from inspire_aki.datasets.tabular import build_tabular_datasets
        from inspire_aki.features.intraop_tabular import build_intraop_features

        preop_df, _ = build_preop_features(config, raw_dir)
        intraop_df = build_intraop_features(pd.read_csv(raw_dir / "vitals.csv"), preop_df, config)
        combined_df = build_tabular_datasets(preop_df, intraop_df, config)["combined"]
    else:
        combined_df = pd.read_csv(synthetic_config.parent / "artifacts" / "datasets" / "tabular" / "tabular_combined.csv")

    labels_df, _ = derive_aki_labels(
        config=config,
        raw_inspire_dir=raw_dir,
        preop_df=preop_df,
        tabular_combined_df=combined_df,
    )
    dialysis_op_id = 12
    assert int(labels_df.loc[labels_df["op_id"] == dialysis_op_id, "aki_boolean"].iloc[0]) == 1


def _activate_outcome(config: dict, outcome_key: str) -> dict:
    updated = copy.deepcopy(config)
    updated["study"]["outcome_key"] = outcome_key
    updated["outcome"] = copy.deepcopy(updated["outcomes"]["catalog"][outcome_key])
    updated["models"]["target"] = updated["outcome"]["target_column"]
    return updated


def _write_diagnosis_csv(raw_dir: Path, rows: list[dict[str, object]]) -> None:
    pd.DataFrame(rows, columns=["subject_id", "chart_time", "icd10_cm"]).to_csv(raw_dir / "diagnosis.csv", index=False)


def _write_operations_csv(raw_dir: Path, rows: list[dict[str, object]], *, source_column: str) -> None:
    pd.DataFrame(rows, columns=["op_id", source_column]).to_csv(raw_dir / "operations.csv", index=False)


def _base_outcome_frames() -> tuple[pd.DataFrame, pd.DataFrame]:
    preop_df = pd.DataFrame(
        {
            "op_id": [1, 2, 3, 4, 5],
            "subject_id": [101, 102, 103, 104, 105],
            "opend_time": [1_000, 2_000, 3_000, 4_000, 5_000],
        }
    )
    tabular_combined_df = pd.DataFrame({"op_id": [1, 2, 3, 4, 5]})
    return preop_df, tabular_combined_df


def test_active_aki_labels_match_legacy_labels(synthetic_config) -> None:
    config = load_config(synthetic_config)
    raw_dir = synthetic_config.parent / "raw"
    preop_df = pd.read_csv(synthetic_config.parent / "artifacts" / "features" / "preop" / "preop_features.csv") if (synthetic_config.parent / "artifacts" / "features" / "preop" / "preop_features.csv").exists() else None
    if preop_df is None:
        from inspire_aki.cohort.preop import build_preop_features
        from inspire_aki.datasets.tabular import build_tabular_datasets
        from inspire_aki.features.intraop_tabular import build_intraop_features

        preop_df, _ = build_preop_features(config, raw_dir)
        intraop_df = build_intraop_features(pd.read_csv(raw_dir / "vitals.csv"), preop_df, config)
        combined_df = build_tabular_datasets(preop_df, intraop_df, config)["combined"]
    else:
        combined_df = pd.read_csv(synthetic_config.parent / "artifacts" / "datasets" / "tabular" / "tabular_combined.csv")

    legacy_labels, legacy_audit = derive_aki_labels(
        config=config,
        raw_inspire_dir=raw_dir,
        preop_df=preop_df,
        tabular_combined_df=combined_df,
    )
    active_labels, active_audit = derive_active_labels(
        config=_activate_outcome(config, "aki"),
        raw_inspire_dir=raw_dir,
        preop_df=preop_df,
        tabular_combined_df=combined_df,
    )

    pd.testing.assert_frame_equal(
        legacy_labels.sort_values("op_id").reset_index(drop=True),
        active_labels.sort_values("op_id").reset_index(drop=True),
    )
    pd.testing.assert_frame_equal(legacy_audit.reset_index(drop=True), active_audit.reset_index(drop=True))


def test_macce_composite_matches_component_or(tmp_path, loaded_synthetic_config) -> None:
    config = _activate_outcome(loaded_synthetic_config, "macce")
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(exist_ok=True)
    preop_df, tabular_combined_df = _base_outcome_frames()
    _write_diagnosis_csv(
        raw_dir,
        [
            {"subject_id": 101, "chart_time": 1_005, "icd10_cm": "I21.9"},
            {"subject_id": 102, "chart_time": 2_005, "icd10_cm": "I63.9"},
            {"subject_id": 102, "chart_time": 2_010, "icd10_cm": "I50.1"},
            {"subject_id": 103, "chart_time": 3_005, "icd10_cm": "I46.0"},
            {"subject_id": 104, "chart_time": 3_999, "icd10_cm": "I20.0"},
            {"subject_id": 105, "chart_time": 5_000 + 30 * 24 * 60 + 1, "icd10_cm": "I50.0"},
        ],
    )

    labels_df, _ = derive_active_labels(
        config=config,
        raw_inspire_dir=raw_dir,
        preop_df=preop_df,
        tabular_combined_df=tabular_combined_df,
    )
    labels_df = labels_df.sort_values("op_id").reset_index(drop=True)

    assert labels_df["macce"].tolist() == [True, True, True, False, False]
    assert labels_df["macce_mi"].tolist() == [True, False, False, False, False]
    assert labels_df["macce_stroke"].tolist() == [False, True, False, False, False]
    assert labels_df["macce_hf"].tolist() == [False, True, False, False, False]
    assert labels_df["macce_cardiac_arrest"].tolist() == [False, False, True, False, False]
    assert labels_df["macce"].equals(
        labels_df[["macce_mi", "macce_stroke", "macce_angina", "macce_hf", "macce_cardiac_arrest"]].any(axis=1)
    )


def test_diagnosis_window_outcomes_respect_time_boundaries(tmp_path, loaded_synthetic_config) -> None:
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(exist_ok=True)
    preop_df, tabular_combined_df = _base_outcome_frames()
    _write_diagnosis_csv(
        raw_dir,
        [
            {"subject_id": 101, "chart_time": 1_000, "icd10_cm": "J18.9"},
            {"subject_id": 102, "chart_time": 2_000 + 30 * 24 * 60, "icd10_cm": "I26.0"},
            {"subject_id": 103, "chart_time": 2_999, "icd10_cm": "J12.0"},
            {"subject_id": 104, "chart_time": 4_000 + 30 * 24 * 60 + 1, "icd10_cm": "I26.9"},
        ],
    )

    pna_labels, _ = derive_active_labels(
        config=_activate_outcome(loaded_synthetic_config, "pna"),
        raw_inspire_dir=raw_dir,
        preop_df=preop_df,
        tabular_combined_df=tabular_combined_df,
    )
    pe_labels, _ = derive_active_labels(
        config=_activate_outcome(loaded_synthetic_config, "pe"),
        raw_inspire_dir=raw_dir,
        preop_df=preop_df,
        tabular_combined_df=tabular_combined_df,
    )

    assert pna_labels.sort_values("op_id")["pna"].tolist() == [True, False, False, False, False]
    assert pe_labels.sort_values("op_id")["pe"].tolist() == [False, True, False, False, False]


def test_postop_icu_admission_requires_strictly_later_event_time(tmp_path, loaded_synthetic_config) -> None:
    config = _activate_outcome(loaded_synthetic_config, "postop_icu_admission")
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(exist_ok=True)
    preop_df, tabular_combined_df = _base_outcome_frames()
    _write_operations_csv(
        raw_dir,
        [
            {"op_id": 1, "icuin_time": 1_001},
            {"op_id": 2, "icuin_time": 2_000},
            {"op_id": 3, "icuin_time": None},
            {"op_id": 4, "icuin_time": 4_500},
            {"op_id": 5, "icuin_time": 4_999},
        ],
        source_column="icuin_time",
    )

    labels_df, _ = derive_active_labels(
        config=config,
        raw_inspire_dir=raw_dir,
        preop_df=preop_df,
        tabular_combined_df=tabular_combined_df,
    )

    assert labels_df.sort_values("op_id")["postop_icu_admission"].tolist() == [True, False, False, True, False]


def test_postop_mortality_30d_respects_window_boundaries(tmp_path, loaded_synthetic_config) -> None:
    config = _activate_outcome(loaded_synthetic_config, "postop_mortality_30d")
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir(exist_ok=True)
    preop_df, tabular_combined_df = _base_outcome_frames()
    window_minutes = 30 * 24 * 60
    _write_operations_csv(
        raw_dir,
        [
            {"op_id": 1, "allcause_death_time": 1_001},
            {"op_id": 2, "allcause_death_time": 2_000},
            {"op_id": 3, "allcause_death_time": 3_000 + window_minutes},
            {"op_id": 4, "allcause_death_time": 4_000 + window_minutes + 1},
            {"op_id": 5, "allcause_death_time": None},
        ],
        source_column="allcause_death_time",
    )

    labels_df, _ = derive_active_labels(
        config=config,
        raw_inspire_dir=raw_dir,
        preop_df=preop_df,
        tabular_combined_df=tabular_combined_df,
    )

    assert labels_df.sort_values("op_id")["postop_mortality_30d"].tolist() == [True, False, True, False, False]
