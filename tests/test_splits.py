from __future__ import annotations

import pandas as pd

from inspire_aki.datasets.splits import build_bootstrap_split_manifest


def test_bootstrap_split_manifest_is_reproducible() -> None:
    df = pd.DataFrame(
        {
            "op_id": list(range(1, 13)),
            "aki_boolean": [0, 1] * 6,
        }
    )
    manifest_a = build_bootstrap_split_manifest(
        df,
        target="aki_boolean",
        dataset_regime="preop",
        population_id="preop",
        random_state=42,
        n_iterations=4,
        n_cv_folds=2,
        use_bootstrapping=True,
    )
    manifest_b = build_bootstrap_split_manifest(
        df,
        target="aki_boolean",
        dataset_regime="preop",
        population_id="preop",
        random_state=42,
        n_iterations=4,
        n_cv_folds=2,
        use_bootstrapping=True,
    )
    pd.testing.assert_frame_equal(manifest_a, manifest_b)
