from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from inspire_aki.datasets.splits import subset_from_manifest
from inspire_aki.evaluation.split_manager import subset_generated_manifest
from inspire_aki.io.artifacts import ArtifactManager
from inspire_aki.models.tabular import load_tabular_bundle
from inspire_aki.registry import SUPPORTED_SHAP_MODELS, model_display_name
from inspire_aki.reporting.rendering import FigureExportSpec, report_figure_style_context, save_figure_variants
from inspire_aki.runtime import build_stage_runtime_plan, thread_limited_context


_DATASET_LABELS = {"preop": "Preop", "intraop": "Intraop", "combined": "Combined"}
_SHAP_SAMPLE_RANDOM_STATE = 42
_SHAP_BACKGROUND_SAMPLE_N = 200
_SHAP_MIN_ROWS_PER_CHUNK = 256
_SCATTER_NEGATIVE_COLOR = "#2d6ba3"
_SCATTER_POSITIVE_COLOR = "#bf3b3b"
_SCATTER_TREND_COLOR = "#111827"
_NUMPY_RANK_WARNING = getattr(getattr(np, "exceptions", None), "RankWarning", RuntimeWarning)


@dataclass(frozen=True)
class ShapExplanationBundle:
    dataset_regime: str
    model_key: str
    feature_names: tuple[str, ...]
    display_frame: pd.DataFrame
    shap_values: np.ndarray
    importance: pd.DataFrame
    y_true: pd.Series

    def feature_index(self, feature_name: str) -> int:
        return self.feature_names.index(feature_name)


def _load_dataset_for_regime(artifacts: ArtifactManager, dataset_regime: str) -> pd.DataFrame:
    return pd.read_csv(artifacts.paths.artifact_path("datasets", "tabular", f"tabular_{dataset_regime}_labeled.csv"))


def _resolve_split_manifest_path(artifacts: ArtifactManager, dataset_regime: str) -> Path:
    candidates = [
        artifacts.paths.artifact_path("datasets", "splits", f"bootstrap_{dataset_regime}.parquet"),
        artifacts.paths.artifact_path("datasets", "splits", f"grouped_nested_cv_{dataset_regime}.parquet"),
        artifacts.paths.artifact_path("datasets", "splits", f"grouped_holdout_{dataset_regime}.parquet"),
    ]
    for path in candidates:
        if path.exists():
            return path
    raise FileNotFoundError(
        "Expected a split manifest for SHAP reporting but none were found. "
        f"Tried: {', '.join(str(path) for path in candidates)}"
    )


def _load_split(artifacts: ArtifactManager, dataset_regime: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    manifest = pd.read_parquet(_resolve_split_manifest_path(artifacts, dataset_regime))
    dataset_df = _load_dataset_for_regime(artifacts, dataset_regime)
    if {"split_scope", "outer_repeat_id", "outer_fold_id"}.issubset(manifest.columns):
        train_df = subset_generated_manifest(dataset_df, manifest, split_name="train", run_id=0)
        test_df = subset_generated_manifest(dataset_df, manifest, split_name="test", run_id=0)
    else:
        train_df = subset_from_manifest(dataset_df, manifest, repeat_id=0, fold_id=0, split_name="train")
        test_df = subset_from_manifest(dataset_df, manifest, repeat_id=0, fold_id=0, split_name="test")
    return train_df, test_df


def _shap_bundle_dir(artifacts: ArtifactManager, dataset_regime: str, model_key: str) -> Path:
    return artifacts.paths.artifact_path("models", "tabular", dataset_regime, model_key, "repeat_0", "fold_0")


def _select_binary_class_shap_values(shap_values: object, feature_count: int) -> np.ndarray:
    if isinstance(shap_values, list):
        if not shap_values:
            raise ValueError("SHAP returned an empty list of explanations.")
        values = np.asarray(shap_values[1] if len(shap_values) > 1 else shap_values[0])
    else:
        values = np.asarray(shap_values)

    if values.ndim == 2:
        return values
    if values.ndim != 3:
        raise ValueError(f"Unsupported SHAP value shape: {values.shape!r}")

    if values.shape[1] == feature_count:
        class_index = 1 if values.shape[2] > 1 else 0
        return values[:, :, class_index]
    if values.shape[2] == feature_count:
        class_index = 1 if values.shape[0] > 1 else 0
        return values[class_index, :, :]
    raise ValueError(f"Could not infer feature axis from SHAP value shape: {values.shape!r}")


def _sample_frame(frame: pd.DataFrame, *, limit: int) -> pd.DataFrame:
    if frame.empty or len(frame) <= limit:
        return frame.copy()
    return frame.sample(n=limit, random_state=_SHAP_SAMPLE_RANDOM_STATE).copy()


def _scale_model_frame(frame: pd.DataFrame, feature_cols: list[str], scaler) -> pd.DataFrame:
    if scaler is None:
        return frame[feature_cols].copy()
    return pd.DataFrame(
        scaler.transform(frame[feature_cols]),
        columns=feature_cols,
        index=frame.index,
    )


def _load_display_frame(artifacts: ArtifactManager, explain_df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    raw_path = artifacts.paths.artifact_path("datasets", "tabular", "tabular_combined_unnormalized.csv")
    raw_df = pd.read_csv(raw_path)
    if "op_id" not in raw_df.columns:
        raise ValueError(f"Expected op_id in raw SHAP display artifact: {raw_path}")
    missing_features = sorted(set(feature_cols) - set(raw_df.columns))
    if missing_features:
        raise ValueError(
            "Could not render SHAP scatter plots because raw display features are missing "
            f"from {raw_path}: {missing_features}"
        )
    raw_lookup = raw_df.set_index("op_id")
    op_ids = explain_df["op_id"].tolist()
    missing_op_ids = [int(op_id) for op_id in op_ids if op_id not in raw_lookup.index]
    if missing_op_ids:
        raise ValueError(
            "Could not align SHAP scatter plots to raw display rows. "
            f"Missing op_id values in {raw_path}: {missing_op_ids[:5]}"
        )
    display_frame = raw_lookup.loc[op_ids, feature_cols].copy()
    display_frame.index = explain_df.index
    return display_frame


def _compute_shap_values_chunk(
    model_key: str,
    model,
    x_background: pd.DataFrame,
    x_explain_chunk: pd.DataFrame,
    feature_cols: list[str],
) -> np.ndarray:
    import shap

    if model_key in {"xgb", "rf"}:
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(x_explain_chunk)
        return _select_binary_class_shap_values(shap_values, feature_count=len(feature_cols))
    if model_key == "log_reg":
        explainer = shap.LinearExplainer(model, x_background)
        return np.asarray(explainer.shap_values(x_explain_chunk))
    raise ValueError(f"Unsupported SHAP model_key '{model_key}'.")


def _compute_shap_values(
    model_key: str,
    model,
    x_background: pd.DataFrame,
    x_explain: pd.DataFrame,
    feature_cols: list[str],
    *,
    worker_budget: int = 1,
) -> np.ndarray:
    if worker_budget <= 1 or len(x_explain) < 32:
        return _compute_shap_values_chunk(model_key, model, x_background, x_explain, feature_cols)

    chunk_count = min(
        int(worker_budget),
        len(x_explain),
        max(1, int(len(x_explain) // _SHAP_MIN_ROWS_PER_CHUNK)),
    )
    if chunk_count <= 1:
        return _compute_shap_values_chunk(model_key, model, x_background, x_explain, feature_cols)

    row_chunks = [chunk for chunk in np.array_split(np.arange(len(x_explain)), chunk_count) if len(chunk) > 0]
    values_nested = Parallel(n_jobs=len(row_chunks), backend="threading")(
        delayed(_compute_shap_values_chunk)(
            model_key,
            model,
            x_background,
            x_explain.iloc[chunk].copy(),
            feature_cols,
        )
        for chunk in row_chunks
    )
    return np.vstack(values_nested)


def _build_importance_table(feature_cols: list[str], values: np.ndarray) -> pd.DataFrame:
    importance = pd.DataFrame(
        {
            "feature": feature_cols,
            "mean_abs_shap": np.abs(values).mean(axis=0),
        }
    )
    return importance.sort_values(["mean_abs_shap", "feature"], ascending=[False, True], kind="mergesort").reset_index(drop=True)


def _build_shap_explanation_bundle(
    config: dict,
    dataset_regime: str,
    model_key: str,
    *,
    shap_calc_workers: int = 1,
) -> ShapExplanationBundle | None:
    artifacts = ArtifactManager(config)
    bundle_dir = _shap_bundle_dir(artifacts, dataset_regime, model_key)
    train_df, test_df = _load_split(artifacts, dataset_regime)
    fitted_bundle = load_tabular_bundle(model_key, bundle_dir)
    feature_cols = list(fitted_bundle.feature_names)

    x_background_input = _sample_frame(train_df[feature_cols], limit=_SHAP_BACKGROUND_SAMPLE_N)
    # Match the archived notebook scatter/dependence behavior by explaining the
    # full held-out split, while still capping the explainer background matrix.
    x_explain_input = test_df[feature_cols].copy()
    explain_test_df = test_df.loc[x_explain_input.index].copy()
    if x_background_input.empty or x_explain_input.empty or explain_test_df.empty:
        return None

    x_background_model = _scale_model_frame(x_background_input, feature_cols, fitted_bundle.scaler)
    x_explain_model = _scale_model_frame(x_explain_input, feature_cols, fitted_bundle.scaler)
    display_frame = _load_display_frame(artifacts, explain_test_df, feature_cols)
    shap_values = _compute_shap_values(
        model_key,
        fitted_bundle.model,
        x_background_model,
        x_explain_model,
        feature_cols,
        worker_budget=shap_calc_workers,
    )
    importance = _build_importance_table(feature_cols, shap_values)
    return ShapExplanationBundle(
        dataset_regime=dataset_regime,
        model_key=model_key,
        feature_names=tuple(feature_cols),
        display_frame=display_frame,
        shap_values=shap_values,
        importance=importance,
        y_true=explain_test_df[config["models"]["target"]].astype(bool).copy(),
    )


def _safe_name(value: str) -> str:
    token = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value).strip()).strip("_")
    return token or "feature"


def _style_shap_axes(ax, *, draw_zero_line: bool = True) -> None:
    if draw_zero_line:
        ax.axhline(0.0, color="#7d8793", linestyle="--", linewidth=1.0, alpha=0.8, zorder=1)
    ax.grid(True, which="major", linestyle="--", linewidth=0.75, alpha=0.16, zorder=0)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#52606d")
    ax.spines["bottom"].set_color("#52606d")
    ax.spines["left"].set_linewidth(1.05)
    ax.spines["bottom"].set_linewidth(1.05)


def _trendline(feature_vals: np.ndarray, shap_vals: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    valid_mask = np.isfinite(feature_vals) & np.isfinite(shap_vals)
    if int(valid_mask.sum()) < 3:
        return None
    x_valid = feature_vals[valid_mask]
    y_valid = shap_vals[valid_mask]
    if np.unique(x_valid).size < 2:
        return None
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", _NUMPY_RANK_WARNING)
            slope, intercept = np.polyfit(x_valid, y_valid, deg=1)
    except _NUMPY_RANK_WARNING:
        return None
    if not np.isfinite(slope) or not np.isfinite(intercept):
        return None
    trend_x = np.linspace(float(np.nanmin(x_valid)), float(np.nanmax(x_valid)), 128)
    return trend_x, slope * trend_x + intercept


def _render_empty_plot(ax, message: str) -> None:
    ax.text(0.5, 0.5, message, ha="center", va="center", transform=ax.transAxes, color="#52606d")


def _scatter_figure(shap_bundle: ShapExplanationBundle, feature_name: str, config: dict):
    import matplotlib.pyplot as plt

    feature_idx = shap_bundle.feature_index(feature_name)
    feature_vals = pd.to_numeric(shap_bundle.display_frame[feature_name], errors="coerce").to_numpy(dtype=float, copy=False)
    shap_vals = shap_bundle.shap_values[:, feature_idx]
    y_true = shap_bundle.y_true.to_numpy(dtype=bool, copy=False)

    with report_figure_style_context(config):
        fig, ax = plt.subplots(figsize=(8.2, 5.8))
        negative_mask = (~y_true) & np.isfinite(feature_vals) & np.isfinite(shap_vals)
        positive_mask = y_true & np.isfinite(feature_vals) & np.isfinite(shap_vals)
        if negative_mask.any():
            ax.scatter(
                feature_vals[negative_mask],
                shap_vals[negative_mask],
                s=20,
                alpha=0.65,
                color=_SCATTER_NEGATIVE_COLOR,
                edgecolors="none",
                label="AKI Negative",
                zorder=3,
            )
        if positive_mask.any():
            ax.scatter(
                feature_vals[positive_mask],
                shap_vals[positive_mask],
                s=20,
                alpha=0.72,
                color=_SCATTER_POSITIVE_COLOR,
                edgecolors="none",
                label="AKI Positive",
                zorder=4,
            )
        trend = _trendline(feature_vals, shap_vals)
        if trend is not None:
            ax.plot(trend[0], trend[1], color=_SCATTER_TREND_COLOR, linewidth=2.0, label="Linear Trend", zorder=5)
        elif not negative_mask.any() and not positive_mask.any():
            _render_empty_plot(ax, "No finite raw display values available")

        label = _DATASET_LABELS.get(shap_bundle.dataset_regime, shap_bundle.dataset_regime.title())
        ax.set_xlabel(feature_name)
        ax.set_ylabel("SHAP Value")
        ax.set_title(f"{model_display_name(shap_bundle.model_key)} {label} SHAP Scatter\n{feature_name}")
        _style_shap_axes(ax)
        handles, labels = ax.get_legend_handles_labels()
        if handles:
            ax.legend(loc="best")
        return fig


def _dependence_figure(shap_bundle: ShapExplanationBundle, main_feature: str, interaction_feature: str, config: dict):
    import matplotlib.pyplot as plt

    main_idx = shap_bundle.feature_index(main_feature)
    main_vals = pd.to_numeric(shap_bundle.display_frame[main_feature], errors="coerce").to_numpy(dtype=float, copy=False)
    interaction_vals = pd.to_numeric(
        shap_bundle.display_frame[interaction_feature], errors="coerce"
    ).to_numpy(dtype=float, copy=False)
    shap_vals = shap_bundle.shap_values[:, main_idx]
    valid_mask = np.isfinite(main_vals) & np.isfinite(interaction_vals) & np.isfinite(shap_vals)

    with report_figure_style_context(config):
        fig, ax = plt.subplots(figsize=(8.2, 5.8))
        if valid_mask.any():
            scatter = ax.scatter(
                main_vals[valid_mask],
                shap_vals[valid_mask],
                c=interaction_vals[valid_mask],
                cmap="viridis",
                s=20,
                alpha=0.75,
                edgecolors="none",
                zorder=3,
            )
            colorbar = fig.colorbar(scatter, ax=ax)
            colorbar.set_label(interaction_feature)
        else:
            _render_empty_plot(ax, "No finite raw display values available")

        label = _DATASET_LABELS.get(shap_bundle.dataset_regime, shap_bundle.dataset_regime.title())
        ax.set_xlabel(main_feature)
        ax.set_ylabel(f"SHAP Value ({main_feature})")
        ax.set_title(
            f"{model_display_name(shap_bundle.model_key)} {label} SHAP Dependence\n"
            f"{main_feature} colored by {interaction_feature}"
        )
        _style_shap_axes(ax)
        return fig


def _write_importance_csv(artifacts: ArtifactManager, shap_bundle: ShapExplanationBundle) -> Path:
    return artifacts.write_dataframe(
        shap_bundle.importance,
        "explainability",
        f"shap_importance_{shap_bundle.dataset_regime}_{shap_bundle.model_key}.csv",
    )


def _beeswarm_outputs(artifacts: ArtifactManager, shap_bundle: ShapExplanationBundle, config: dict) -> list[Path]:
    import matplotlib.pyplot as plt
    import shap

    with report_figure_style_context(config):
        plt.figure()
        shap.summary_plot(
            shap_bundle.shap_values,
            shap_bundle.display_frame,
            show=False,
            max_display=config["reports"]["max_display_features"],
        )
        figure = plt.gcf()
        plt.tight_layout()
    try:
        return save_figure_variants(
            figure,
            artifacts,
            FigureExportSpec(stem=f"shap_beeswarm_{shap_bundle.dataset_regime}_{shap_bundle.model_key}"),
            config,
        )
    finally:
        plt.close(figure)


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    seen: set[str] = set()
    ordered: list[str] = []
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _resolve_scatter_features(job: dict, shap_bundle: ShapExplanationBundle) -> list[str]:
    requested = _dedupe_preserve_order(list(job.get("scatter_features", [])))
    available = set(shap_bundle.feature_names)
    if requested:
        missing = sorted(set(requested) - available)
        if missing:
            raise ValueError(
                f"SHAP scatter requested unknown features for {shap_bundle.dataset_regime}/{shap_bundle.model_key}: {missing}"
            )
        return requested
    ranked = shap_bundle.importance["feature"].astype(str).tolist()
    featured = [
        feature_name
        for feature_name in _dedupe_preserve_order(list(job.get("featured_scatter_features", [])))
        if feature_name in available
    ]
    featured_set = set(featured)
    return featured + [feature_name for feature_name in ranked if feature_name not in featured_set]


def _resolve_dependence_pairs(job: dict, shap_bundle: ShapExplanationBundle) -> list[tuple[str, str]]:
    available = set(shap_bundle.feature_names)
    pairs: list[tuple[str, str]] = []
    for pair in job.get("dependence_pairs", []):
        main_feature = str(pair["main_feature"]).strip()
        interaction_feature = str(pair["interaction_feature"]).strip()
        if main_feature == interaction_feature:
            raise ValueError(
                f"SHAP dependence plot requires distinct features for {shap_bundle.dataset_regime}/{shap_bundle.model_key}: {pair!r}"
            )
        missing = sorted({main_feature, interaction_feature} - available)
        if missing:
            raise ValueError(
                "SHAP dependence requested unknown features for "
                f"{shap_bundle.dataset_regime}/{shap_bundle.model_key}: {missing}"
            )
        pairs.append((main_feature, interaction_feature))
    return pairs


def _scatter_outputs(artifacts: ArtifactManager, shap_bundle: ShapExplanationBundle, config: dict, job: dict) -> list[Path]:
    import matplotlib.pyplot as plt

    outputs: list[Path] = []
    featured_features = set(job.get("featured_scatter_features", []))
    for feature_name in _resolve_scatter_features(job, shap_bundle):
        figure = _scatter_figure(shap_bundle, feature_name, config)
        try:
            outputs.extend(
                save_figure_variants(
                    figure,
                    artifacts,
                    FigureExportSpec(
                        stem=f"shap_scatter_{shap_bundle.dataset_regime}_{shap_bundle.model_key}_{_safe_name(feature_name)}",
                        directory_parts=("reports", "figures", "shap_scatter"),
                    ),
                    config,
                )
            )
            if feature_name in featured_features:
                outputs.extend(
                    save_figure_variants(
                        figure,
                        artifacts,
                        FigureExportSpec(
                            stem=(
                                f"manuscript_shap_scatter_{shap_bundle.dataset_regime}_{shap_bundle.model_key}_"
                                f"{_safe_name(feature_name)}"
                            ),
                            directory_parts=("reports", "figures", "shap_scatter_featured"),
                        ),
                        config,
                    )
                )
        finally:
            plt.close(figure)
    return outputs


def _dependence_outputs(artifacts: ArtifactManager, shap_bundle: ShapExplanationBundle, config: dict, job: dict) -> list[Path]:
    import matplotlib.pyplot as plt

    outputs: list[Path] = []
    for main_feature, interaction_feature in _resolve_dependence_pairs(job, shap_bundle):
        figure = _dependence_figure(shap_bundle, main_feature, interaction_feature, config)
        try:
            outputs.extend(
                save_figure_variants(
                    figure,
                    artifacts,
                    FigureExportSpec(
                        stem=(
                            f"shap_dependence_{shap_bundle.dataset_regime}_{shap_bundle.model_key}_"
                            f"{_safe_name(main_feature)}_vs_{_safe_name(interaction_feature)}"
                        ),
                        directory_parts=("reports", "figures", "shap_dependence"),
                    ),
                    config,
                )
            )
        finally:
            plt.close(figure)
    return outputs


def _generate_shap_job(config: dict, job: dict, *, shap_calc_workers: int = 1) -> list[str]:
    artifacts = ArtifactManager(config)
    runtime_plan = build_stage_runtime_plan(config, "report_shap")
    job_with_featured = {
        **job,
        "featured_scatter_features": list(config.get("reports", {}).get("featured_shap_scatter_features", [])),
    }
    dataset_regime = job_with_featured["dataset_regime"]
    model_key = job_with_featured["model_key"]
    with thread_limited_context(runtime_plan.nested_blas_threads):
        shap_bundle = _build_shap_explanation_bundle(
            config,
            dataset_regime,
            model_key,
            shap_calc_workers=shap_calc_workers,
        )
    if shap_bundle is None:
        return []

    outputs: list[Path] = [_write_importance_csv(artifacts, shap_bundle)]
    plots = set(job_with_featured.get("plots", ["beeswarm"]))
    if "beeswarm" in plots:
        outputs.extend(_beeswarm_outputs(artifacts, shap_bundle, config))
    if "scatter" in plots:
        outputs.extend(_scatter_outputs(artifacts, shap_bundle, config, job_with_featured))
    if "dependence" in plots:
        outputs.extend(_dependence_outputs(artifacts, shap_bundle, config, job_with_featured))
    return [str(path) for path in outputs]


def _resolve_shap_parallelism(job_count: int, worker_budget: int) -> tuple[int, int]:
    total_workers = max(1, int(worker_budget))
    total_jobs = max(1, int(job_count))
    if total_jobs == 1:
        return 1, total_workers

    # Keep both the SHAP-compute phase and the plot-writing phase busy by
    # balancing breadth across jobs with depth inside each active job.
    outer_jobs = min(total_jobs, max(1, int(round(total_workers**0.5))))
    per_job_shap_workers = max(1, total_workers // outer_jobs)
    return outer_jobs, per_job_shap_workers


def generate_shap_outputs(artifacts: ArtifactManager, config: dict) -> list[Path]:
    outputs: list[Path] = []
    jobs = config["reports"]["shap_jobs"]
    if not jobs:
        return outputs

    for job in jobs:
        dataset_regime = job["dataset_regime"]
        model_key = job["model_key"]
        if model_key not in SUPPORTED_SHAP_MODELS:
            raise ValueError(f"Unsupported SHAP model_key '{model_key}'. Supported SHAP models: {list(SUPPORTED_SHAP_MODELS)}.")
        bundle_dir = _shap_bundle_dir(artifacts, dataset_regime, model_key)
        bundle_path = bundle_dir / "bundle.joblib"
        if not bundle_path.exists():
            raise FileNotFoundError(f"Expected SHAP model artifact was not found: {bundle_path}")

    runtime_plan = build_stage_runtime_plan(config, "report_shap")
    outer_jobs, per_job_shap_workers = _resolve_shap_parallelism(len(jobs), runtime_plan.shap_workers)
    outputs_nested = Parallel(n_jobs=outer_jobs, backend="loky")(
        delayed(_generate_shap_job)(config, job, shap_calc_workers=per_job_shap_workers)
        for job in jobs
    )
    for job_outputs in outputs_nested:
        outputs.extend(Path(path) for path in job_outputs)
    return outputs
