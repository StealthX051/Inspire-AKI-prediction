from __future__ import annotations

import copy
import importlib
import importlib.util
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from inspire_aki.evaluation.split_manager import train_validation_split
from inspire_aki.models.weighting import balance_sample_weights, balance_weight_series, positive_balance_weight, safe_balanced_accuracy, weighted_resample_for_knn
from inspire_aki.runtime import build_stage_runtime_plan, configure_torch_threads, thread_limited_context

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    import torch.optim as optim
except ImportError:  # pragma: no cover - optional dependency guard
    torch = None
    nn = None
    functional = None
    optim = None

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover - optional dependency guard
    xgb = None


_AUTOGLUON_SAMPLE_WEIGHT_COLUMN = "__inspire_sample_weight__"
_TABULAR_DATASET_REGIMES = ("preop", "intraop", "combined")
_TABULAR_EXECUTION_POLICY_ENV = "INSPIRE_AKI_EXECUTION_POLICY"
_TABULAR_MODEL_KEYS_ENV = "INSPIRE_AKI_MODEL_KEYS"
_TABULAR_DATASET_REGIMES_ENV = "INSPIRE_AKI_DATASET_REGIMES"
_TABULAR_IDENTIFIER_COLUMNS = {"op_id", "subject_id", "patient_id"}


@dataclass(frozen=True)
class TabularExecutionPolicy:
    hpo_parallel_by_regime: bool = False
    hpo_thread_cap: int | None = None
    train_parallel_by_repeat: bool = False
    train_thread_cap: int | None = None
    train_tol: float | None = None


@dataclass
class PreparedTabularFold:
    feature_cols: list[str]
    target: str
    train_df: pd.DataFrame
    test_df: pd.DataFrame
    x_train_scaled: pd.DataFrame
    x_test_scaled: pd.DataFrame
    y_train: np.ndarray
    sample_weights: np.ndarray
    scaler: StandardScaler


_SERIAL_EXECUTION_POLICIES: dict[str, TabularExecutionPolicy] = {
    "log_reg": TabularExecutionPolicy(hpo_thread_cap=1, train_thread_cap=1),
    "svm": TabularExecutionPolicy(hpo_thread_cap=1, train_thread_cap=1),
}

_OPTIMIZED_LOW_CPU_POLICIES: dict[str, TabularExecutionPolicy] = {
    "log_reg": TabularExecutionPolicy(hpo_thread_cap=4, train_thread_cap=4),
    "svm": TabularExecutionPolicy(
        hpo_parallel_by_regime=True,
        hpo_thread_cap=1,
        train_parallel_by_repeat=True,
        train_thread_cap=1,
        train_tol=0.01,
    ),
}


def tabular_feature_columns(df: pd.DataFrame, target: str) -> list[str]:
    excluded = set(_TABULAR_IDENTIFIER_COLUMNS)
    excluded.update({target, f"{target}_boolean", f"{target}_positive"})
    excluded.update({column for column in df.columns if str(column).endswith("_event_codes")})
    return [col for col in df.columns if col not in excluded]


def tabular_execution_policy_name() -> str:
    value = os.environ.get(_TABULAR_EXECUTION_POLICY_ENV, "optimized_low_cpu").strip() or "optimized_low_cpu"
    return value if value in {"optimized_low_cpu", "serial"} else "optimized_low_cpu"


def tabular_execution_policy(model_key: str) -> TabularExecutionPolicy:
    if tabular_execution_policy_name() == "serial":
        return _SERIAL_EXECUTION_POLICIES.get(model_key, TabularExecutionPolicy())
    return _OPTIMIZED_LOW_CPU_POLICIES.get(model_key, TabularExecutionPolicy())


def selected_tabular_models(config: dict[str, Any], stage: str) -> list[str]:
    raw = os.environ.get(_TABULAR_MODEL_KEYS_ENV, "")
    if raw.strip():
        return [item.strip() for item in raw.split(",") if item.strip()]
    if stage == "tune":
        return list(config["models"]["tabular_hpo_enabled"])
    if stage == "train":
        return list(config["models"]["tabular_enabled"])
    raise ValueError(f"Unsupported tabular stage '{stage}'.")


def selected_tabular_dataset_regimes() -> list[str]:
    raw = os.environ.get(_TABULAR_DATASET_REGIMES_ENV, "")
    if raw.strip():
        return [item.strip() for item in raw.split(",") if item.strip()]
    return list(_TABULAR_DATASET_REGIMES)


def _autogluon_num_gpus(config: dict[str, Any]) -> int | str:
    num_gpus = config["models"]["autogluon"].get("num_gpus", "auto")
    return "auto" if num_gpus == "auto" else int(float(num_gpus))


def _module_available(module_name: str) -> bool:
    return importlib.util.find_spec(module_name) is not None


def _autogluon_model_type_available(model_type: str, module_name: str, checker_name: str) -> bool:
    if not _module_available(module_name):
        return False
    try:
        try_import_module = importlib.import_module("autogluon.common.utils.try_import")
        getattr(try_import_module, checker_name)()
    except (ImportError, ModuleNotFoundError, AttributeError):
        return False
    return True


def _autogluon_excluded_model_types() -> list[str]:
    excluded: list[str] = []
    optional_model_modules = {
        "CAT": ("catboost", "try_import_catboost"),
        "FASTAI": ("fastai", "try_import_fastai"),
        "XGB": ("xgboost", "try_import_xgboost"),
    }
    for model_type, (module_name, checker_name) in optional_model_modules.items():
        if not _autogluon_model_type_available(model_type, module_name, checker_name):
            excluded.append(model_type)
    return excluded


def _as_float32_array(frame: pd.DataFrame | np.ndarray) -> np.ndarray:
    if isinstance(frame, pd.DataFrame):
        return frame.to_numpy(dtype=np.float32, copy=False)
    return np.asarray(frame, dtype=np.float32)


if nn is not None:
    class PyTorchMLP(nn.Module):
        def __init__(self, input_size: int, n_layers: int, n_units: int, dropout_rate: float):
            super().__init__()
            layers = []
            in_features = input_size
            for _ in range(n_layers):
                layers.append(nn.Linear(in_features, n_units))
                layers.append(nn.ReLU())
                layers.append(nn.Dropout(dropout_rate))
                in_features = n_units
            layers.append(nn.Linear(in_features, 1))
            self.layers = nn.Sequential(*layers)

        def forward(self, tensor: torch.Tensor) -> torch.Tensor:
            return self.layers(tensor)
else:  # pragma: no cover - optional dependency guard
    class PyTorchMLP:
        def __init__(self, *_args, **_kwargs):
            raise ImportError("torch is required for the MLP tabular model.")


@dataclass
class FittedTabularBundle:
    model_key: str
    feature_names: list[str]
    scaler: StandardScaler | None
    model: Any
    metadata: dict[str, Any]


def _scale_if_needed(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    model_key: str,
) -> tuple[pd.DataFrame, pd.DataFrame, StandardScaler | None]:
    if model_key in {"autogluon", "asa_rule"}:
        return train_df[feature_cols].copy(), test_df[feature_cols].copy(), None
    scaler = StandardScaler()
    x_train = pd.DataFrame(
        scaler.fit_transform(train_df[feature_cols]),
        columns=feature_cols,
        index=train_df.index,
    )
    x_test = pd.DataFrame(
        scaler.transform(test_df[feature_cols]),
        columns=feature_cols,
        index=test_df.index,
    )
    return x_train, x_test, scaler


def prepare_tabular_fold(
    *,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    feature_cols: list[str],
    target: str,
) -> PreparedTabularFold:
    scaler = StandardScaler()
    x_train_scaled = pd.DataFrame(
        scaler.fit_transform(train_df[feature_cols]),
        columns=feature_cols,
        index=train_df.index,
    )
    x_test_scaled = pd.DataFrame(
        scaler.transform(test_df[feature_cols]),
        columns=feature_cols,
        index=test_df.index,
    )
    y_train = train_df[target].to_numpy()
    return PreparedTabularFold(
        feature_cols=list(feature_cols),
        target=target,
        train_df=train_df.copy(),
        test_df=test_df.copy(),
        x_train_scaled=x_train_scaled,
        x_test_scaled=x_test_scaled,
        y_train=y_train,
        sample_weights=balance_sample_weights(y_train),
        scaler=scaler,
    )


def fit_tabular_model(
    *,
    model_key: str,
    train_df: pd.DataFrame,
    feature_cols: list[str],
    target: str,
    params: dict[str, Any],
    config: dict,
    model_output_dir: Path,
    seed: int,
    prepared_fold: PreparedTabularFold | None = None,
    thread_cap: int | None = None,
) -> FittedTabularBundle:
    runtime_plan = build_stage_runtime_plan(config, "train_tabular")
    training_workers = runtime_plan.train_model_threads
    policy = tabular_execution_policy(model_key)
    effective_thread_cap = thread_cap if thread_cap is not None else policy.train_thread_cap
    if prepared_fold is None or model_key in {"autogluon", "asa_rule"}:
        x_train, _, scaler = _scale_if_needed(train_df, train_df, feature_cols, model_key)
        y_train = train_df[target].values
        sample_weights = balance_sample_weights(y_train)
    else:
        x_train = prepared_fold.x_train_scaled
        scaler = prepared_fold.scaler
        y_train = prepared_fold.y_train
        sample_weights = prepared_fold.sample_weights
    model_output_dir.mkdir(parents=True, exist_ok=True)

    if model_key == "log_reg":
        model = LogisticRegression(
            **params,
            max_iter=10000,
            tol=0.01,
            random_state=seed,
        )
        with thread_limited_context(effective_thread_cap or runtime_plan.nested_blas_threads):
            model.fit(x_train, y_train, sample_weight=sample_weights)
    elif model_key == "rf":
        model = RandomForestClassifier(**params, n_jobs=training_workers, random_state=seed)
        model.fit(x_train, y_train, sample_weight=sample_weights)
    elif model_key == "xgb":
        if xgb is None:
            raise ImportError("xgboost is required for the xgb model.")
        xgb_params = dict(params)
        xgb_params.setdefault("objective", "binary:logistic")
        xgb_params.setdefault("eval_metric", "logloss")
        xgb_params.setdefault("random_state", seed)
        xgb_params.setdefault("n_jobs", training_workers)
        if runtime_plan.xgb_use_gpu and torch is not None and torch.cuda.is_available():
            xgb_params.setdefault("device", "cuda")
        model = xgb.XGBClassifier(**xgb_params)
        model.fit(x_train, y_train, sample_weight=sample_weights, verbose=False)
    elif model_key == "svm":
        svm_params = dict(params)
        if policy.train_tol is not None and "tol" not in svm_params:
            svm_params["tol"] = policy.train_tol
        model = LinearSVC(**svm_params, dual="auto", random_state=seed, max_iter=5000)
        with thread_limited_context(effective_thread_cap or runtime_plan.nested_blas_threads):
            model.fit(x_train, y_train, sample_weight=sample_weights)
    elif model_key == "knn":
        model = KNeighborsClassifier(**params, weights="distance", n_jobs=training_workers)
        x_train_resampled, y_train_resampled = weighted_resample_for_knn(x_train, y_train, seed=seed)
        model.fit(x_train_resampled, y_train_resampled)
    elif model_key == "mlp":
        if torch is None or optim is None or functional is None:
            raise ImportError("torch is required for the MLP tabular model.")
        configure_torch_threads(config, stage="train_tabular")
        lr = params.get("lr", 0.001)
        n_layers = params.get("n_layers", 2)
        n_units = params.get("n_units", 32)
        dropout_rate = params.get("dropout_rate", 0.5)
        device = torch.device("cuda" if runtime_plan.sequence_use_gpu and torch.cuda.is_available() else "cpu")
        validation_source = train_df.copy()
        validation_source.loc[:, feature_cols] = x_train.to_numpy()
        train_part_df, val_part_df = train_validation_split(
            validation_source,
            target=target,
            validation_fraction=config["splits"]["tabular_mlp_validation_fraction"],
            random_state=seed,
            evaluation_mode=config.get("evaluation_mode", "legacy_repeated_cv"),
        )
        x_train_part = train_part_df[feature_cols].to_numpy()
        x_val_part = val_part_df[feature_cols].to_numpy()
        y_train_part = train_part_df[target].to_numpy()
        y_val_part = val_part_df[target].to_numpy()
        x_train_tensor = torch.FloatTensor(x_train_part).to(device)
        y_train_tensor = torch.FloatTensor(y_train_part).reshape(-1, 1).to(device)
        x_val_tensor = torch.FloatTensor(x_val_part).to(device)
        model = PyTorchMLP(len(feature_cols), n_layers, n_units, dropout_rate).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        train_positive_weight = positive_balance_weight(y_train_part)
        train_weight_tensor = torch.FloatTensor(
            balance_sample_weights(y_train_part, positive_weight=train_positive_weight)
        ).reshape(-1, 1).to(device)
        best_val_metric, patience_counter, best_model_state = 0.0, 0, None
        for epoch in range(5000):
            model.train()
            optimizer.zero_grad()
            outputs = model(x_train_tensor)
            loss = functional.binary_cross_entropy_with_logits(outputs, y_train_tensor, weight=train_weight_tensor)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    val_probs = torch.sigmoid(model(x_val_tensor)).cpu().numpy().flatten()
                current_val_metric = safe_balanced_accuracy(y_val_part, (val_probs >= 0.5).astype(int))
                if current_val_metric > best_val_metric:
                    best_val_metric = current_val_metric
                    patience_counter = 0
                    best_model_state = copy.deepcopy(model.state_dict())
                else:
                    patience_counter += 1
                if patience_counter >= 20:
                    break
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
    elif model_key == "autogluon":
        from autogluon.tabular import TabularPredictor

        train_ag = train_df[feature_cols + [target]].copy()
        train_ag[_AUTOGLUON_SAMPLE_WEIGHT_COLUMN] = balance_weight_series(train_ag[target].astype(int))
        ag_cfg = config["models"]["autogluon"]
        fit_kwargs: dict[str, Any] = {
            "train_data": train_ag,
            "time_limit": ag_cfg["time_limit_seconds"],
            "presets": ag_cfg["presets"],
            "num_cpus": training_workers if ag_cfg.get("num_cpus", "auto") == "auto" else ag_cfg["num_cpus"],
            "num_gpus": _autogluon_num_gpus(config),
            # DyStack launches a Ray sub-fit and has been the unstable failure point on this host.
            "dynamic_stacking": False,
            "fit_strategy": "sequential",
        }
        excluded_model_types = _autogluon_excluded_model_types()
        if excluded_model_types:
            fit_kwargs["excluded_model_types"] = excluded_model_types
        model = TabularPredictor(
            label=target,
            eval_metric="balanced_accuracy",
            path=str(model_output_dir),
            sample_weight=_AUTOGLUON_SAMPLE_WEIGHT_COLUMN,
        )
        model.fit(**fit_kwargs)
    elif model_key == "asa_rule":
        model = None
    else:
        raise ValueError(f"Unsupported tabular model key: {model_key}")

    metadata = {"seed": seed, "target": target}
    bundle = FittedTabularBundle(model_key=model_key, feature_names=feature_cols, scaler=scaler, model=model, metadata=metadata)
    save_tabular_bundle(bundle, model_output_dir)
    return bundle


def predict_tabular_bundle(
    bundle: FittedTabularBundle,
    test_df: pd.DataFrame,
    target: str,
    prepared_fold: PreparedTabularFold | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    feature_cols = bundle.feature_names
    if bundle.scaler is None or bundle.model_key in {"autogluon", "asa_rule"}:
        x_test = test_df[feature_cols].copy()
    elif prepared_fold is not None and prepared_fold.feature_cols == feature_cols and prepared_fold.target == target:
        x_test = prepared_fold.x_test_scaled
    else:
        x_test = pd.DataFrame(
            bundle.scaler.transform(test_df[feature_cols]),
            columns=feature_cols,
            index=test_df.index,
        )

    if bundle.model_key == "autogluon":
        test_frame = test_df[feature_cols].copy()
        y_prob = bundle.model.predict_proba(test_frame, as_pandas=False)[:, 1]
        y_pred = bundle.model.predict(test_frame).values
    elif bundle.model_key == "log_reg":
        y_prob = bundle.model.predict_proba(x_test)[:, 1]
        y_pred = bundle.model.predict(x_test)
    elif bundle.model_key == "rf":
        y_prob = bundle.model.predict_proba(x_test)[:, 1]
        y_pred = bundle.model.predict(x_test)
    elif bundle.model_key == "xgb":
        y_prob = bundle.model.predict_proba(x_test)[:, 1]
        y_pred = bundle.model.predict(x_test)
    elif bundle.model_key == "svm":
        y_scores = bundle.model.decision_function(x_test)
        y_prob = 1.0 / (1.0 + np.exp(-y_scores))
        y_pred = (y_scores > 0).astype(int)
    elif bundle.model_key == "knn":
        y_prob = bundle.model.predict_proba(x_test)[:, 1]
        y_pred = bundle.model.predict(x_test)
    elif bundle.model_key == "mlp":
        if torch is None:
            raise ImportError("torch is required for the MLP tabular model.")
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bundle.model.eval()
        with torch.no_grad():
            x_test_tensor = torch.from_numpy(_as_float32_array(x_test)).to(device)
            y_prob = torch.sigmoid(bundle.model(x_test_tensor)).cpu().numpy().flatten()
        y_pred = (y_prob >= 0.5).astype(int)
    elif bundle.model_key == "asa_rule":
        asa_idx = feature_cols.index("asa")
        asa_values = test_df[feature_cols].iloc[:, asa_idx].to_numpy()
        y_prob = (asa_values / 6.0).astype(float)
        y_pred = (asa_values >= 4).astype(int)
    else:
        raise ValueError(f"Unsupported tabular model key: {bundle.model_key}")
    return y_pred, y_prob


def save_tabular_bundle(bundle: FittedTabularBundle, model_output_dir: Path) -> None:
    if bundle.model_key == "autogluon":
        meta_path = model_output_dir / "bundle.joblib"
        joblib.dump({"model_key": bundle.model_key, "feature_names": bundle.feature_names, "metadata": bundle.metadata}, meta_path)
        return
    path = model_output_dir / "bundle.joblib"
    joblib.dump(bundle, path)


def load_tabular_bundle(model_key: str, model_output_dir: Path) -> FittedTabularBundle:
    if model_key == "autogluon":
        from autogluon.tabular import TabularPredictor

        meta = joblib.load(model_output_dir / "bundle.joblib")
        predictor = TabularPredictor.load(str(model_output_dir))
        return FittedTabularBundle(
            model_key=model_key,
            feature_names=meta["feature_names"],
            scaler=None,
            model=predictor,
            metadata=meta["metadata"],
        )
    return joblib.load(model_output_dir / "bundle.joblib")


def raw_prediction_rows(
    *,
    dataset_regime: str,
    population_id: str,
    model_key: str,
    target: str,
    repeat_id: int,
    fold_id: int,
    test_df: pd.DataFrame,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> pd.DataFrame:
    frame = pd.DataFrame(
        {
            "op_id": test_df["op_id"].values,
            "dataset_regime": dataset_regime,
            "population_id": population_id,
            "repeat_id": repeat_id,
            "fold_id": fold_id,
            "split_name": "test",
            "model_key": model_key,
            "target": target,
            "y_true": test_df[target].astype(int).values,
            "y_prob_raw": y_prob,
            "y_prob_calibrated": np.nan,
            "y_pred": y_pred.astype(int),
            "threshold": 0.5,
            "calibration_method": None,
            "run_id": [f"{dataset_regime}:{model_key}:r{repeat_id}:f{fold_id}"] * len(test_df),
            "source_index": test_df.index.values,
        }
    )
    return frame
