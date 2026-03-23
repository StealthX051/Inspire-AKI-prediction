from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC

from inspire_aki.runtime import configure_torch_threads, worker_count

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except ImportError:  # pragma: no cover - optional dependency guard
    torch = None
    nn = None
    optim = None

try:
    import xgboost as xgb
except ImportError:  # pragma: no cover - optional dependency guard
    xgb = None


def tabular_feature_columns(df: pd.DataFrame, target: str) -> list[str]:
    return [col for col in df.columns if col not in ["op_id", target, f"{target}_boolean", f"{target}_positive"]]


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
) -> tuple[np.ndarray, np.ndarray, StandardScaler | None]:
    if model_key in {"autogluon", "asa_rule"}:
        return train_df[feature_cols].values, test_df[feature_cols].values, None
    scaler = StandardScaler()
    x_train = scaler.fit_transform(train_df[feature_cols].values)
    x_test = scaler.transform(test_df[feature_cols].values)
    return x_train, x_test, scaler


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
) -> FittedTabularBundle:
    training_workers = worker_count(config)
    x_train, _, scaler = _scale_if_needed(train_df, train_df, feature_cols, model_key)
    y_train = train_df[target].values
    model_output_dir.mkdir(parents=True, exist_ok=True)

    if model_key == "log_reg":
        model = LogisticRegression(
            **params,
            class_weight="balanced",
            max_iter=10000,
            tol=0.01,
            random_state=seed,
        )
        model.fit(x_train, y_train)
    elif model_key == "rf":
        model = RandomForestClassifier(**params, class_weight="balanced", n_jobs=training_workers, random_state=seed)
        model.fit(x_train, y_train)
    elif model_key == "xgb":
        if xgb is None:
            raise ImportError("xgboost is required for the xgb model.")
        xgb_params = dict(params)
        xgb_params.setdefault("objective", "binary:logistic")
        xgb_params.setdefault("eval_metric", "logloss")
        xgb_params.setdefault("random_state", seed)
        xgb_params.setdefault("n_jobs", training_workers)
        if torch is not None and torch.cuda.is_available():
            xgb_params.setdefault("device", "cuda")
        scale_pos_weight = np.sum(y_train == 0) / np.sum(y_train == 1) if np.sum(y_train == 1) > 0 else 1
        model = xgb.XGBClassifier(**xgb_params, scale_pos_weight=scale_pos_weight)
        model.fit(x_train, y_train, verbose=False)
    elif model_key == "svm":
        model = LinearSVC(**params, class_weight="balanced", dual="auto", random_state=seed, max_iter=5000)
        model.fit(x_train, y_train)
    elif model_key == "knn":
        model = KNeighborsClassifier(**params, weights="distance", n_jobs=training_workers)
        model.fit(x_train, y_train)
    elif model_key == "mlp":
        if torch is None or optim is None:
            raise ImportError("torch is required for the MLP tabular model.")
        configure_torch_threads(config)
        lr = params.get("lr", 0.001)
        n_layers = params.get("n_layers", 2)
        n_units = params.get("n_units", 32)
        dropout_rate = params.get("dropout_rate", 0.5)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        x_train_part, x_val_part, y_train_part, y_val_part = train_test_split(
            x_train,
            y_train,
            test_size=config["splits"]["tabular_mlp_validation_fraction"],
            random_state=seed,
            stratify=y_train,
        )
        x_train_tensor = torch.FloatTensor(x_train_part).to(device)
        y_train_tensor = torch.FloatTensor(y_train_part).reshape(-1, 1).to(device)
        x_val_tensor = torch.FloatTensor(x_val_part).to(device)
        model = PyTorchMLP(len(feature_cols), n_layers, n_units, dropout_rate).to(device)
        optimizer = optim.Adam(model.parameters(), lr=lr)
        pos_weight_val = np.sum(y_train_part == 0) / np.sum(y_train_part == 1) if np.sum(y_train_part == 1) > 0 else 1.0
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], device=device))
        best_val_auc, patience_counter, best_model_state = 0.0, 0, None
        for epoch in range(5000):
            model.train()
            optimizer.zero_grad()
            outputs = model(x_train_tensor)
            loss = criterion(outputs, y_train_tensor)
            loss.backward()
            optimizer.step()
            if (epoch + 1) % 10 == 0:
                model.eval()
                with torch.no_grad():
                    val_probs = torch.sigmoid(model(x_val_tensor)).cpu().numpy().flatten()
                if len(np.unique(y_val_part)) > 1:
                    from sklearn.metrics import roc_auc_score

                    current_val_auc = roc_auc_score(y_val_part, val_probs)
                else:
                    current_val_auc = 0.5
                if current_val_auc > best_val_auc:
                    best_val_auc = current_val_auc
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
        balance_weight = (train_ag[target].astype(int) * 17) + 1
        train_ag["balance_weight"] = balance_weight
        ag_cfg = config["models"]["autogluon"]
        model = TabularPredictor(
            label=target,
            eval_metric="balanced_accuracy",
            path=str(model_output_dir),
            sample_weight="balance_weight",
        )
        model.fit(
            train_data=train_ag,
            time_limit=ag_cfg["time_limit_seconds"],
            presets=ag_cfg["presets"],
            num_cpus=training_workers if ag_cfg.get("num_cpus", "auto") == "auto" else ag_cfg["num_cpus"],
        )
    elif model_key == "asa_rule":
        model = None
    else:
        raise ValueError(f"Unsupported tabular model key: {model_key}")

    metadata = {"seed": seed, "target": target}
    bundle = FittedTabularBundle(model_key=model_key, feature_names=feature_cols, scaler=scaler, model=model, metadata=metadata)
    save_tabular_bundle(bundle, model_output_dir)
    return bundle


def predict_tabular_bundle(bundle: FittedTabularBundle, test_df: pd.DataFrame, target: str) -> tuple[np.ndarray, np.ndarray]:
    feature_cols = bundle.feature_names
    if bundle.scaler is None or bundle.model_key in {"autogluon", "asa_rule"}:
        x_test = test_df[feature_cols].values
    else:
        x_test = bundle.scaler.transform(test_df[feature_cols].values)

    if bundle.model_key == "autogluon":
        test_frame = pd.DataFrame(test_df[feature_cols].values, columns=feature_cols)
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
            x_test_tensor = torch.FloatTensor(x_test).to(device)
            y_prob = torch.sigmoid(bundle.model(x_test_tensor)).cpu().numpy().flatten()
        y_pred = (y_prob >= 0.5).astype(int)
    elif bundle.model_key == "asa_rule":
        asa_idx = feature_cols.index("asa")
        y_prob = (test_df[feature_cols].values[:, asa_idx] / 6.0).astype(float)
        y_pred = (test_df[feature_cols].values[:, asa_idx] >= 4).astype(int)
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
