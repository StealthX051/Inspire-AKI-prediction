from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from inspire_aki.evaluation.split_manager import train_validation_split
from inspire_aki.models.weighting import balance_sample_weights, positive_balance_weight, safe_balanced_accuracy
from inspire_aki.runtime import build_stage_runtime_plan, configure_torch_threads

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as functional
    import torch.optim as optim
    from torch.nn.utils.rnn import pack_padded_sequence
    from torch.utils.data import DataLoader, TensorDataset
except ImportError:  # pragma: no cover - optional dependency guard
    torch = None
    nn = None
    functional = None
    optim = None
    pack_padded_sequence = None
    DataLoader = None
    TensorDataset = None


@dataclass
class FittedSequenceBundle:
    model_key: str
    feature_names: list[str]
    scaler: StandardScaler | None
    model: Any
    time_input_size: int
    lstm_hidden_size: int
    lstm_num_layers: int
    dropout_rate: float
    mlp_dims: list[int]
    mode: str
    metadata: dict[str, Any]


SEQUENCE_BUNDLE_FORMAT_VERSION = 1
_SEQUENCE_IDENTIFIER_COLUMNS = {"op_id", "subject_id", "patient_id", "time_tensors", "seq_len"}


if nn is not None:
    class HybridModel(nn.Module):
        def __init__(
            self,
            tabular_input_size: int,
            lstm_input_size: int,
            lstm_hidden_size: int,
            lstm_num_layers: int,
            mlp_dims: list[int],
            dropout_rate: float,
            mode: str = "hybrid",
        ):
            super().__init__()
            self.mode = mode
            if self.mode in {"lstm_only", "hybrid"}:
                lstm_dropout = dropout_rate if lstm_num_layers > 1 else 0.0
                self.lstm = nn.LSTM(
                    input_size=lstm_input_size,
                    hidden_size=lstm_hidden_size,
                    num_layers=lstm_num_layers,
                    batch_first=True,
                    dropout=lstm_dropout,
                )
            if self.mode in {"mlp_only", "hybrid"}:
                mlp_layers = []
                in_features = tabular_input_size
                for dim in mlp_dims:
                    mlp_layers.extend([nn.Linear(in_features, dim), nn.ReLU(), nn.Dropout(dropout_rate)])
                    in_features = dim
                self.mlp = nn.Sequential(*mlp_layers) if mlp_layers else nn.Identity()
            if self.mode == "hybrid":
                classifier_input_size = lstm_hidden_size + (mlp_dims[-1] if mlp_dims else 0)
            elif self.mode == "lstm_only":
                classifier_input_size = lstm_hidden_size
            else:
                classifier_input_size = mlp_dims[-1] if mlp_dims else tabular_input_size
            self.classifier = nn.Sequential(
                nn.Linear(classifier_input_size, max(classifier_input_size // 2, 1)),
                nn.ReLU(),
                nn.Dropout(dropout_rate),
                nn.Linear(max(classifier_input_size // 2, 1), 1),
            )

        def forward(self, x_tab: torch.Tensor | None = None, x_time: torch.Tensor | None = None, seq_len: torch.Tensor | None = None) -> torch.Tensor:
            if self.mode == "hybrid":
                packed = pack_padded_sequence(x_time, seq_len.cpu(), batch_first=True, enforce_sorted=False)
                _, (hn, _) = self.lstm(packed)
                lstm_out = hn[-1]
                mlp_out = self.mlp(x_tab)
                combined = torch.cat((lstm_out, mlp_out), dim=1)
            elif self.mode == "lstm_only":
                packed = pack_padded_sequence(x_time, seq_len.cpu(), batch_first=True, enforce_sorted=False)
                _, (hn, _) = self.lstm(packed)
                combined = hn[-1]
            else:
                combined = self.mlp(x_tab)
            return self.classifier(combined)
else:  # pragma: no cover - optional dependency guard
    class HybridModel:
        def __init__(self, *_args, **_kwargs):
            raise ImportError("torch is required for sequence models.")


def sequence_feature_columns(df: pd.DataFrame, target: str) -> list[str]:
    excluded = set(_SEQUENCE_IDENTIFIER_COLUMNS)
    excluded.add(target)
    excluded.update({column for column in df.columns if str(column).endswith("_event_codes")})
    return [column for column in df.columns if column not in excluded]


def _df_to_tensors(
    sub_df: pd.DataFrame,
    feature_cols_tab: list[str],
    *,
    target: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    if torch is None:
        raise ImportError("torch is required for sequence models.")
    x_tab = torch.tensor(sub_df[feature_cols_tab].values, dtype=torch.float32)
    y = torch.tensor(sub_df[target].values, dtype=torch.float32)
    if "time_tensors" in sub_df.columns:
        x_time = torch.stack([torch.as_tensor(tensor, dtype=torch.float32).clone().detach() for tensor in sub_df["time_tensors"]]).to(torch.float32)
        seq_len = torch.tensor(sub_df["seq_len"].tolist(), dtype=torch.long)
    else:
        n_samples = len(sub_df)
        x_time = torch.zeros(n_samples, 1, 1, dtype=torch.float32)
        seq_len = torch.zeros(n_samples, dtype=torch.long)
    return x_tab, x_time, seq_len, y


def _enable_sequence_cuda_benchmark(device: Any) -> None:
    if torch is None or getattr(device, "type", None) != "cuda":
        return
    cudnn = getattr(getattr(torch, "backends", None), "cudnn", None)
    if cudnn is not None:
        cudnn.benchmark = True


def _sequence_loader_kwargs(*, loader_workers: int, use_gpu: bool, shuffle: bool) -> dict[str, Any]:
    num_workers = max(0, int(loader_workers))
    kwargs: dict[str, Any] = {
        "shuffle": shuffle,
        "pin_memory": bool(use_gpu),
        "num_workers": num_workers,
    }
    if num_workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return kwargs


def fit_sequence_model(
    *,
    model_key: str,
    train_df: pd.DataFrame,
    feature_cols_tab: list[str],
    target: str,
    params: dict[str, Any],
    config: dict,
    model_output_dir: Path,
    seed: int,
    progress_callback: Any | None = None,
) -> FittedSequenceBundle:
    if torch is None or optim is None or DataLoader is None or TensorDataset is None or functional is None:
        raise ImportError("torch is required for sequence models.")
    runtime_plan = build_stage_runtime_plan(config, "train_sequence")
    configure_torch_threads(config, stage="train_sequence")
    loader_workers = runtime_plan.dataloader_workers
    device = torch.device("cuda" if runtime_plan.sequence_use_gpu and torch.cuda.is_available() else "cpu")
    _enable_sequence_cuda_benchmark(device)
    model_output_dir.mkdir(parents=True, exist_ok=True)

    train_split, val_split = train_validation_split(
        train_df,
        target=target,
        validation_fraction=config["splits"]["sequence_validation_fraction"],
        random_state=seed,
        evaluation_mode=config.get("evaluation_mode", "legacy_repeated_cv"),
    )
    scaler = StandardScaler()
    train_split.loc[:, feature_cols_tab] = scaler.fit_transform(train_split[feature_cols_tab])
    val_split.loc[:, feature_cols_tab] = scaler.transform(val_split[feature_cols_tab])

    train_tensors = _df_to_tensors(train_split, feature_cols_tab, target=target)
    val_tensors = _df_to_tensors(val_split, feature_cols_tab, target=target)
    train_dataset = TensorDataset(*train_tensors)
    val_dataset = TensorDataset(*val_tensors)
    train_loader = DataLoader(
        train_dataset,
        batch_size=params["batch_size"],
        **_sequence_loader_kwargs(loader_workers=loader_workers, use_gpu=device.type == "cuda", shuffle=True),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=params["batch_size"],
        **_sequence_loader_kwargs(loader_workers=loader_workers, use_gpu=device.type == "cuda", shuffle=False),
    )

    lstm_input_size = train_tensors[1].shape[2] if train_tensors[1].numel() > 1 else 0
    model = HybridModel(
        tabular_input_size=len(feature_cols_tab),
        lstm_input_size=lstm_input_size,
        lstm_hidden_size=params["lstm_hidden_size"],
        lstm_num_layers=params["lstm_num_layers"],
        mlp_dims=params.get("mlp_dims", []),
        dropout_rate=params["dropout_rate"],
        mode=model_key,
    ).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params["learning_rate"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=params["lr_scheduler_factor"],
        patience=params["lr_scheduler_patience"],
    )
    y_train = train_loader.dataset.tensors[3].cpu().numpy()
    positive_weight = positive_balance_weight(y_train)

    start = perf_counter()
    best_val_metric, best_val_loss, patience_counter, best_state = 0.0, float("inf"), 0, None
    for epoch in range(params["epochs"]):
        model.train()
        for batch in train_loader:
            x_tab_batch, x_time_batch, seq_len_batch, y_batch = [tensor.to(device) for tensor in batch]
            optimizer.zero_grad()
            outputs = model(x_tab_batch, x_time_batch, seq_len_batch)
            batch_weights = torch.FloatTensor(
                balance_sample_weights(y_batch.cpu().numpy(), positive_weight=positive_weight)
            ).reshape(-1, 1).to(device)
            loss = functional.binary_cross_entropy_with_logits(outputs, y_batch.unsqueeze(1), weight=batch_weights)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=params["gradient_clip_value"])
            optimizer.step()
        if (epoch + 1) % params["es_check_interval"] == 0:
            model.eval()
            val_loss = 0.0
            all_val_probs = []
            with torch.no_grad():
                for batch in val_loader:
                    x_tab_batch, x_time_batch, seq_len_batch, y_batch = [tensor.to(device) for tensor in batch]
                    val_outputs = model(x_tab_batch, x_time_batch, seq_len_batch)
                    batch_weights = torch.FloatTensor(
                        balance_sample_weights(y_batch.cpu().numpy(), positive_weight=positive_weight)
                    ).reshape(-1, 1).to(device)
                    val_loss += functional.binary_cross_entropy_with_logits(
                        val_outputs,
                        y_batch.unsqueeze(1),
                        weight=batch_weights,
                    ).item()
                    all_val_probs.append(torch.sigmoid(val_outputs).cpu())
            avg_val_loss = val_loss / max(len(val_loader), 1)
            val_probs = torch.cat(all_val_probs).numpy().flatten()
            current_val_metric = safe_balanced_accuracy(val_loader.dataset.tensors[3].cpu().numpy(), (val_probs >= 0.5).astype(int))
            scheduler.step(current_val_metric)
            best_val_loss = min(best_val_loss, avg_val_loss)
            if current_val_metric > best_val_metric:
                best_val_metric = current_val_metric
                patience_counter = 0
                best_state = copy.deepcopy(model.state_dict())
            else:
                patience_counter += 1
            if progress_callback is not None:
                progress_callback(
                    epoch=epoch + 1,
                    val_loss=float(avg_val_loss),
                    best_val_loss=float(best_val_loss),
                    val_balanced_accuracy=float(current_val_metric),
                    best_val_balanced_accuracy=float(best_val_metric),
                    patience_counter=patience_counter,
                    learning_rate=float(optimizer.param_groups[0]["lr"]),
                    elapsed_seconds=perf_counter() - start,
                )
            if patience_counter >= params["patience"]:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    bundle = FittedSequenceBundle(
        model_key=model_key,
        feature_names=feature_cols_tab,
        scaler=scaler,
        model=model,
        time_input_size=lstm_input_size,
        lstm_hidden_size=params["lstm_hidden_size"],
        lstm_num_layers=params["lstm_num_layers"],
        dropout_rate=params["dropout_rate"],
        mlp_dims=list(params.get("mlp_dims", [])),
        mode=model_key,
        metadata={
            "seed": seed,
            "loader_workers": loader_workers,
            "sequence_use_gpu": runtime_plan.sequence_use_gpu,
            "batch_size": int(params["batch_size"]),
            "target": target,
        },
    )
    save_sequence_bundle(bundle, model_output_dir)
    return bundle


def predict_sequence_bundle(bundle: FittedSequenceBundle, test_df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    if torch is None or DataLoader is None or TensorDataset is None:
        raise ImportError("torch is required for sequence models.")
    loader_workers = int(bundle.metadata.get("loader_workers", 0))
    prefer_gpu = bool(bundle.metadata.get("sequence_use_gpu", True))
    device = torch.device("cuda" if prefer_gpu and torch.cuda.is_available() else "cpu")
    feature_cols = bundle.feature_names
    test_copy = test_df.copy()
    if bundle.scaler is not None:
        test_copy.loc[:, feature_cols] = bundle.scaler.transform(test_copy[feature_cols])
    target = str(bundle.metadata.get("target", "aki_boolean"))
    test_tensors = _df_to_tensors(test_copy, feature_cols, target=target)
    test_dataset = TensorDataset(*test_tensors)
    test_loader = DataLoader(
        test_dataset,
        batch_size=int(bundle.metadata.get("batch_size", 512)),
        **_sequence_loader_kwargs(loader_workers=loader_workers, use_gpu=device.type == "cuda", shuffle=False),
    )
    bundle.model = bundle.model.to(device)
    bundle.model.eval()
    all_probs = []
    with torch.no_grad():
        for batch in test_loader:
            x_tab_batch, x_time_batch, seq_len_batch, _ = [tensor.to(device) for tensor in batch]
            outputs = bundle.model(x_tab_batch, x_time_batch, seq_len_batch)
            all_probs.append(torch.sigmoid(outputs).cpu())
    y_prob = torch.cat(all_probs).numpy().flatten()
    y_pred = (y_prob >= 0.5).astype(int)
    return y_pred, y_prob


def save_sequence_bundle(bundle: FittedSequenceBundle, model_output_dir: Path) -> None:
    if torch is None:
        raise ImportError("torch is required for sequence models.")
    model_output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format_version": SEQUENCE_BUNDLE_FORMAT_VERSION,
            "model_key": bundle.model_key,
            "feature_names": bundle.feature_names,
            "time_input_size": bundle.time_input_size,
            "lstm_hidden_size": bundle.lstm_hidden_size,
            "lstm_num_layers": bundle.lstm_num_layers,
            "dropout_rate": bundle.dropout_rate,
            "mlp_dims": bundle.mlp_dims,
            "mode": bundle.mode,
            "scaler": bundle.scaler,
            "state_dict": bundle.model.state_dict(),
            "metadata": bundle.metadata,
        },
        model_output_dir / "bundle.pt",
    )


def load_sequence_bundle(model_output_dir: Path) -> FittedSequenceBundle:
    if torch is None:
        raise ImportError("torch is required for sequence models.")
    payload = torch.load(model_output_dir / "bundle.pt", map_location="cpu")
    format_version = int(payload.get("format_version", 0))
    if format_version != SEQUENCE_BUNDLE_FORMAT_VERSION:
        raise ValueError(
            f"Unsupported sequence bundle format_version {format_version}. "
            f"Expected {SEQUENCE_BUNDLE_FORMAT_VERSION}."
        )
    model = HybridModel(
        tabular_input_size=len(payload["feature_names"]),
        lstm_input_size=int(payload["time_input_size"]),
        lstm_hidden_size=int(payload["lstm_hidden_size"]),
        lstm_num_layers=int(payload["lstm_num_layers"]),
        mlp_dims=list(payload.get("mlp_dims", [])),
        dropout_rate=float(payload["dropout_rate"]),
        mode=payload.get("mode", payload["model_key"]),
    )
    model.load_state_dict(payload["state_dict"])
    model.eval()
    return FittedSequenceBundle(
        model_key=payload["model_key"],
        feature_names=list(payload["feature_names"]),
        scaler=payload.get("scaler"),
        model=model,
        time_input_size=int(payload["time_input_size"]),
        lstm_hidden_size=int(payload["lstm_hidden_size"]),
        lstm_num_layers=int(payload["lstm_num_layers"]),
        dropout_rate=float(payload["dropout_rate"]),
        mlp_dims=list(payload.get("mlp_dims", [])),
        mode=payload.get("mode", payload["model_key"]),
        metadata=dict(payload.get("metadata", {})),
    )


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
    split_name: str = "test",
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "op_id": test_df["op_id"].values,
            "patient_id": test_df["patient_id"].values if "patient_id" in test_df.columns else pd.Series(pd.NA, index=test_df.index, dtype="object").values,
            "dataset_regime": dataset_regime,
            "population_id": population_id,
            "repeat_id": repeat_id,
            "fold_id": fold_id,
            "split_name": split_name,
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
