from __future__ import annotations

import hashlib
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from inspire_aki.datasets.splits import subset_from_manifest
from inspire_aki.models.sequence import HybridModel, _enable_sequence_cuda_benchmark, _sequence_loader_kwargs
from inspire_aki.models.tabular import PyTorchMLP, fit_tabular_model, tabular_execution_policy, tabular_feature_columns
from inspire_aki.models.weighting import balance_sample_weights, positive_balance_weight, safe_balanced_accuracy, weighted_resample_for_knn
from inspire_aki.runtime import build_stage_runtime_plan, configure_torch_threads, thread_limited_context


def _hpo_cfg(config: dict) -> dict[str, Any]:
    return config.get("models", {}).get("hpo", {})


def _trial_state_name(state: Any) -> str:
    if state is None:
        return ""
    name = getattr(state, "name", None)
    if name:
        return str(name)
    try:
        from optuna.trial import TrialState

        if state == TrialState.COMPLETE:
            return "COMPLETE"
    except Exception:
        pass
    state_text = str(state)
    if state_text.isdigit():
        return "COMPLETE" if int(state_text) == 1 else state_text
    if "." in state_text:
        return state_text.rsplit(".", 1)[-1]
    return state_text


def _has_completed_trials(study: Any) -> bool:
    return any(_trial_state_name(getattr(trial, "state", None)) == "COMPLETE" for trial in getattr(study, "trials", []))


def _stable_study_seed(config: dict, dataset_regime: str, model_key: str) -> int:
    seed_input = f"{dataset_regime}::{model_key}".encode("utf-8")
    stable_hash = int(hashlib.sha256(seed_input).hexdigest()[:8], 16)
    return int(config["splits"]["random_state"]) + (stable_hash % 100_000)


def tune_tabular_model(
    df: pd.DataFrame,
    dataset_regime: str,
    manifest: pd.DataFrame,
    config: dict,
    *,
    model_key: str,
    progress_callback: Any | None = None,
    sampler_seed: int | None = None,
) -> tuple[dict[str, Any], pd.DataFrame]:
    import optuna

    target = config["models"]["target"]
    train_df = subset_from_manifest(df, manifest, repeat_id=0, fold_id=0, split_name="train")
    val_df = subset_from_manifest(df, manifest, repeat_id=0, fold_id=0, split_name="val")
    feature_cols = tabular_feature_columns(df, target)
    x_train = train_df[feature_cols].copy()
    y_train = train_df[target].values
    x_val = val_df[feature_cols].copy()
    y_val = val_df[target].values
    runtime_plan = build_stage_runtime_plan(config, "tune_tabular")
    scaler = StandardScaler()
    with thread_limited_context(runtime_plan.nested_blas_threads):
        x_train_scaled = pd.DataFrame(scaler.fit_transform(x_train), columns=feature_cols, index=train_df.index)
        x_val_scaled = pd.DataFrame(scaler.transform(x_val), columns=feature_cols, index=val_df.index)
    training_workers = runtime_plan.hpo_model_threads
    sample_weights = balance_sample_weights(y_train)
    search_spaces = config["models"]["tabular_hpo_search_spaces"]
    hpo_cfg = _hpo_cfg(config)
    policy = tabular_execution_policy(model_key)

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.svm import LinearSVC
    import xgboost as xgb

    try:
        import torch
        import torch.nn as nn
        import torch.optim as optim
    except ImportError:  # pragma: no cover - optional dependency guard
        torch = None
        nn = None
        optim = None

    def objective(trial: optuna.Trial) -> float:
        if model_key == "log_reg":
            model = LogisticRegression(
                C=trial.suggest_float("C", *search_spaces["log_reg"]["C"], log=True),
                penalty="l2",
                solver="lbfgs",
                tol=0.001,
                max_iter=1000,
                random_state=config["splits"]["random_state"],
            )
            with thread_limited_context(policy.hpo_thread_cap or runtime_plan.nested_blas_threads):
                model.fit(x_train_scaled, y_train, sample_weight=sample_weights)
            return safe_balanced_accuracy(y_val, model.predict(x_val_scaled))
        if model_key == "xgb":
            params = {
                "objective": "binary:logistic",
                "eval_metric": "logloss",
                "random_state": config["splits"]["random_state"],
                "n_jobs": training_workers,
                "n_estimators": trial.suggest_int("n_estimators", *search_spaces["xgb"]["n_estimators"]),
                "learning_rate": trial.suggest_float("learning_rate", *search_spaces["xgb"]["learning_rate"]),
                "max_depth": trial.suggest_int("max_depth", *search_spaces["xgb"]["max_depth"]),
                "subsample": trial.suggest_float("subsample", *search_spaces["xgb"]["subsample"]),
                "colsample_bytree": trial.suggest_float("colsample_bytree", *search_spaces["xgb"]["colsample_bytree"]),
                "gamma": trial.suggest_float("gamma", *search_spaces["xgb"]["gamma"]),
            }
            if runtime_plan.xgb_use_gpu and torch is not None and torch.cuda.is_available():
                params["device"] = "cuda"
            model = xgb.XGBClassifier(**params)
            model.fit(x_train_scaled, y_train, sample_weight=sample_weights, verbose=False)
            return safe_balanced_accuracy(y_val, model.predict(x_val_scaled))
        if model_key == "rf":
            model = RandomForestClassifier(
                n_jobs=training_workers,
                random_state=config["splits"]["random_state"],
                n_estimators=trial.suggest_int("n_estimators", *search_spaces["rf"]["n_estimators"]),
                max_depth=trial.suggest_int("max_depth", *search_spaces["rf"]["max_depth"], log=True),
                min_samples_split=trial.suggest_int("min_samples_split", *search_spaces["rf"]["min_samples_split"]),
                min_samples_leaf=trial.suggest_int("min_samples_leaf", *search_spaces["rf"]["min_samples_leaf"]),
                max_features=trial.suggest_categorical("max_features", ["sqrt", "log2"]),
            )
            model.fit(x_train_scaled, y_train, sample_weight=sample_weights)
            return safe_balanced_accuracy(y_val, model.predict(x_val_scaled))
        if model_key == "svm":
            model = LinearSVC(
                C=trial.suggest_float("C", *search_spaces["svm"]["C"], log=True),
                random_state=config["splits"]["random_state"],
                dual="auto",
                max_iter=5000,
            )
            with thread_limited_context(policy.hpo_thread_cap or runtime_plan.nested_blas_threads):
                model.fit(x_train_scaled, y_train, sample_weight=sample_weights)
            return safe_balanced_accuracy(y_val, model.predict(x_val_scaled))
        if model_key == "knn":
            model = KNeighborsClassifier(
                n_neighbors=trial.suggest_int("n_neighbors", *search_spaces["knn"]["n_neighbors"]),
                weights="distance",
                n_jobs=training_workers,
            )
            x_train_resampled, y_train_resampled = weighted_resample_for_knn(
                x_train_scaled,
                y_train,
                seed=config["splits"]["random_state"],
            )
            model.fit(x_train_resampled, y_train_resampled)
            return safe_balanced_accuracy(y_val, model.predict(x_val_scaled))
        if model_key == "mlp":
            if torch is None or nn is None or optim is None:
                raise ImportError("torch is required for MLP HPO.")
            configure_torch_threads(config, stage="hpo")
            lr = trial.suggest_float("lr", *search_spaces["mlp"]["lr"], log=True)
            n_layers = trial.suggest_int("n_layers", *search_spaces["mlp"]["n_layers"])
            n_units = trial.suggest_int("n_units", *search_spaces["mlp"]["n_units"])
            dropout_rate = trial.suggest_float("dropout_rate", *search_spaces["mlp"]["dropout_rate"])
            device = torch.device("cuda" if runtime_plan.sequence_use_gpu and torch.cuda.is_available() else "cpu")
            x_train_tensor = torch.FloatTensor(x_train_scaled.to_numpy()).to(device)
            y_train_tensor = torch.FloatTensor(y_train).reshape(-1, 1).to(device)
            x_val_tensor = torch.FloatTensor(x_val_scaled.to_numpy()).to(device)
            model = PyTorchMLP(x_train_scaled.shape[1], n_layers, n_units, dropout_rate).to(device)
            optimizer = optim.Adam(model.parameters(), lr=lr)
            positive_weight = positive_balance_weight(y_train)
            train_weight_tensor = torch.FloatTensor(
                balance_sample_weights(y_train, positive_weight=positive_weight)
            ).reshape(-1, 1).to(device)
            for _ in range(int(hpo_cfg.get("tabular_mlp_epochs", 100))):
                model.train()
                outputs = model(x_train_tensor)
                loss = nn.functional.binary_cross_entropy_with_logits(outputs, y_train_tensor, weight=train_weight_tensor)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            model.eval()
            with torch.no_grad():
                val_outputs = model(x_val_tensor)
                val_probs = torch.sigmoid(val_outputs).cpu().numpy().flatten()
            return safe_balanced_accuracy(y_val, (val_probs >= 0.5).astype(int))
        raise ValueError(model_key)

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = None
    samplers_module = getattr(optuna, "samplers", None)
    if samplers_module is not None and hasattr(samplers_module, "TPESampler"):
        sampler = samplers_module.TPESampler(seed=sampler_seed or _stable_study_seed(config, dataset_regime, model_key))
    create_study_kwargs = {
        "direction": "maximize",
        "study_name": f"{dataset_regime}_{model_key}",
    }
    if sampler is not None:
        create_study_kwargs["sampler"] = sampler
    study = optuna.create_study(**create_study_kwargs)
    model_start = perf_counter()

    def _trial_callback(study: Any, trial: Any) -> None:
        if progress_callback is None:
            return
        progress_callback(
            dataset_regime=dataset_regime,
            model_key=model_key,
            study_key=f"{dataset_regime}::{model_key}",
            trial_number=int(trial.number),
            state=_trial_state_name(getattr(trial, "state", None)),
            value=trial.value,
            best_value=getattr(study, "best_value", None),
            elapsed_seconds=perf_counter() - model_start,
        )

    optimize_kwargs = {
        "n_trials": int(hpo_cfg.get("n_trials", 50)),
        "show_progress_bar": False,
        "callbacks": [_trial_callback],
    }
    try:
        study.optimize(objective, **optimize_kwargs)
    except TypeError:
        optimize_kwargs.pop("callbacks", None)
        study.optimize(objective, **optimize_kwargs)
    if not _has_completed_trials(study):
        raise RuntimeError(
            f"Tabular HPO completed no trials for dataset_regime='{dataset_regime}', model_key='{model_key}'. "
            "Consider relaxing the HPO search or pruning settings."
        )
    trials = pd.DataFrame(
        [
            {
                "dataset_regime": dataset_regime,
                "model_key": model_key,
                "trial_number": trial.number,
                "value": trial.value,
                "params": trial.params,
                "state": _trial_state_name(getattr(trial, "state", None)),
            }
            for trial in study.trials
        ]
    )
    return study.best_params, trials


def tune_tabular_dataset(
    df: pd.DataFrame,
    dataset_regime: str,
    manifest: pd.DataFrame,
    config: dict,
    *,
    progress_callback: Any | None = None,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    enabled = config["models"]["tabular_hpo_enabled"]
    if not enabled:
        return {}, pd.DataFrame()
    results: dict[str, dict[str, Any]] = {}
    trial_frames: list[pd.DataFrame] = []
    for model_key in enabled:
        best_params, trials_df = tune_tabular_model(
            df,
            dataset_regime,
            manifest,
            config,
            model_key=model_key,
            progress_callback=progress_callback,
            sampler_seed=_stable_study_seed(config, dataset_regime, model_key),
        )
        results[model_key] = best_params
        if not trials_df.empty:
            trial_frames.append(trials_df)
    return results, pd.concat(trial_frames, ignore_index=True) if trial_frames else pd.DataFrame()


def tune_sequence_dataset(
    df_sequence: pd.DataFrame,
    manifest: pd.DataFrame,
    config: dict,
    *,
    progress_callback: Any | None = None,
) -> tuple[dict[str, dict[str, Any]], pd.DataFrame]:
    target = config["models"]["target"]
    results: dict[str, dict[str, Any]] = {}
    trials_out: list[dict[str, Any]] = []
    enabled = config["models"]["sequence_hpo_enabled"]
    if not enabled:
        return {}, pd.DataFrame()

    import optuna

    try:
        import torch
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:  # pragma: no cover - optional dependency guard
        raise ImportError("torch is required for sequence HPO.") from exc

    search_spaces = config["models"]["sequence_hpo_search_spaces"]
    hpo_cfg = _hpo_cfg(config)
    feature_cols_tab = [col for col in df_sequence.columns if col not in ["op_id", "time_tensors", "seq_len", target]]
    train_df = subset_from_manifest(df_sequence, manifest, repeat_id=0, fold_id=0, split_name="train")
    val_df = subset_from_manifest(df_sequence, manifest, repeat_id=0, fold_id=0, split_name="val")
    scaler = StandardScaler()
    runtime_plan = build_stage_runtime_plan(config, "tune_sequence")
    with thread_limited_context(runtime_plan.nested_blas_threads):
        train_df.loc[:, feature_cols_tab] = scaler.fit_transform(train_df[feature_cols_tab])
        val_df.loc[:, feature_cols_tab] = scaler.transform(val_df[feature_cols_tab])

    def df_to_tensors(sub_df: pd.DataFrame):
        x_tab = torch.tensor(sub_df[feature_cols_tab].values, dtype=torch.float32)
        x_time = torch.stack([torch.as_tensor(tensor, dtype=torch.float32).clone().detach() for tensor in sub_df["time_tensors"]]).to(torch.float32)
        seq_len = torch.tensor(sub_df["seq_len"].tolist(), dtype=torch.long)
        y = torch.tensor(sub_df[target].values, dtype=torch.float32)
        return x_tab, x_time, seq_len, y

    train_dataset = TensorDataset(*df_to_tensors(train_df))
    val_dataset = TensorDataset(*df_to_tensors(val_df))
    configure_torch_threads(config, stage="hpo")
    loader_workers = runtime_plan.dataloader_workers
    batch_size = int(hpo_cfg.get("sequence_batch_size", 1024))
    device = torch.device("cuda" if runtime_plan.sequence_use_gpu and torch.cuda.is_available() else "cpu")
    _enable_sequence_cuda_benchmark(device)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        **_sequence_loader_kwargs(loader_workers=loader_workers, use_gpu=device.type == "cuda", shuffle=True),
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        **_sequence_loader_kwargs(loader_workers=loader_workers, use_gpu=device.type == "cuda", shuffle=False),
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    def objective_builder(model_key: str):
        def objective(trial: optuna.Trial) -> float:
            lr = trial.suggest_float("lr", *search_spaces[model_key]["lr"], log=True)
            dropout_rate = trial.suggest_float("dropout_rate", *search_spaces[model_key]["dropout_rate"])
            if model_key == "lstm_only":
                lstm_hidden_size = trial.suggest_int("lstm_hidden_size", *search_spaces[model_key]["lstm_hidden_size"])
                lstm_num_layers = trial.suggest_int("lstm_num_layers", *search_spaces[model_key]["lstm_num_layers"])
                mlp_dims: list[int] = []
            else:
                lstm_hidden_size = trial.suggest_int("lstm_hidden_size", *search_spaces[model_key]["lstm_hidden_size"])
                lstm_num_layers = trial.suggest_int("lstm_num_layers", *search_spaces[model_key]["lstm_num_layers"])
                n_mlp_layers = trial.suggest_int("n_mlp_layers", *search_spaces[model_key]["n_mlp_layers"])
                mlp_dims = [trial.suggest_int(f"mlp_layer_{i}_size", *search_spaces[model_key]["mlp_layer_size"]) for i in range(n_mlp_layers)]

            model = HybridModel(
                tabular_input_size=len(feature_cols_tab),
                lstm_input_size=train_dataset.tensors[1].shape[2],
                lstm_hidden_size=lstm_hidden_size,
                lstm_num_layers=lstm_num_layers,
                mlp_dims=mlp_dims,
                dropout_rate=dropout_rate,
                mode=model_key,
            ).to(device)
            optimizer = torch.optim.Adam(model.parameters(), lr=lr)
            y_train_numpy = train_loader.dataset.tensors[3].numpy()
            positive_weight = positive_balance_weight(y_train_numpy)

            best_val_metric = 0.0
            patience_counter = 0
            for epoch in range(int(hpo_cfg.get("sequence_epochs", 150))):
                model.train()
                for batch in train_loader:
                    x_tab_batch, x_time_batch, seq_len_batch, y_batch = [tensor.to(device) for tensor in batch]
                    optimizer.zero_grad()
                    outputs = model(x_tab_batch, x_time_batch, seq_len_batch)
                    batch_weights = torch.FloatTensor(
                        balance_sample_weights(y_batch.cpu().numpy(), positive_weight=positive_weight)
                    ).reshape(-1, 1).to(device)
                    loss = torch.nn.functional.binary_cross_entropy_with_logits(
                        outputs,
                        y_batch.unsqueeze(1),
                        weight=batch_weights,
                    )
                    loss.backward()
                    optimizer.step()

                model.eval()
                all_val_probs = []
                with torch.no_grad():
                    for batch in val_loader:
                        x_tab_batch, x_time_batch, seq_len_batch, _ = [tensor.to(device) for tensor in batch]
                        val_outputs = model(x_tab_batch, x_time_batch, seq_len_batch)
                        all_val_probs.append(torch.sigmoid(val_outputs).cpu())
                val_probs = torch.cat(all_val_probs).numpy().flatten()
                val_pred_binary = (val_probs >= 0.5).astype(int)
                current_val_metric = safe_balanced_accuracy(val_loader.dataset.tensors[3].numpy(), val_pred_binary)
                if current_val_metric > best_val_metric:
                    best_val_metric = current_val_metric
                    patience_counter = 0
                else:
                    patience_counter += 1
                trial.report(best_val_metric, epoch)
                if trial.should_prune() or patience_counter >= int(hpo_cfg.get("sequence_patience", 15)):
                    raise optuna.exceptions.TrialPruned()
            return best_val_metric

        return objective

    for model_key in enabled:
        study = optuna.create_study(direction="maximize", study_name=f"{model_key}_hpo")
        model_start = perf_counter()

        def _trial_callback(study: Any, trial: Any) -> None:
            if progress_callback is None:
                return
            progress_callback(
                dataset_regime="sequence_common",
                model_key=model_key,
                trial_number=int(trial.number),
                state=_trial_state_name(getattr(trial, "state", None)),
                value=trial.value,
                best_value=getattr(study, "best_value", None),
                elapsed_seconds=perf_counter() - model_start,
            )

        optimize_kwargs = {
            "n_trials": int(hpo_cfg.get("n_trials", 50)),
            "show_progress_bar": False,
            "callbacks": [_trial_callback],
        }
        try:
            study.optimize(objective_builder(model_key), **optimize_kwargs)
        except TypeError:
            optimize_kwargs.pop("callbacks", None)
            study.optimize(objective_builder(model_key), **optimize_kwargs)
        if not _has_completed_trials(study):
            raise RuntimeError(
                f"Sequence HPO completed no trials for model_key='{model_key}'. "
                "Consider increasing models.hpo.sequence_epochs or models.hpo.sequence_patience."
            )
        best_params = study.best_params
        if "n_mlp_layers" in best_params:
            mlp_dims = [best_params[f"mlp_layer_{i}_size"] for i in range(best_params["n_mlp_layers"])]
            final_params = {"mlp_dims": mlp_dims}
            for key, val in best_params.items():
                if not key.startswith("mlp_layer_") and key != "n_mlp_layers":
                    final_params[key] = val
            results[model_key] = final_params
        else:
            results[model_key] = best_params
        for trial in study.trials:
            trials_out.append(
                {
                    "dataset_regime": "sequence_common",
                    "model_key": model_key,
                    "trial_number": trial.number,
                    "value": trial.value,
                    "params": trial.params,
                    "state": _trial_state_name(getattr(trial, "state", None)),
                }
            )
    return results, pd.DataFrame(trials_out)
