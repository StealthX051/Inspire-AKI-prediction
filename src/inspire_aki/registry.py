from __future__ import annotations

MODEL_DISPLAY_NAMES = {
    "base": "base",
    "base_54k": "base_54k",
    "log_reg": "Logistic Regression",
    "xgb": "GBT",
    "svm": "SVM (Linear)",
    "mlp": "MLP",
    "rf": "Random Forest",
    "knn": "KNN",
    "asa_rule": "ASA Rule",
    "autogluon": "AutoGluon",
    "lstm_only": "LSTM",
    "lstm": "LSTM",
    "hybrid": "Hybrid (MLP + LSTM)",
    "mlp_only": "MLP-only",
}

MODEL_SAFE_NAMES = {
    key: value.replace(" ", "_").replace("(", "").replace(")", "").replace("+", "plus")
    for key, value in MODEL_DISPLAY_NAMES.items()
}

MODEL_COLORS = {
    "Logistic Regression": "green",
    "GBT": "red",
    "SVM (Linear)": "purple",
    "MLP": "brown",
    "Random Forest": "crimson",
    "KNN": "gray",
    "ASA Rule": "orange",
    "AutoGluon": "blue",
    "LSTM": "darkorange",
    "Hybrid (MLP + LSTM)": "teal",
    "MLP-only": "olive",
}

DATASET_REGIMES = ("preop", "intraop", "combined")
MANUSCRIPT_SECTIONS = ("consort", "tables", "curves", "statistics", "reclassification", "shap")
SUPPORTED_SHAP_MODELS = ("xgb", "rf", "log_reg")
MANUSCRIPT_MODEL_ORDER = (
    "asa_rule",
    "autogluon",
    "xgb",
    "knn",
    "log_reg",
    "lstm_only",
    "lstm",
    "mlp",
    "hybrid",
    "rf",
    "svm",
)
LEGACY_DELONG_EXCLUSIONS = (
    "preop_ASA Rule",
    "combined_ASA Rule",
    "combined_Hybrid (MLP + LSTM)",
)
SEQUENCE_MODEL_DATASET = {
    "lstm_only": "intraop",
    "hybrid": "combined",
    "mlp_only": "preop",
}
SEQUENCE_MODEL_POPULATION = {
    "lstm_only": "sequence_common",
    "hybrid": "sequence_common",
    "mlp_only": "preop",
}

LEGACY_BASE_MODEL_NAME = {
    "sequence_common": "base_54k",
    "preop": "base",
    "intraop": "base",
    "combined": "base",
}


def model_display_name(model_key: str) -> str:
    return MODEL_DISPLAY_NAMES.get(model_key, model_key)


def model_safe_name(model_key: str) -> str:
    return MODEL_SAFE_NAMES.get(model_key, model_key.replace(" ", "_"))


model_colors = MODEL_COLORS
