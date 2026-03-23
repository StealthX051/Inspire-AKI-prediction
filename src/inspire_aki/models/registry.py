from __future__ import annotations

from inspire_aki.registry import DATASET_REGIMES, SEQUENCE_MODEL_DATASET, SEQUENCE_MODEL_POPULATION

TABULAR_DATASETS = DATASET_REGIMES
SEQUENCE_DATASETS = tuple(SEQUENCE_MODEL_DATASET.keys())


def sequence_dataset_for_model(model_key: str) -> str:
    return SEQUENCE_MODEL_DATASET[model_key]


def sequence_population_for_model(model_key: str) -> str:
    return SEQUENCE_MODEL_POPULATION[model_key]
