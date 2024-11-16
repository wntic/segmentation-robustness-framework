import json
import os
from pathlib import Path
from typing import Union

VALID_METRICS = [
    "mean_iou",
    "pixel_accuracy",
    "precision_macro",
    "precision_micro",
    "recall_macro",
    "recall_micro",
    "dice_macro",
    "dice_micro",
]


def validate_metrics(metrics: list[str]) -> None:
    """Validates metrics.

    Args:
        metrics (list[str]): Metrics for validation.

    Raises:
        ValueError: If the given metric is not correct.
    """
    for metric in metrics:
        if metric not in VALID_METRICS:
            raise ValueError(f"Got unexpected metric '{metric}'. Valid metrics: {VALID_METRICS}")


def initialize_metrics_storage(metrics: list[str]) -> dict[str, list[float]]:
    """Initializes storage for performance metrics.

    Args:
        metrics (list[str]): List of metrics to be computed.

    Returns:
        dict[str, list[float]]: A dictionary to store lists of performance metrics.
    """
    return {key: [] for key in metrics}


def save_metrics(output_dir: Union[str, Path], metrics_storage: dict[str, list[float]]) -> None:
    """Saves the calculated metrics to a JSON file.

    Args:
        output_dir (Union[str, Path]): Path to the output directory to the computed save metrics.
        metrics_storage (dcit[str, list[float]]): Storage containing metric values.
    """
    metrics_file = os.path.join(output_dir, "metrics.json")
    with open(metrics_file, "w") as f:
        json.dump(metrics_storage, f, indent=4)


def append_metrics(storage: dict[str, list[float]], metrics: dict[str, float]) -> None:
    """Appends calculated metrics to the storage dictionary.

    Args:
        storage (dict[str, list[float]]): The dictionary where metrics are stored.
        metrics (dict[str, float]): The metrics to append to the storage.
    """
    for key, value in metrics.items():
        storage[key].append(value)
