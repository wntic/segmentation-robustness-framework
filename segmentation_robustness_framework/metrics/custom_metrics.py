import logging
from typing import Callable, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)


CUSTOM_METRICS_REGISTRY: dict[str, Callable] = {}


def register_custom_metric(name: str) -> Callable:
    """Decorator to register a custom metric function.

    Args:
        name (str): Name to register the metric under.

    Returns:
        Callable: Decorator function.

    Example:
        @register_custom_metric("my_dice_score")
        def my_dice_score(targets, preds):
            # Custom implementation
            return score
    """

    def decorator(func: Callable) -> Callable:
        CUSTOM_METRICS_REGISTRY[name] = func
        logger.info(f"Registered custom metric: {name}")
        return func

    return decorator


def get_custom_metric(name: str) -> Callable:
    """Get a custom metric function by name.

    Args:
        name (str): Name of the registered metric.

    Returns:
        Callable: The metric function.

    Raises:
        KeyError: If the metric name is not registered.
    """
    if name not in CUSTOM_METRICS_REGISTRY:
        available_metrics = list(CUSTOM_METRICS_REGISTRY.keys())
        raise KeyError(f"Custom metric '{name}' not found. Available metrics: {available_metrics}")
    return CUSTOM_METRICS_REGISTRY[name]


def list_custom_metrics() -> list[str]:
    """List all registered custom metrics.

    Returns:
        list[str]: List of registered metric names.
    """
    return list(CUSTOM_METRICS_REGISTRY.keys())


# Example custom metrics
@register_custom_metric("custom_dice_score")
def custom_dice_score(targets: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]) -> float:
    """Custom Dice score implementation.

    Args:
        targets: Ground truth labels
        preds: Predicted labels

    Returns:
        float: Dice score
    """
    # Convert to numpy if needed
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()

    # Flatten arrays
    targets = targets.flatten()
    preds = preds.flatten()

    # Calculate Dice score
    intersection = np.sum(targets * preds)
    union = np.sum(targets) + np.sum(preds)

    if union == 0:
        return 1.0  # Perfect score if both are empty

    return 2.0 * intersection / union


@register_custom_metric("weighted_iou")
def weighted_iou(targets: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]) -> float:
    """Weighted IoU metric.

    This metric gives more weight to certain classes (e.g., foreground objects).

    Args:
        targets: Ground truth labels
        preds: Predicted labels

    Returns:
        float: Weighted IoU score
    """
    # Convert to numpy if needed
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()

    # Define class weights (example: give more weight to foreground classes)
    class_weights = {
        0: 0.1,  # Background
        1: 1.0,  # Foreground class 1
        2: 1.0,  # Foreground class 2
        # ... add more classes as needed
    }

    # Calculate IoU for each class
    unique_classes = np.unique(np.concatenate([targets, preds]))
    weighted_ious = []

    for cls in unique_classes:
        if cls == 255:  # Ignore index
            continue

        target_mask = targets == cls
        pred_mask = preds == cls

        intersection = np.logical_and(target_mask, pred_mask).sum()
        union = np.logical_or(target_mask, pred_mask).sum()

        if union > 0:
            iou = intersection / union
            weight = class_weights.get(cls, 1.0)
            weighted_ious.append(iou * weight)

    return np.mean(weighted_ious) if weighted_ious else 0.0


@register_custom_metric("f1_score")
def f1_score(targets: Union[torch.Tensor, np.ndarray], preds: Union[torch.Tensor, np.ndarray]) -> float:
    """F1 score metric.

    Args:
        targets: Ground truth labels
        preds: Predicted labels

    Returns:
        float: F1 score
    """
    # Convert to numpy if needed
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if isinstance(preds, torch.Tensor):
        preds = preds.cpu().numpy()

    # Flatten arrays
    targets = targets.flatten()
    preds = preds.flatten()

    # Calculate precision and recall
    tp = np.sum((targets == 1) & (preds == 1))
    fp = np.sum((targets == 0) & (preds == 1))
    fn = np.sum((targets == 1) & (preds == 0))

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    # Calculate F1 score
    if precision + recall > 0:
        return 2 * (precision * recall) / (precision + recall)
    else:
        return 0.0
