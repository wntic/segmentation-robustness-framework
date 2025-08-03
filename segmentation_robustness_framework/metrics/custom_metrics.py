import logging
from typing import Callable

logger = logging.getLogger(__name__)


CUSTOM_METRICS_REGISTRY: dict[str, Callable] = {}


def register_custom_metric(name: str) -> Callable:
    """Register a custom metric function.

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
