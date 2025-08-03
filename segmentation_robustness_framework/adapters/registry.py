from typing import Callable

ADAPTER_REGISTRY: dict[str, type] = {}


def register_adapter(name: str) -> Callable:
    """Register a segmentation model adapter class.

    Args:
        name (str): Name to register the adapter under.

    Returns:
        Callable: Decorator that registers the adapter class.

    Raises:
        ValueError: If the name is already registered to a different class.
    """

    def decorator(cls: type) -> type:
        if name in ADAPTER_REGISTRY and ADAPTER_REGISTRY[name] is not cls:
            raise ValueError(
                f"Adapter name '{name}' is already registered to {ADAPTER_REGISTRY[name].__qualname__}, "
                f"cannot register {cls.__qualname__}."
            )
        ADAPTER_REGISTRY[name] = cls
        return cls

    return decorator


def get_adapter(name: str) -> type:
    """Get a registered adapter class by name.

    Args:
        name (str): Name of the registered adapter.

    Returns:
        Type: Adapter class.

    Raises:
        KeyError: If the adapter name is not registered.
    """
    if name not in ADAPTER_REGISTRY:
        raise KeyError(f"Adapter '{name}' is not registered. Available: {list(ADAPTER_REGISTRY.keys())}")
    return ADAPTER_REGISTRY[name]
