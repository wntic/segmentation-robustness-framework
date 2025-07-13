from typing import Callable

DATASET_REGISTRY: dict[str, Callable] = {}


def register_dataset(name: str) -> Callable:
    """Register a dataset class with a given name.

    Args:
        name: The name of the dataset.

    Returns:
        A decorator that registers the dataset class with the given name.

    Example:
        ```python
        @register_dataset("my_dataset")
        class MyDataset:
            def __init__(self, *args, **kwargs): ...

            def __getitem__(self, index: int) -> Any: ...

            def __len__(self) -> int: ...

            def __call__(self, *args, **kwargs) -> Any: ...
        ```
    """

    def decorator(cls: Callable) -> Callable:
        DATASET_REGISTRY[name] = cls
        return cls

    return decorator
