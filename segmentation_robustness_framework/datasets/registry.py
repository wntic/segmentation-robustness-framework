from typing import Callable

from torch.utils.data import Dataset as TorchDataset

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
        if not issubclass(cls, TorchDataset):
            raise TypeError(
                f"Registered dataset '{name}' must inherit from torch.utils.data.Dataset, got {cls.__qualname__}."
            )
        if name in DATASET_REGISTRY and DATASET_REGISTRY[name] is not cls:
            raise ValueError(
                f"Dataset name '{name}' is already registered to {DATASET_REGISTRY[name].__qualname__}, cannot register {cls.__qualname__}."
            )
        DATASET_REGISTRY[name] = cls
        return cls

    return decorator
