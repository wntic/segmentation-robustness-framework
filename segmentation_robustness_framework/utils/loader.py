import importlib
from typing import Any, Callable


def resolve_model_class(model_class: str) -> Callable[..., Any]:
    """Resolve a class from a string"""
    if "." in model_class:
        module_name, class_name = model_class.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    return model_class
