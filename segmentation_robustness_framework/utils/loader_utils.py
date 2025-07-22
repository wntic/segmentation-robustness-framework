import importlib
from typing import Any, Callable


def resolve_model_class(model_class: str) -> Callable[..., Any]:
    """Resolve a class or callable from a string path or return as-is.

    If the input string contains a dot, it is interpreted as a module path and class name,
    and the class is imported and returned. Otherwise, the input is returned as-is (assumed callable).

    Args:
        model_class (str): Dotted path to a class (e.g., 'module.submodule.ClassName') or a callable name.

    Returns:
        Callable[..., Any]: The resolved class or callable object.

    Example:
        ```python
        MyClass = resolve_model_class("my_package.my_module.MyClass")
        instance = MyClass()
        ```
    """
    if "." in model_class:
        module_name, class_name = model_class.rsplit(".", 1)
        module = importlib.import_module(module_name)
        return getattr(module, class_name)
    return model_class
