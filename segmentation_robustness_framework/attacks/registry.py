from typing import Callable

ATTACK_REGISTRY: dict[str, Callable] = {}


def register_attack(name: str) -> Callable:
    """Register an attack class.

    Args:
        name (str): The name of the attack.

    Returns:
        Callable: Decorator function that registers the attack class.

    Raises:
        ValueError: If the name is already registered to a different class.
    """

    def decorator(cls: Callable) -> Callable:
        if name in ATTACK_REGISTRY and ATTACK_REGISTRY[name] is not cls:
            raise ValueError(
                f"Attack name '{name}' is already registered to {ATTACK_REGISTRY[name].__qualname__}, "
                f"cannot register {cls.__qualname__}."
            )
        ATTACK_REGISTRY[name] = cls
        return cls

    return decorator
