"""List all attacks registered in the framework.

Run:
    python -m segmentation_robustness_framework.list_attacks

This utility prints the keys present in `segmentation_robustness_framework.attacks.registry.ATTACK_REGISTRY` so
users can quickly see which attack names can be used in their configuration files.
"""

from importlib import import_module

from segmentation_robustness_framework.attacks.registry import ATTACK_REGISTRY


def _ensure_builtin_attacks_imported() -> None:
    """Import built-in attack modules so that they register themselves.

    The registry is populated when an attack module is imported (via the
    `@register` decorator).  If a user imports this utility *before* any of the
    built-in attack modules, the registry might be empty.  Explicitly importing
    the `segmentation_robustness_framework.attacks` sub-package guarantees all
    built-ins are loaded.
    """

    import_module("segmentation_robustness_framework.attacks")


def main() -> None:
    """Entry-point that prints available attacks to *stdout*."""

    _ensure_builtin_attacks_imported()

    if not ATTACK_REGISTRY:
        print("No attacks are currently registered.")
        return

    print("Available attacks (name → class):\n")
    width = max(len(name) for name in ATTACK_REGISTRY) + 2
    for name, cls in sorted(ATTACK_REGISTRY.items()):
        summary = (cls.__doc__ or "").strip().split("\n")[0]
        print(f"{name.ljust(width)}{summary}")


if __name__ == "__main__":
    main()
