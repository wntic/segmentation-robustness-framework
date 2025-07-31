"""List all datasets registered in the framework.

Run:
    python -m segmentation_robustness_framework.list_datasets

This utility prints the keys present in `segmentation_robustness_framework.datasets.registry.DATASET_REGISTRY` so
users can quickly see which dataset names can be used in their configuration files.
"""

from importlib import import_module

from segmentation_robustness_framework.datasets.registry import DATASET_REGISTRY


def _ensure_builtin_datasets_imported() -> None:
    """Import built-in dataset modules so that they register themselves.

    The registry is populated when a dataset module is imported (via the
    `@register` decorator).  If a user imports this utility *before* any of the
    built-in dataset modules, the registry might be empty.  Explicitly importing
    the `segmentation_robustness_framework.datasets` sub-package guarantees all
    built-ins are loaded.
    """

    import_module("segmentation_robustness_framework.datasets")


def main() -> None:
    """Entry-point that prints available datasets to *stdout*."""

    _ensure_builtin_datasets_imported()

    if not DATASET_REGISTRY:
        print("No datasets are currently registered.")
        return

    print("Available datasets (name → class):\n")
    width = max(len(name) for name in DATASET_REGISTRY) + 2
    for name, cls in sorted(DATASET_REGISTRY.items()):
        summary = (cls.__doc__ or "").strip().split("\n")[0]
        print(f"{name.ljust(width)}{summary}")


if __name__ == "__main__":
    main()
