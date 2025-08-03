from importlib import import_module

from segmentation_robustness_framework.metrics import MetricsCollection, list_custom_metrics


def _ensure_metrics_imported() -> None:
    """Import metrics modules so that custom metrics are registered.

    The custom metrics registry is populated when metrics modules are imported.
    If a user imports this utility *before* any custom metric modules, the
    registry might be incomplete. Explicitly importing the metrics package
    guarantees all custom metrics are loaded.
    """

    import_module("segmentation_robustness_framework.metrics")


def main() -> None:
    """Entry-point that prints available metrics to *stdout*."""

    _ensure_metrics_imported()

    print("Available metrics:\n")

    print("Built-in metrics:")
    print("-" * 50)

    temp_collection = MetricsCollection(num_classes=2, ignore_index=255)
    builtin_metrics, builtin_names = temp_collection.get_all_metrics_with_averaging(include_pixel_accuracy=True)

    width = max(len(name) for name in builtin_names) + 2 if builtin_names else 20

    for name, metric_func in zip(builtin_names, builtin_metrics):
        summary = (metric_func.__doc__ or "").strip().split("\n")[0]
        if not summary:
            if "mean_iou" in name:
                summary = "Mean Intersection over Union"
            elif "precision" in name:
                summary = "Precision metric"
            elif "recall" in name:
                summary = "Recall metric"
            elif "dice_score" in name:
                summary = "Dice score (F1 score for segmentation)"
            elif "pixel_accuracy" in name:
                summary = "Pixel accuracy"
            else:
                summary = "Built-in metric"
        print(f"{name.ljust(width)}{summary}")

    print("\nCustom metrics:")
    print("-" * 50)

    custom_metrics = list_custom_metrics()
    if not custom_metrics:
        print("No custom metrics are currently registered.")
    else:
        width = max(len(name) for name in custom_metrics) + 2 if custom_metrics else 20
        for name in sorted(custom_metrics):
            try:
                from segmentation_robustness_framework.metrics import get_custom_metric

                metric_func = get_custom_metric(name)
                summary = (metric_func.__doc__ or "").strip().split("\n")[0]
                print(f"{name.ljust(width)}{summary}")
            except Exception:
                print(f"{name.ljust(width)}<no description available>")

    print("\nUsage in configuration:")
    print("-" * 50)
    print("Built-in metrics can be specified as strings:")
    print('  selected_metrics: ["mean_iou", "pixel_accuracy"]')
    print("\nBuilt-in metrics with averaging:")
    print('  selected_metrics: [{"name": "dice_score", "average": "micro"}]')
    print("\nCustom metrics:")
    print('  selected_metrics: ["custom_dice_score", "weighted_iou"]')

    print("\nBase metric names (without averaging):")
    print("-" * 50)
    base_metrics = ["mean_iou", "precision", "recall", "dice_score", "pixel_accuracy"]
    for metric in base_metrics:
        print(f"  {metric}")


if __name__ == "__main__":
    main()
