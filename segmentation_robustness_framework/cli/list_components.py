#!/usr/bin/env python3
"""List all available components in the segmentation robustness framework.

This CLI tool provides a comprehensive view of all available:
- Models and adapters
- Datasets
- Attacks
- Metrics
- Configuration examples

Usage:
    python -m segmentation_robustness_framework.cli.list_components
    python -m segmentation_robustness_framework.cli.list_components --attacks
    python -m segmentation_robustness_framework.cli.list_components --metrics
    python -m segmentation_robustness_framework.cli.list_components --datasets
    python -m segmentation_robustness_framework.cli.list_components --models
    python -m segmentation_robustness_framework.cli.list_components --examples
"""

import argparse
from pathlib import Path

from segmentation_robustness_framework.attacks.registry import ATTACK_REGISTRY
from segmentation_robustness_framework.loaders.models.universal_loader import UniversalModelLoader
from segmentation_robustness_framework.metrics import list_custom_metrics


def list_models() -> None:
    """List available models and their configurations."""
    print("Available Models:")
    print("-" * 50)

    loader = UniversalModelLoader()

    # Torchvision models
    print("Torchvision Models:")
    print("  deeplabv3_resnet50")
    print("  deeplabv3_resnet101")
    print("  fcn_resnet50")
    print("  fcn_resnet101")
    print("  lraspp_mobilenet_v3_large")

    print("\nHuggingFace Models:")
    print("  nvidia/mit-b0")
    print("  nvidia/mit-b1")
    print("  nvidia/mit-b2")
    print("  microsoft/beit-base-patch16-224-pt22k-ft22k")
    print("  You can find more models at the link: https://huggingface.co/models/")

    print("\nSMP Models:")
    print("  Unet")
    print("  UnetPlusPlus")
    print("  MAnet")
    print("  Linknet")
    print("  FPN")
    print("  PSPNet")
    print("  PAN")
    print("  DeepLabV3")
    print("  DeepLabV3Plus")
    print("  UPerNet")
    print("  SegFormer")
    print("  DPT")
    print("  Additional information can be found at the link: https://smp.readthedocs.io/en/latest/")

    print("\nExample Configuration:")
    print("  model:")
    print("    type: torchvision")
    print("    config:")
    print("      name: deeplabv3_resnet50")
    print("      num_classes: 21")


def list_attacks() -> None:
    """List available attacks and their parameters."""
    print("Available Attacks:")
    print("-" * 50)

    for attack_name in sorted(ATTACK_REGISTRY.keys()):
        attack_class = ATTACK_REGISTRY[attack_name]
        print(f"  {attack_name}")

        # Try to get docstring for description
        if hasattr(attack_class, "__doc__") and attack_class.__doc__:
            doc_lines = attack_class.__doc__.strip().split("\n")
            if doc_lines:
                description = doc_lines[0]
                print(f"    {description}")

    print("\nExample Configuration:")
    print("  attacks:")
    print("    - name: fgsm")
    print("      eps: 0.02")
    print("    - name: pgd")
    print("      eps: 0.02")
    print("      alpha: 0.01")
    print("      iters: 10")


def list_metrics() -> None:
    """List available metrics."""
    print("Available Metrics:")
    print("-" * 50)

    print("Built-in Metrics:")
    base_metrics = ["mean_iou", "precision", "recall", "dice_score", "pixel_accuracy"]
    for metric in base_metrics:
        print(f"  {metric}")

    print("\nCustom Metrics:")
    custom_metrics = list_custom_metrics()
    if custom_metrics:
        for metric in sorted(custom_metrics):
            print(f"  {metric}")
    else:
        print("  No custom metrics registered")

    print("\nExample Configuration:")
    print("  metrics:")
    print("    ignore_index: 255")
    print("    selected_metrics:")
    print("      - mean_iou")
    print("      - pixel_accuracy")
    print("      - {name: dice_score, average: micro}")


def list_datasets() -> None:
    """List available datasets."""
    print("Available Datasets:")
    print("-" * 50)

    datasets = [
        ("voc", "Pascal VOC 2012", "21 classes"),
        ("ade20k", "ADE20K", "150 classes"),
        ("cityscapes", "Cityscapes", "19 classes"),
        ("stanford_background", "Stanford Background", "9 classes"),
    ]

    for name, description, classes in datasets:
        print(f"  {name}: {description} ({classes})")

    print("\nExample Configuration:")
    print("  dataset:")
    print("    name: voc")
    print("    split: val")
    print("    image_shape: [256, 256]")
    print("    download: true")


def list_examples() -> None:
    """List available configuration examples."""
    examples_dir = Path(__file__).parent.parent / "pipeline" / "examples"

    if not examples_dir.exists():
        print("No example configurations found.")
        return

    print("Available Configuration Examples:")
    print("-" * 50)

    for example_file in sorted(examples_dir.glob("*.yaml")):
        print(f"  {example_file.name}")

    print("\nUsage:")
    print(f"  python -m segmentation_robustness_framework.cli.run_config {examples_dir}/torchvision_ade20k.yaml")


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(
        description="List available components in the segmentation robustness framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m segmentation_robustness_framework.cli.list_components
  python -m segmentation_robustness_framework.cli.list_components --attacks
  python -m segmentation_robustness_framework.cli.list_components --metrics --datasets
        """,
    )

    parser.add_argument("--models", action="store_true", help="List available models")

    parser.add_argument("--attacks", action="store_true", help="List available attacks")

    parser.add_argument("--metrics", action="store_true", help="List available metrics")

    parser.add_argument("--datasets", action="store_true", help="List available datasets")

    parser.add_argument("--examples", action="store_true", help="List available configuration examples")

    args = parser.parse_args()

    # If no specific component requested, show all
    if not any([args.models, args.attacks, args.metrics, args.datasets, args.examples]):
        list_models()
        print("\n" + "=" * 60 + "\n")
        list_attacks()
        print("\n" + "=" * 60 + "\n")
        list_metrics()
        print("\n" + "=" * 60 + "\n")
        list_datasets()
        print("\n" + "=" * 60 + "\n")
        list_examples()
    else:
        if args.models:
            list_models()
            print()
        if args.attacks:
            list_attacks()
            print()
        if args.metrics:
            list_metrics()
            print()
        if args.datasets:
            list_datasets()
            print()
        if args.examples:
            list_examples()
            print()


if __name__ == "__main__":
    main()
