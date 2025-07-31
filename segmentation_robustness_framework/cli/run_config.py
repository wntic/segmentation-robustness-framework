#!/usr/bin/env python3
"""Command-line interface for running segmentation robustness pipelines from configuration files.

This script allows users to run pipeline configurations from YAML or JSON files
with a simple command-line interface.

Usage:
    python run_config.py config.yaml
    python run_config.py config.json --save --show
    python run_config.py config.yaml --batch-size 4 --device cpu
"""

import argparse
import logging
import sys
from pathlib import Path

from segmentation_robustness_framework.pipeline import PipelineConfig


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration.

    Args:
        verbose (bool): Whether to enable verbose logging.
    """
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def parse_args() -> argparse.Namespace:
    """Parse command line arguments.

    Returns:
        argparse.Namespace: Parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Run segmentation robustness pipeline from configuration file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_config.py config.yaml
  python run_config.py config.json --save --show
  python run_config.py config.yaml --batch-size 4 --device cpu
  python run_config.py config.yaml --override pipeline.batch_size=4 pipeline.device=cpu
        """,
    )

    parser.add_argument(
        "config_file",
        type=str,
        help="Path to configuration file (YAML or JSON). See pipeline/examples/ for sample configurations.",
    )

    parser.add_argument(
        "--save",
        action="store_true",
        default=True,
        help="Save results to output directory (default: True)",
    )

    parser.add_argument(
        "--show",
        action="store_true",
        help="Show visualizations (default: False)",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    parser.add_argument(
        "--override",
        "-o",
        nargs="*",
        help="Override configuration values (format: section.key=value)",
    )

    parser.add_argument(
        "--summary-only",
        action="store_true",
        help="Only print configuration summary without running pipeline",
    )

    return parser.parse_args()


def override_config(config: dict, overrides: list[str]) -> dict:
    """Override configuration values from command line.

    Args:
        config (dict): Original configuration.
        overrides (list[str]): List of override strings in format "section.key=value".

    Returns:
        dict: Modified configuration.

    Raises:
        ValueError: If override format is invalid.
    """
    for override in overrides:
        try:
            path, value = override.split("=", 1)
            keys = path.split(".")

            # Navigate to the target location
            current = config
            for key in keys[:-1]:
                if key not in current:
                    current[key] = {}
                current = current[key]

            # Set the value
            key = keys[-1]

            # Try to convert value to appropriate type
            if value.lower() in ("true", "false"):
                value = value.lower() == "true"
            elif value.isdigit():
                value = int(value)
            elif value.replace(".", "").isdigit():
                value = float(value)
            elif value.startswith("[") and value.endswith("]"):
                # Handle list values
                value = eval(value)  # Safe for simple lists

            current[key] = value

        except (ValueError, AttributeError) as e:
            raise ValueError(f"Invalid override format '{override}': {e}")

    return config


def main() -> None:
    """Main function."""
    args = parse_args()
    setup_logging(args.verbose)

    config_file = Path(args.config_file)
    if not config_file.exists():
        print(f"Error: Configuration file not found: {config_file}")
        sys.exit(1)

    try:
        # Load configuration
        if config_file.suffix.lower() in (".yaml", ".yml"):
            config = PipelineConfig.from_yaml(config_file)
        elif config_file.suffix.lower() == ".json":
            config = PipelineConfig.from_json(config_file)
        else:
            print(f"Error: Unsupported file format: {config_file.suffix}")
            print("Supported formats: .yaml, .yml, .json")
            sys.exit(1)

        # Apply overrides if specified
        if args.override:
            config.config = override_config(config.config, args.override)

        # Print configuration summary
        summary = config.get_config_summary()
        print("\n" + "=" * 60)
        print("CONFIGURATION SUMMARY")
        print("=" * 60)
        print(f"Model: {summary['model']['type']}")
        print(f"Dataset: {summary['dataset']['name']} ({summary['dataset']['split']})")
        print(f"Image shape: {summary['dataset']['image_shape']}")
        print(f"Attacks: {', '.join(summary['attacks'])}")
        print(f"Batch size: {summary['pipeline']['batch_size']}")
        print(f"Device: {summary['pipeline']['device']}")
        print(f"Output directory: {summary['pipeline']['output_dir']}")
        print("=" * 60 + "\n")

        if args.summary_only:
            return

        # Run pipeline
        print("Starting pipeline execution...")
        results = config.run_pipeline(save=args.save, show=args.show)

        print("\n" + "=" * 60)
        print("PIPELINE COMPLETED SUCCESSFULLY")
        print("=" * 60)
        print(f"Results saved to: {summary['pipeline']['output_dir']}")
        print("=" * 60)

    except Exception as e:
        print(f"Error: {e}")
        if args.verbose:
            import traceback

            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
