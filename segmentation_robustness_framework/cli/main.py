#!/usr/bin/env python3
"""Main CLI entry point for the segmentation robustness framework.

This provides a unified command-line interface for all framework operations.

Usage:
    python -m segmentation_robustness_framework.cli.main run config.yaml
    python -m segmentation_robustness_framework.cli.main list
    python -m segmentation_robustness_framework.cli.main test
    python -m segmentation_robustness_framework.cli.main list --attacks
    python -m segmentation_robustness_framework.cli.main test --loaders --verbose
"""

import argparse
import sys

from .list_components import main as list_components_main
from .run_config import main as run_config_main
from .run_tests import main as run_tests_main


def main() -> None:
    """Provide the main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Segmentation Robustness Framework CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m segmentation_robustness_framework.cli.main run config.yaml
  python -m segmentation_robustness_framework.cli.main list --attacks
  python -m segmentation_robustness_framework.cli.main test --loaders --verbose
        """,
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run pipeline from configuration file")
    run_parser.add_argument("config_file", help="Path to configuration file")
    run_parser.add_argument("--save", action="store_true", default=True, help="Save results")
    run_parser.add_argument("--show", action="store_true", help="Show visualizations")
    run_parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose logging")
    run_parser.add_argument("--override", "-o", nargs="*", help="Override configuration values")
    run_parser.add_argument("--summary-only", action="store_true", help="Only print configuration summary")

    # List command
    list_parser = subparsers.add_parser("list", help="List available components")
    list_parser.add_argument("--models", action="store_true", help="List available models")
    list_parser.add_argument("--attacks", action="store_true", help="List available attacks")
    list_parser.add_argument("--metrics", action="store_true", help="List available metrics")
    list_parser.add_argument("--datasets", action="store_true", help="List available datasets")
    list_parser.add_argument("--examples", action="store_true", help="List available configuration examples")

    # Test command
    test_parser = subparsers.add_parser("test", help="Run tests")
    test_parser.add_argument("--unit", action="store_true", help="Run unit tests")
    test_parser.add_argument("--integration", action="store_true", help="Run integration tests")
    test_parser.add_argument("--loaders", action="store_true", help="Run loader tests")
    test_parser.add_argument("--adapters", action="store_true", help="Run adapter tests")
    test_parser.add_argument("--attacks", action="store_true", help="Run attack tests")
    test_parser.add_argument("--metrics", action="store_true", help="Run metric tests")
    test_parser.add_argument("--engine", action="store_true", help="Run engine tests")
    test_parser.add_argument("--cli", action="store_true", help="Run CLI tests")
    test_parser.add_argument("--coverage", action="store_true", help="Run tests with coverage report")
    test_parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    test_parser.add_argument("test_path", nargs="?", help="Specific test file or test to run")

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Set up sys.argv for the subcommands
    original_argv = sys.argv.copy()

    if args.command == "run":
        # Build argv for run_config
        run_args = ["run_config.py", args.config_file]
        if args.save:
            run_args.append("--save")
        if args.show:
            run_args.append("--show")
        if args.verbose:
            run_args.append("--verbose")
        if args.override:
            run_args.extend(["--override"] + args.override)
        if args.summary_only:
            run_args.append("--summary-only")

        sys.argv = run_args
        run_config_main()

    elif args.command == "list":
        # Build argv for list_components
        list_args = ["list_components.py"]
        if args.models:
            list_args.append("--models")
        if args.attacks:
            list_args.append("--attacks")
        if args.metrics:
            list_args.append("--metrics")
        if args.datasets:
            list_args.append("--datasets")
        if args.examples:
            list_args.append("--examples")

        sys.argv = list_args
        list_components_main()

    elif args.command == "test":
        # Build argv for run_tests
        test_args = ["run_tests.py"]
        if args.unit:
            test_args.append("--unit")
        if args.integration:
            test_args.append("--integration")
        if args.loaders:
            test_args.append("--loaders")
        if args.adapters:
            test_args.append("--adapters")
        if args.attacks:
            test_args.append("--attacks")
        if args.metrics:
            test_args.append("--metrics")
        if args.engine:
            test_args.append("--engine")
        if args.cli:
            test_args.append("--cli")
        if args.coverage:
            test_args.append("--coverage")
        if args.verbose:
            test_args.append("--verbose")
        if args.test_path:
            test_args.append(args.test_path)

        sys.argv = test_args
        run_tests_main()

    # Restore original argv
    sys.argv = original_argv


if __name__ == "__main__":
    main()
