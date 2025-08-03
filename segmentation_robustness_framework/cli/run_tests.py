#!/usr/bin/env python3
"""Run tests for the segmentation robustness framework.

This CLI tool provides a convenient way to run tests with various options:
- Run all tests
- Run specific test categories
- Run tests with different configurations
- Generate test reports

Usage:
    python -m segmentation_robustness_framework.cli.run_tests
    python -m segmentation_robustness_framework.cli.run_tests --loaders
    python -m segmentation_robustness_framework.cli.run_tests --adapters
    python -m segmentation_robustness_framework.cli.run_tests --verbose
    python -m segmentation_robustness_framework.cli.run_tests --coverage
"""

import argparse
import subprocess
import sys
from pathlib import Path


def run_pytest_command(args: list[str], verbose: bool = False) -> int:
    """Run pytest with given arguments.

    Args:
        args: List of pytest arguments
        verbose: Whether to show verbose output

    Returns:
        Exit code from pytest
    """
    cmd = ["python", "-m", "pytest"] + args

    if verbose:
        cmd.append("-v")

    print(f"Running: {' '.join(cmd)}")
    print("-" * 60)

    result = subprocess.run(cmd, capture_output=False)
    return result.returncode


def run_loader_tests(verbose: bool = False) -> int:
    """Run loader tests.

    Args:
        verbose: Whether to show verbose output

    Returns:
        Exit code
    """
    print("Running loader tests...")
    return run_pytest_command(["tests/loaders/"], verbose=verbose)


def run_adapter_tests(verbose: bool = False) -> int:
    """Run adapter tests.

    Args:
        verbose: Whether to show verbose output

    Returns:
        Exit code
    """
    print("Running adapter tests...")
    return run_pytest_command(["tests/adapters/"], verbose=verbose)


def run_attack_tests(verbose: bool = False) -> int:
    """Run attack tests.

    Args:
        verbose: Whether to show verbose output

    Returns:
        Exit code
    """
    print("Running attack tests...")
    return run_pytest_command(["tests/attacks/"], verbose=verbose)


def run_metric_tests(verbose: bool = False) -> int:
    """Run metric tests.

    Args:
        verbose: Whether to show verbose output

    Returns:
        Exit code
    """
    print("Running metric tests...")
    return run_pytest_command(["tests/utils/"], verbose=verbose)


def run_pipeline_tests(verbose: bool = False) -> int:
    """Run pipeline tests.

    Args:
        verbose: Whether to show verbose output

    Returns:
        Exit code
    """
    print("Running pipeline tests...")
    return run_pytest_command(["tests/pipeline/"], verbose=verbose)


def run_with_coverage(verbose: bool = False) -> int:
    """Run tests with coverage report.

    Args:
        verbose: Whether to show verbose output

    Returns:
        Exit code
    """
    print("Running tests with coverage...")
    args = [
        "--cov=segmentation_robustness_framework",
        "--cov-report=term-missing",
        "--cov-report=html:htmlcov",
        "tests/",
    ]
    return run_pytest_command(args, verbose=verbose)


def run_specific_test(test_path: str, verbose: bool = False) -> int:
    """Run a specific test file or test function.

    Args:
        test_path: Path to test file or specific test
        verbose: Whether to show verbose output

    Returns:
        Exit code
    """
    print(f"Running specific test: {test_path}")
    return run_pytest_command([test_path], verbose=verbose)


def main() -> None:
    """Main function."""
    parser = argparse.ArgumentParser(
        description="Run tests for the segmentation robustness framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m segmentation_robustness_framework.cli.run_tests
  python -m segmentation_robustness_framework.cli.run_tests --verbose
  python -m segmentation_robustness_framework.cli.run_tests --loaders --adapters
  python -m segmentation_robustness_framework.cli.run_tests --coverage
  python -m segmentation_robustness_framework.cli.run_tests tests/loaders/test_attack_loader.py
        """,
    )

    parser.add_argument("--loaders", action="store_true", help="Run loader tests")

    parser.add_argument("--adapters", action="store_true", help="Run adapter tests")

    parser.add_argument("--attacks", action="store_true", help="Run attack tests")

    parser.add_argument("--metrics", action="store_true", help="Run metric tests")

    parser.add_argument("--pipeline", action="store_true", help="Run pipeline tests")

    parser.add_argument("--coverage", action="store_true", help="Run tests with coverage report")

    parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")

    parser.add_argument("test_path", nargs="?", help="Specific test file or test to run")

    args = parser.parse_args()

    # Check if tests directory exists
    tests_dir = Path("tests")
    if not tests_dir.exists():
        print("Error: tests directory not found. Make sure you're in the project root.")
        sys.exit(1)

    exit_codes = []

    # If specific test path provided, run only that
    if args.test_path:
        exit_codes.append(run_specific_test(args.test_path, args.verbose))
    else:
        # Run specific test categories if requested
        if args.loaders:
            exit_codes.append(run_loader_tests(args.verbose))

        if args.adapters:
            exit_codes.append(run_adapter_tests(args.verbose))

        if args.attacks:
            exit_codes.append(run_attack_tests(args.verbose))

        if args.metrics:
            exit_codes.append(run_metric_tests(args.verbose))

        if args.pipeline:
            exit_codes.append(run_pipeline_tests(args.verbose))

        if args.coverage:
            exit_codes.append(run_with_coverage(args.verbose))

        # If no specific category requested, run all tests
        if not any([
            args.loaders,
            args.adapters,
            args.attacks,
            args.metrics,
            args.pipeline,
            args.coverage,
        ]):
            print("Running all tests...")
            exit_codes.append(run_pytest_command(["tests/"], args.verbose))

    # Determine overall exit code
    overall_exit_code = 0 if all(code == 0 for code in exit_codes) else 1

    if overall_exit_code == 0:
        print("\n" + "=" * 60)
        print("✅ All tests passed!")
        print("=" * 60)
    else:
        print("\n" + "=" * 60)
        print("❌ Some tests failed!")
        print("=" * 60)

    sys.exit(overall_exit_code)


if __name__ == "__main__":
    main()
