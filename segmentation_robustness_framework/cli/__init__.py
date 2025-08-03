"""Command-line interface tools for the segmentation robustness framework.

This module provides CLI utilities for:
- Running pipeline configurations
- Listing available components
- Running tests
- Managing experiments

Available CLI commands:
- run_config: Execute pipeline from configuration files
- list_components: List all available components (models, attacks, metrics, datasets)
- run_tests: Run tests with various options
"""

# Import main functions for direct access
# Import individual list functions for backward compatibility
from .list_attacks import main as list_attacks_main
from .list_components import main as list_components_main
from .list_datasets import main as list_datasets_main
from .list_metrics import main as list_metrics_main
from .run_config import main as run_config_main
from .run_tests import main as run_tests_main

__all__ = [
    "run_config_main",
    "list_components_main",
    "run_tests_main",
    "list_attacks_main",
    "list_metrics_main",
    "list_datasets_main",
]
