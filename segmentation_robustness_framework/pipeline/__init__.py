"""Pipeline module for segmentation robustness evaluation.

This module provides the core pipeline implementation and configuration
for evaluating segmentation models under adversarial attacks.
"""

from .config import PipelineConfig
from .core import SegmentationRobustnessPipeline

__all__ = [
    "SegmentationRobustnessPipeline",
    "PipelineConfig",
]
