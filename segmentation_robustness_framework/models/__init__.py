# Base segmentation models
from .base_model import SegmentationModel

# Segmentation models
from .deeplab import DeepLabV3
from .fcn import FCN

__all__ = ["SegmentationModel", "DeepLabV3", "FCN"]
