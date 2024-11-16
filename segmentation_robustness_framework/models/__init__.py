# Base segmentation models
from .base_model import SegmentationModel
from .torchvision_models.deeplabv3 import TorchvisionDeepLabV3

# Torchvision segmentation models
from .torchvision_models.fcn import TorchvisionFCN

# Other models below ...

__all__ = ["SegmentationModel", "TorchvisionDeepLabV3", "TorchvisionFCN"]
