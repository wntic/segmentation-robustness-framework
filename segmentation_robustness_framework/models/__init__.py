# Base segmentation models
from .base_model import SegmentationModel

# Torchvision segmentation models
from .torchvision_models.fcn import TorchvisionFCN
from .torchvision_models.deeplabv3 import TorchvisionDeepLabV3

# Other models below ...

__all__ = ["SegmentationModel", "TorchvisionDeepLabV3", "TorchvisionFCN"]
