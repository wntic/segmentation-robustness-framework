from .base_protocol import SegmentationModelProtocol
from .custom_adapter import CustomAdapter
from .huggingface_adapter import HuggingFaceAdapter
from .registry import ADAPTER_REGISTRY, get_adapter, register_adapter
from .smp_adapter import SMPAdapter
from .torchvision_adapter import TorchvisionAdapter

__all__ = [
    "ADAPTER_REGISTRY",
    "CustomAdapter",
    "get_adapter",
    "HuggingFaceAdapter",
    "register_adapter",
    "SegmentationModelProtocol",
    "SMPAdapter",
    "TorchvisionAdapter",
]
