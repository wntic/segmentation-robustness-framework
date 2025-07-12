from importlib.util import find_spec
from typing import Any, Optional

import torch.nn as nn

from segmentation_robustness_framework.loaders.models.custom_loader import CustomModelLoader
from segmentation_robustness_framework.loaders.models.hf_bundle import HFSegmentationBundle
from segmentation_robustness_framework.loaders.models.huggingface_loader import HuggingFaceModelLoader
from segmentation_robustness_framework.loaders.models.smp_loader import SMPModelLoader
from segmentation_robustness_framework.loaders.models.torchvision_loader import TorchvisionModelLoader


def _is_module_installed(module_name: str) -> bool:
    """
    Check if a Python module is installed.

    Args:
        module_name (str): Name of the module to check.

    Returns:
        bool: True if the module is installed, False otherwise.
    """
    spec = find_spec(module_name)
    return spec is not None


SMP_INSTALLED = _is_module_installed("segmentation_models_pytorch")
TORCHVISION_INSTALLED = _is_module_installed("torchvision")
HUGGINGFACE_INSTALLED = _is_module_installed("transformers")


class UniversalModelLoader:
    """Universal model loader that handles different model types.

    Supported model types:
        - `torchvision`
        - `smp`
        - `huggingface`
        - `custom`

    Example:
        ```python
        loader = UniversalModelLoader()
        model = loader.load_model(
            model_type="torchvision",
            model_config={"name": "deeplabv3_resnet50", "num_classes": 21},
        )
        ```

    Example:
        ```python
        loader = UniversalModelLoader()
        model = loader.load_model(
            model_type="smp",
            model_config={
                "architecture": "unet",
                "encoder_name": "resnet34",
                "classes": 2,
            },
        )
        ```

    Example:
        ```python
        loader = UniversalModelLoader()
        model = loader.load_model(
            model_type="huggingface",
            model_config={"model_name": "facebook/maskformer-swin-tiny-coco"},
        )
        ```

    Example:
        ```python
        loader = UniversalModelLoader()
        model = loader.load_model(
            model_type="custom",
            model_config={
                "model_class": "SomeModel",
                "model_args": [],
                "model_kwargs": {},
            },
        )
        ```
    """

    def __init__(self):
        self.loaders = {
            "torchvision": TorchvisionModelLoader() if TORCHVISION_INSTALLED else None,
            "smp": SMPModelLoader() if SMP_INSTALLED else None,
            "huggingface": HuggingFaceModelLoader() if HUGGINGFACE_INSTALLED else None,
            "custom": CustomModelLoader(),
        }

    def load_model(
        self,
        model_type: str,
        model_config: dict[str, Any],
        weights_path: Optional[str] = None,
        weight_type: str = "full",
    ) -> nn.Module:
        """Load model using appropriate loader.

        Args:
            model_type (str): Type of model ('torchvision', 'smp', 'huggingface', 'custom')
            model_config (dict[str, Any]): Configuration for model loading
            weights_path (Optional[str]): Path to weights file (optional)
            weight_type (str): Type of weights to load ('full' or 'encoder').

        Returns:
            `nn.Module`: Loaded model.
        """
        if model_type not in self.loaders:
            raise ValueError(f"Unsupported model type: {model_type}")

        loader = self.loaders[model_type]
        if loader is None:
            raise ImportError(f"Required dependencies for {model_type} not available")

        model = loader.load_model(model_config)  # may return bundle

        if isinstance(model, HFSegmentationBundle):
            bundle = model
            model = bundle.model

        if weights_path is not None:
            model = loader.load_weights(model, weights_path, weight_type)

        return model
