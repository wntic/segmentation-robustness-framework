import logging
from importlib.util import find_spec
from typing import Any, Optional

import torch.nn as nn

from segmentation_robustness_framework.adapters.base_protocol import SegmentationModelProtocol
from segmentation_robustness_framework.adapters.registry import get_adapter
from segmentation_robustness_framework.loaders.models.custom_loader import CustomModelLoader
from segmentation_robustness_framework.loaders.models.huggingface_loader import HuggingFaceModelLoader
from segmentation_robustness_framework.loaders.models.smp_loader import SMPModelLoader
from segmentation_robustness_framework.loaders.models.torchvision_loader import TorchvisionModelLoader


def _is_module_installed(module_name: str) -> bool:
    """Check if a Python module is installed.

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

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class UniversalModelLoader:
    """Universal model loader that handles different model types and wraps them with adapters.

    Supported model types:
        - 'torchvision'
        - 'smp'
        - 'huggingface'
        - 'custom'
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
        adapter_cls: Optional[type] = None,
    ) -> nn.Module:
        """Load model using appropriate loader and wrap with the correct adapter.

        Args:
            model_type (str): Model type identifier. Supported values:
                - `'torchvision'`: Torchvision segmentation models.
                - `'smp'`: segmentation-models-pytorch models.
                - `'huggingface'`: HuggingFace Transformers models.
                - Any string starting with `'custom_'`: Alias for custom user-defined models.
            model_config (dict[str, Any]): Configuration for model loading.
            weights_path (Optional[str]): Path to weights file (optional).
            weight_type (str): Type of weights to load ('full' or 'encoder').
            adapter_cls (Optional[type]): Adapter class to wrap the model. If provided, this
                adapter will be used instead of the default adapter for the model type.

        Returns:
            nn.Module: Loaded and adapted model.
        """
        if model_type in self.loaders:
            loader = self.loaders[model_type]
            if loader is None:
                logger.error(f"Required dependencies for {model_type} not available")
                raise ImportError(f"Required dependencies for {model_type} not available")
        elif model_type.startswith("custom_"):
            loader = self.loaders["custom"]
        else:
            logger.error(f"Unsupported model type: {model_type}")
            raise ValueError(f"Unsupported model type: {model_type}")

        try:
            model = loader.load_model(model_config)  # may return bundle
        except Exception as e:
            logger.exception(f"Failed to load model for type {model_type}: {e}")
            raise

        if hasattr(model, "model"):
            bundle = model
            model = bundle.model

        if weights_path is not None:
            try:
                model = loader.load_weights(model, weights_path, weight_type)
                logger.info(f"Loaded weights for {model_type} model from {weights_path} (type: {weight_type})")
            except Exception as e:
                logger.exception(f"Failed to load weights for {model_type} model: {e}")
                raise

        # Wrap with adapter if not already adapted
        if not isinstance(model, SegmentationModelProtocol):
            AdapterCls = adapter_cls or get_adapter(model_type)
            model = AdapterCls(model)
            logger.info(f"Wrapped model with {AdapterCls.__name__} adapter for type '{model_type}'")
        else:
            logger.info("Model already implements SegmentationModelProtocol; no adapter wrapping needed.")

        return model
