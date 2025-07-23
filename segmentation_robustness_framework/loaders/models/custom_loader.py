import logging
from typing import Any

import torch
import torch.nn as nn

from segmentation_robustness_framework.loaders.models.base import BaseModelLoader

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class CustomModelLoader(BaseModelLoader):
    """Loader for custom user models.

    Supported model_config keys:
        - `model_class` (str | Callable[..., Any]): Model class or factory function (required).
        - `model_args` (list[Any]): List of positional arguments for model initialization (optional).
        - `model_kwargs` (dict[str, Any]): Dict of keyword arguments for model initialization (optional).

    Example:
        ```python
        loader = CustomModelLoader()
        model_config = {
            "model_class": MyCustomSegmentationModel,
            "model_args": [3, 21],
            "model_kwargs": {},
        }
        model = loader.load_model(model_config)
        model = loader.load_weights(model, "weights.pth", weight_type="full")
        ```
    """

    def load_model(self, model_config: dict[str, Any]) -> nn.Module:
        """Load custom model.

        Args:
            model_config (dict[str, Any]):
                - `model_class` (str | Callable[..., Any]): Model class or factory function.
                - `model_args` (list[Any]): List of positional arguments for model initialization.
                - `model_kwargs` (dict[str, Any]): Dict of keyword arguments for model initialization.

        Raises:
            ValueError: If `model_class` is not callable.

        Returns:
            `nn.Module`: Instantiated model.
        """
        try:
            model_class = model_config["model_class"]
            model_args = model_config.get("model_args", [])
            model_kwargs = model_config.get("model_kwargs", {})

            if isinstance(model_class, str):
                from segmentation_robustness_framework.utils.loader_utils import resolve_model_class

                model_class = resolve_model_class(model_class)
                model = model_class(*model_args, **model_kwargs)
                logger.info(f"Loaded custom model: {model_class.__name__}")
            elif callable(model_class):
                model = model_class(*model_args, **model_kwargs)
                logger.info(f"Loaded custom model: {model_class.__name__}")
            else:
                raise ValueError(f"model_class must be a string or callable, got {type(model_class)}")
            return model
        except Exception as e:
            logger.exception(f"Failed to load custom model: {e}")
            raise

    def load_weights(self, model: nn.Module, weights_path: str, weight_type: str = "full") -> nn.Module:
        """Load weights into custom model.

        Args:
            model (nn.Module): Model instance.
            weights_path (str | Path): Path to weights file.
            weight_type (str): `'full'` for entire model, `'encoder'` for encoder only.

        Supported weight_type values:
            - `'full'`: Load entire model weights.
            - `'encoder'`: Load encoder weights only.

        Returns:
            `nn.Module`: Model with loaded weights.
        """
        try:
            checkpoint = torch.load(weights_path, map_location="cpu")

            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            if weight_type == "full":
                model.load_state_dict(state_dict, strict=False)
                logger.info(f"Loaded full model weights into custom model from {weights_path}")
            elif weight_type == "encoder":
                if hasattr(model, "encoder"):
                    encoder_state_dict = {
                        k.replace("encoder.", ""): v for k, v in state_dict.items() if k.startswith("encoder.")
                    }
                    model.encoder.load_state_dict(encoder_state_dict, strict=False)
                    logger.info(f"Loaded encoder (backbone) weights into custom model from {weights_path}")
                else:
                    logger.warning("Model has no 'encoder' attribute, loading full model weights")
                    model.load_state_dict(state_dict, strict=False)
            else:
                raise ValueError(f"Unknown weight_type: {weight_type}. No weights loaded.")
            return model
        except Exception as e:
            logger.exception(f"Failed to load weights into custom model: {e}")
            raise
