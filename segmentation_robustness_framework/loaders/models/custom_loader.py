import logging
from typing import Any

import torch
import torch.nn as nn

from segmentation_robustness_framework.loaders.models.base import BaseModelLoader

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class CustomModelLoader(BaseModelLoader):
    """
    Loader for custom user models.

    model_config keys:
        - model_class: Model class or factory function (required)
        - model_args: List of positional arguments for model initialization (optional)
        - model_kwargs: Dict of keyword arguments for model initialization (optional)

    Example usage:
        loader = CustomModelLoader()
        model = loader.load_model({
            "model_class": MyCustomSegmentationModel,
            "model_args": [3, 21],
            "model_kwargs": {}
        })
        model = loader.load_weights(model, "weights.pth", weight_type="full")
    """

    def load_model(self, model_config: dict[str, Any]) -> nn.Module:
        """
        Load custom model

        Args:
            model_config: dict containing:
                - model_class: Model class or factory function
                - model_args: Arguments for model initialization
                - model_kwargs: Keyword arguments for model initialization
        """
        model_class = model_config["model_class"]
        model_args = model_config.get("model_args", [])
        model_kwargs = model_config.get("model_kwargs", {})

        if isinstance(model_class, str):
            from segmentation_robustness_framework.utils.model_loader import resolve_model_class

            model_class = resolve_model_class(model_class)
            model = model_class(*model_args, **model_kwargs)
            logger.info(f"Loaded custom model: {model_class.__name__}")
        elif callable(model_class):
            model = model_class(*model_args, **model_kwargs)
            logger.info(f"Loaded custom model: {model_class.__name__}")
        else:
            raise ValueError("model_class must be callable")

        return model  # type: ignore

    def load_weights(self, model: nn.Module, weights_path: str, weight_type: str = "full") -> nn.Module:
        """Load weights into custom model"""
        checkpoint = torch.load(weights_path, map_location="cpu")

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        if weight_type == "full":
            model.load_state_dict(state_dict, strict=False)
        elif weight_type == "encoder":
            # Attempt to load encoder weights (assumes encoder attribute exists)
            if hasattr(model, "encoder"):
                encoder_state_dict = {
                    k.replace("encoder.", ""): v for k, v in state_dict.items() if k.startswith("encoder.")
                }
                model.encoder.load_state_dict(encoder_state_dict, strict=False)
            else:
                logger.warning("Model has no 'encoder' attribute, loading full weights")
                model.load_state_dict(state_dict, strict=False)

        return model
