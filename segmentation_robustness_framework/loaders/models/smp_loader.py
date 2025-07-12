import importlib
import logging
from typing import Any

import torch
import torch.nn as nn

from segmentation_robustness_framework.loaders.models.base import BaseModelLoader

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SMPModelLoader(BaseModelLoader):
    """
    Loader for segmentation_models_pytorch (SMP) models.

    Supports loading models and weights, including encoder-only weights.

    Example usage:
        loader = SMPModelLoader()
        model = loader.load_model({"architecture": "unet", "encoder_name": "resnet34", "classes": 2})
        model = loader.load_weights(model, "weights.pth", weight_type="full")
    """

    _smp = None  # class-level cache

    @classmethod
    def _import_smp(cls):
        if cls._smp is None:
            try:
                cls._smp = importlib.import_module("segmentation_models_pytorch")
            except ImportError as err:
                raise ImportError(
                    "segmentation_models_pytorch is not installed. Install it or choose another model origin."
                ) from err
        return cls._smp

    def load_model(self, model_config: dict[str, Any]) -> nn.Module:
        """
        Load an SMP model from config or checkpoint.

        Args:
            model_config: dict with keys like 'architecture', 'encoder_name', 'encoder_weights', 'classes', 'activation', or 'checkpoint'.
        Returns:
            nn.Module: Instantiated SMP model.
        Example:
            loader = SMPModelLoader()
            model = loader.load_model({"architecture": "unet", "encoder_name": "resnet34", "classes": 2})
        """
        smp = self._import_smp()
        checkpoint = model_config.get("checkpoint")

        if checkpoint and checkpoint.startswith("smp-hub/"):
            try:
                model = smp.from_pretrained(checkpoint)
            except Exception as e:
                raise RuntimeError(f"Could not load checkpoint {checkpoint}") from e
        else:
            architecture = model_config.get("architecture", "unet")
            encoder_name = model_config.get("encoder_name", "resnet34")
            encoder_weights = model_config.get("encoder_weights", "imagenet")
            classes = model_config.get("classes", 1)
            activation = model_config.get("activation", None)

            model = smp.create_model(
                arch=architecture,
                encoder_name=encoder_name,
                encoder_weights=encoder_weights,
                classes=classes,
                activation=activation,
            )

        if "classes" in model_config and hasattr(model, "classifier"):
            if model.classifier.out_channels != model_config["classes"]:
                in_ch = model.classifier.in_channels
                model.classifier = nn.Conv2d(in_ch, model_config["classes"], 1)

        return model

    def load_weights(self, model: nn.Module, weights_path: str, weight_type: str = "full") -> nn.Module:
        """Load weights into SMP model

        Args:
            model: Model instance
            weights_path: Path to weights file
            weight_type: 'full' for entire model, 'encoder' for encoder only
        Returns:
            nn.Module: Model with loaded weights
        Example:
            loader = SMPModelLoader()
            model = loader.load_model({"architecture": "unet", "encoder_name": "resnet34", "classes": 2})
            model = loader.load_weights(model, "weights.pth", weight_type="encoder")
        """
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)

        if "state_dict" in checkpoint:
            state_dict = checkpoint["state_dict"]
        else:
            state_dict = checkpoint

        if weight_type == "full":
            missing, unexpected = model.load_state_dict(state_dict, strict=False)
            if missing:
                logger.warning(f"Missing keys when loading weights: {missing}")
            if unexpected:
                logger.warning(f"Unexpected keys when loading weights: {unexpected}")
        elif weight_type == "encoder":
            encoder_state_dict = {
                k.replace("encoder.", ""): v for k, v in state_dict.items() if k.startswith("encoder.")
            }
            missing, unexpected = model.encoder.load_state_dict(encoder_state_dict, strict=False)
            if missing:
                logger.warning(f"Missing keys when loading encoder weights: {missing}")
            if unexpected:
                logger.warning(f"Unexpected keys when loading encoder weights: {unexpected}")
        return model
