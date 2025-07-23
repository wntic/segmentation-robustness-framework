import logging
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torchvision.models.segmentation as tv_segmentation

from segmentation_robustness_framework.loaders.models.base import BaseModelLoader

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class TorchvisionModelLoader(BaseModelLoader):
    """Loader for torchvision segmentation models.

    Supports loading models and weights, including encoder-only weights.
    Uses the 'weights' argument as recommended by torchvision >=0.13.

    Supported model_config keys:
        - `name` (str): Model name.
        - `num_classes` (int): Number of classes (optional).
        - `weights` (str | TorchvisionWeightsEnum): Torchvision weights enum, string, or None (optional).

    Example:
        ```python
        loader = TorchvisionModelLoader()
        model_config = {"name": "deeplabv3_resnet50", "num_classes": 21}
        model = loader.load_model(model_config)
        model = loader.load_weights(model, "weights.pth", weight_type="encoder")
        ```
    """

    SUPPORTED_MODELS = {
        "deeplabv3_resnet50": tv_segmentation.deeplabv3_resnet50,
        "deeplabv3_resnet101": tv_segmentation.deeplabv3_resnet101,
        "deeplabv3_mobilenetv3_large": tv_segmentation.deeplabv3_mobilenet_v3_large,
        "fcn_resnet50": tv_segmentation.fcn_resnet50,
        "fcn_resnet101": tv_segmentation.fcn_resnet101,
        "lraspp_mobilenet_v3_large": tv_segmentation.lraspp_mobilenet_v3_large,
    }

    TORCHVISION_WEIGHTS_ENUMS = {
        "deeplabv3_resnet50": tv_segmentation.DeepLabV3_ResNet50_Weights,
        "deeplabv3_resnet101": tv_segmentation.DeepLabV3_ResNet101_Weights,
        "deeplabv3_mobilenetv3_large": tv_segmentation.DeepLabV3_MobileNet_V3_Large_Weights,
        "fcn_resnet50": tv_segmentation.FCN_ResNet50_Weights,
        "fcn_resnet101": tv_segmentation.FCN_ResNet101_Weights,
        "lraspp_mobilenet_v3_large": tv_segmentation.LRASPP_MobileNet_V3_Large_Weights,
    }

    def load_model(self, model_config: dict[str, Any]) -> nn.Module:
        """Load a torchvision segmentation model using the 'weights' argument.

        Args:
            model_config (dict):
                - `name` (str): Model name.
                - `num_classes` (int): Number of classes (optional).
                - `weights` (str | TorchvisionWeightsEnum): Torchvision weights enum, string, or None (optional).

        Raises:
            ValueError: If the model name is not supported.

        Returns:
            `nn.Module`: Instantiated torchvision model.
        """
        try:
            name = model_config.get("name")
            num_classes = model_config.get("num_classes", 21)
            weights = model_config.get("weights", "__not_provided__")  # Sentinel value

            if name not in self.SUPPORTED_MODELS:
                logger.error(f"Unsupported model: {name}")
                raise ValueError(f"Unsupported model: {name}")

            model_fn = self.SUPPORTED_MODELS[name]

            weights_enum_cls = self.TORCHVISION_WEIGHTS_ENUMS.get(name)

            if weights == "__not_provided__":
                if weights_enum_cls is not None and hasattr(weights_enum_cls, "DEFAULT"):
                    weights = weights_enum_cls.DEFAULT
                    logger.info(f"No weights specified, using default weights for {name}.")
            elif isinstance(weights, str):
                if weights_enum_cls is not None and hasattr(weights_enum_cls, weights):
                    weights = getattr(weights_enum_cls, weights)
                elif (
                    weights.lower() == "default"
                    and weights_enum_cls is not None
                    and hasattr(weights_enum_cls, "DEFAULT")
                ):
                    weights = weights_enum_cls.DEFAULT
                else:
                    logger.error(f"Invalid weights: {weights}")
                    raise ValueError(f"Invalid weights: {weights}")

            model = model_fn(weights=weights, num_classes=num_classes)
            logger.info(f"Loaded torchvision model: {name} with weights={weights}")

            default_num_classes = 21
            if name.startswith("lraspp"):
                default_num_classes = 21
            if num_classes != default_num_classes:
                if hasattr(model, "classifier"):
                    cls = model.classifier
                    if isinstance(cls, nn.Sequential):
                        cls[-1] = nn.Conv2d(cls[-1].in_channels, num_classes, kernel_size=1)
                    elif hasattr(cls, "low_classifier") and hasattr(cls, "high_classifier"):
                        cls.low_classifier = nn.Conv2d(cls.low_classifier.in_channels, num_classes, kernel_size=1)
                        cls.high_classifier = nn.Conv2d(cls.high_classifier.in_channels, num_classes, kernel_size=1)
                    else:
                        logger.warning(
                            "Unknown classifier type for model %s; num_classes may not be updated correctly.",
                            name,
                        )
                elif hasattr(model, "aux_classifier"):
                    model.aux_classifier[-1] = nn.Conv2d(
                        model.aux_classifier[-1].in_channels, num_classes, kernel_size=1
                    )
                logger.info(f"Modified classifier for {name} to output {num_classes} classes.")

            return model

        except Exception as e:
            logger.exception(f"Failed to load torchvision model: {e}")
            raise

    def load_weights(self, model: nn.Module, weights_path: str | Path, weight_type: str = "full") -> nn.Module:
        """Load weights into a torchvision model.

        Args:
            model (nn.Module): Model instance.
            weights_path (str | Path): Path to weights file.
            weight_type (str): `'full'` for entire model, `'encoder'` for backbone only.

        Supported weight_type values:
            - `'full'`: Load entire model weights.
            - `'encoder'`: Load encoder weights only.

        Returns:
            `nn.Module`: Model with loaded weights.
        """
        try:
            logger.info(f"Loading weights from {weights_path} (type: {weight_type})")
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)

            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            elif "model" in checkpoint:
                state_dict = checkpoint["model"]
            else:
                state_dict = checkpoint

            if weight_type == "full":
                missing, unexpected = model.load_state_dict(state_dict, strict=False)
                if missing:
                    logger.warning(f"Missing keys when loading weights: {missing}")
                if unexpected:
                    logger.warning(f"Unexpected keys when loading weights: {unexpected}")
                logger.info(f"Loaded full model weights into torchvision model from {weights_path}")
            elif weight_type == "encoder":
                backbone_state_dict = {
                    k.replace("backbone.", ""): v for k, v in state_dict.items() if k.startswith("backbone.")
                }
                missing, unexpected = model.backbone.load_state_dict(backbone_state_dict, strict=False)
                if missing:
                    logger.warning(f"Missing keys when loading encoder weights: {missing}")
                if unexpected:
                    logger.warning(f"Unexpected keys when loading encoder weights: {unexpected}")
                logger.info(f"Loaded encoder (backbone) weights into torchvision model from {weights_path}")
            else:
                logger.warning(f"Unknown weight_type: {weight_type}. No weights loaded.")
            return model
        except Exception as e:
            logger.exception(f"Failed to load weights into torchvision model: {e}")
            raise
