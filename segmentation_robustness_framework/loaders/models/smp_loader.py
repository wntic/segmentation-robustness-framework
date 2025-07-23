import importlib
import logging
from typing import Any

import torch
import torch.nn as nn

from segmentation_robustness_framework.loaders.models.base import BaseModelLoader

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class SMPModelLoader(BaseModelLoader):
    """Loader for segmentation_models_pytorch (SMP) models.

    Supports loading models and weights, including encoder-only weights.

    Supported model_config keys:
        - `architecture` (str): Architecture of the model (default: `'unet'`).
        - `encoder_name` (str): Name of the encoder (default: `'resnet34'`).
        - `encoder_weights` (str): Weights of the encoder (default: `'imagenet'`).
        - `classes` (int): Number of classes (default: `1`).
        - `activation` (str): Activation function (default: `None`).
        - `checkpoint` (str | Path): Path to the checkpoint (default: `None`).

    Example:
        ```python
        # Basic usage
        loader = SMPModelLoader()
        model_config = {
            "architecture": "unet",
            "encoder_name": "resnet34",
            "classes": 2,
        }
        model = loader.load_model(model_config)
        model = loader.load_weights(
            model, "weights.pth", weight_type="full"
        )  # load full model weights if needed
        ```

    Example:
        ```python
        # With checkpoint
        loader = SMPModelLoader()
        model_config = {"checkpoint": "smp-hub/upernet-convnext-tiny"}
        model = loader.load_model(model_config)
        ```

    Example:
        ```python
        # With encoder-only weights
        loader = SMPModelLoader()
        model_config = {
            "architecture": "unet",
            "encoder_name": "resnet34",
            "classes": 2,
        }
        model = loader.load_model(model_config)
        model = loader.load_weights(model, "weights.pth", weight_type="encoder")
        ```
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
        """Load an SMP model from config or checkpoint.

        Args:
            model_config (dict[str, Any]):
                - `architecture` (str): Architecture of the model.
                - `encoder_name` (str): Name of the encoder (optional).
                - `encoder_weights` (str): Weights of the encoder (optional).
                - `classes` (int): Number of classes (optional).
                - `activation` (str): Activation function (optional).
                - `checkpoint` (str): Path to the checkpoint (optional).

        Raises:
            RuntimeError: If the checkpoint cannot be loaded.

        Returns:
            `nn.Module`: Instantiated SMP model.
        """
        smp = self._import_smp()
        checkpoint = model_config.get("checkpoint")
        try:
            if checkpoint and checkpoint.startswith("smp-hub/"):
                try:
                    model = smp.from_pretrained(checkpoint)
                    logger.info(f"Loaded SMP model from checkpoint: {checkpoint}")
                except Exception as e:
                    logger.exception(f"Could not load checkpoint {checkpoint}: {e}")
                    raise RuntimeError(f"Could not load checkpoint {checkpoint}") from e
            else:
                architecture = model_config.get("architecture", "unet")
                encoder_name = model_config.get("encoder_name", "resnet34")
                encoder_weights = model_config.get("encoder_weights", "imagenet")
                classes = model_config.get("classes", 3)
                activation = model_config.get("activation", None)

                model = smp.create_model(
                    arch=architecture,
                    encoder_name=encoder_name,
                    encoder_weights=encoder_weights,
                    classes=classes,
                    activation=activation,
                )
                logger.info(f"Loaded SMP model: {architecture} with encoder {encoder_name}")

            if "classes" in model_config and hasattr(model, "classifier"):
                if model.classifier.out_channels != model_config["classes"]:
                    in_ch = model.classifier.in_channels
                    model.classifier = nn.Conv2d(in_ch, model_config["classes"], 1)
                    logger.info(f"Adjusted classifier out_channels to {model_config['classes']}")
            return model
        except Exception as e:
            logger.exception(f"Failed to load SMP model: {e}")
            raise

    def load_weights(self, model: nn.Module, weights_path: str, weight_type: str = "full") -> nn.Module:
        """Load weights into SMP model.

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
            checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)

            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            if weight_type == "full":
                result = model.load_state_dict(state_dict, strict=False)
                if hasattr(result, "missing_keys") and hasattr(result, "unexpected_keys"):
                    missing = result.missing_keys
                    unexpected = result.unexpected_keys

                    if missing:
                        logger.warning(f"Missing keys when loading full model weights: {missing}")
                    if unexpected:
                        logger.warning(f"Unexpected keys when loading full model weights: {unexpected}")
                logger.info(f"Loaded full model weights into SMP model from {weights_path}")
            elif weight_type == "encoder":
                encoder_state_dict = {
                    k.replace("encoder.", ""): v for k, v in state_dict.items() if k.startswith("encoder.")
                }
                result = model.encoder.load_state_dict(encoder_state_dict, strict=False)

                if hasattr(result, "missing_keys") and hasattr(result, "unexpected_keys"):
                    missing = result.missing_keys
                    unexpected = result.unexpected_keys

                    if missing:
                        logger.warning(f"Missing keys when loading encoder weights: {missing}")
                    if unexpected:
                        logger.warning(f"Unexpected keys when loading encoder weights: {unexpected}")
                    logger.info(f"Loaded encoder (backbone) weights into SMP model from {weights_path}")
                else:
                    logger.info(
                        f"Loaded encoder weights (no missing/unexpected keys info available) into SMP model from {weights_path}"
                    )
            else:
                logger.warning(f"Unknown weight_type: {weight_type}. No weights loaded.")
            return model
        except Exception as e:
            logger.exception(f"Failed to load weights into SMP model: {e}")
            raise
