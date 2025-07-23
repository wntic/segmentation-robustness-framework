import importlib
import logging
from typing import Any

import torch
import torch.nn as nn

from segmentation_robustness_framework.loaders.models.base import BaseModelLoader
from segmentation_robustness_framework.loaders.models.hf_bundle import HFSegmentationBundle

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class HuggingFaceModelLoader(BaseModelLoader):
    """Loader for HuggingFace models.

    Supported model_config keys:
        - `model_name` (str): HuggingFace model id or path (required).
        - `num_labels` (int): Number of output classes (optional).
        - `trust_remote_code` (bool): Allow loading custom code from model repo (optional).
        - `task` (str): "semantic_segmentation", "instance_segmentation", "panoptic_segmentation", "image_segmentation" (optional).
        - `return_processor` (bool): Whether to return processor along with model (default: True).
        - `config_overrides` (dict): Arbitrary config attributes to override (optional).
        - `processor_overrides` (dict): Arbitrary processor attributes to override (optional).

    Example:
        ```python
        # Basic usage
        loader = HuggingFaceModelLoader()
        model_config = {
            "model_name": "nvidia/segformer-b2-finetuned-ade-512-512",
        }
        bundle = loader.load_model(model_config)

        # Access model and processor (optional)
        model = bundle.model
        processor = bundle.processor
        ```

    Example:
        ```python
        # With config and processor overrides
        loader = HuggingFaceModelLoader()
        model_config = {
            "model_name": "nvidia/segformer-b2-finetuned-ade-512-512",
            "num_labels": 150,
            "config_overrides": {"ignore_mismatched_sizes": True},
            "processor_overrides": {"do_resize": False},
        }
        bundle = loader.load_model(model_config)
        ```

    Example:
        ```python
        # With task override
        loader = HuggingFaceModelLoader()
        model_config = {
            "model_name": "facebook/maskformer-swin-tiny-coco",
            "task": "instance_segmentation",
        }
        bundle = loader.load_model(model_config)
        ```
    """

    _hf = None  # class-level cache

    @classmethod
    def _import_transformers(cls):
        if cls._hf is None:
            try:
                cls._hf = importlib.import_module("transformers")
            except ImportError as err:
                raise ImportError("transformers is not installed. Install it or choose another model origin.") from err
        return cls._hf

    def load_model(self, model_config: dict[str, Any]) -> "HFSegmentationBundle | nn.Module":
        """Load HuggingFace model and optionally its processor.

        Args:
            model_config (dict[str, Any]):
                - `model_name` (str): Model name/path (required).
                - `model_cls` (Callable): Model class to use (optional).
                - `num_labels` (int): Number of classes (optional).
                - `trust_remote_code` (bool): Trust remote code (optional).
                - `task` (str): Model task (optional).
                - `return_processor` (bool): Return processor (optional).
                - `config_overrides` (dict[str, Any]): Config attribute overrides (optional).
                - `processor_overrides` (dict[str, Any]): Processor attribute overrides (optional).

        Returns:
            `HFSegmentationBundle | nn.Module`: Model (and processor if requested).

        """
        try:
            transformers = self._import_transformers()
            model_name = model_config["model_name"]
            model_cls = model_config.get("model_cls", None)
            task = model_config.get("task", None)
            return_processor = model_config.get("return_processor", True)
            trust_remote_code = model_config.get("trust_remote_code", False)

            config = transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
            if "num_labels" in model_config:
                config.num_labels = model_config["num_labels"]
            if "config_overrides" in model_config:
                for k, v in model_config["config_overrides"].items():
                    setattr(config, k, v)

            if model_cls is None:
                if task == "semantic_segmentation":
                    model_cls = transformers.AutoModelForSemanticSegmentation
                elif task == "instance_segmentation":
                    model_cls = transformers.AutoModelForInstanceSegmentation
                elif task == "panoptic_segmentation":
                    model_cls = transformers.AutoModelForPanopticSegmentation
                elif task == "image_segmentation":
                    model_cls = transformers.AutoModelForImageSegmentation
                else:
                    model_cls = transformers.AutoModel
            else:
                model_cls = getattr(transformers, model_cls)

            model = model_cls.from_pretrained(model_name, config=config, trust_remote_code=trust_remote_code)
            logger.info(f"Loaded HuggingFace model: {model_name} (task: {task})")

            if return_processor:
                processor = transformers.AutoImageProcessor.from_pretrained(
                    model_name, trust_remote_code=trust_remote_code
                )
                if "processor_overrides" in model_config:
                    for k, v in model_config["processor_overrides"].items():
                        setattr(processor, k, v)
                from segmentation_robustness_framework.loaders.models.hf_bundle import HFSegmentationBundle

                logger.info(f"Loaded processor for HuggingFace model: {model_name}")
                return HFSegmentationBundle(model=model, processor=processor)
            return model
        except Exception as e:
            logger.exception(f"Failed to load HuggingFace model: {e}")
            raise

    def load_weights(self, model: nn.Module, weights_path: str, weight_type: str = "full") -> nn.Module:
        """Load weights into HuggingFace model.

        Args:
            model (nn.Module): Model instance.
            weights_path (str | Path): Path to weights file.
            weight_type (str): 'full' for entire model, 'encoder' for encoder only.

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
                result = model.load_state_dict(state_dict, strict=False)
                if hasattr(result, "missing_keys") and hasattr(result, "unexpected_keys"):
                    missing = result.missing_keys
                    unexpected = result.unexpected_keys
                    if missing:
                        logger.warning(f"Missing keys when loading full model weights: {missing}")
                    if unexpected:
                        logger.warning(f"Unexpected keys when loading full model weights: {unexpected}")
                logger.info(f"Loaded full model weights into HuggingFace model from {weights_path}")
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
                    logger.info(f"Loaded encoder (backbone) weights into HuggingFace model from {weights_path}")
                else:
                    logger.info(
                        f"Loaded encoder weights (no missing/unexpected keys info available) into HuggingFace model from {weights_path}"
                    )
            else:
                logger.warning(f"Unknown weight_type: {weight_type}. No weights loaded.")
            return model
        except Exception as e:
            logger.exception(f"Failed to load weights into HuggingFace model: {e}")
            raise
