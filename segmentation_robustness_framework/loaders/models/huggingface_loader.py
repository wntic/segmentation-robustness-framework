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
        - `task` (str): "semantic_segmentation", "instance_segmentation", etc. (default: "semantic_segmentation").
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
            task = model_config.get("task", "semantic_segmentation")
            return_processor = model_config.get("return_processor", True)
            trust_remote_code = model_config.get("trust_remote_code", False)

            config = transformers.AutoConfig.from_pretrained(model_name, trust_remote_code=trust_remote_code)
            if "num_labels" in model_config:
                config.num_labels = model_config["num_labels"]
            if "config_overrides" in model_config:
                for k, v in model_config["config_overrides"].items():
                    setattr(config, k, v)

            if task == "semantic_segmentation":
                model_cls = transformers.AutoModelForSemanticSegmentation
            elif task == "instance_segmentation":
                model_cls = transformers.AutoModelForInstanceSegmentation
            else:
                model_cls = transformers.AutoModel

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

    def load_weights(self, model: nn.Module, weights_path: str) -> nn.Module:
        """Load weights into HuggingFace model.

        Args:
            model (nn.Module): Model instance.
            weights_path (str | Path): Path to weights file.

        Returns:
            `nn.Module`: Model with loaded weights.
        """
        try:
            checkpoint = torch.load(weights_path, map_location="cpu")

            if "state_dict" in checkpoint:
                state_dict = checkpoint["state_dict"]
            else:
                state_dict = checkpoint

            model.load_state_dict(state_dict, strict=False)
            logger.info(f"Loaded weights into HuggingFace model from {weights_path}")
            return model
        except Exception as e:
            logger.exception(f"Failed to load weights into HuggingFace model: {e}")
            raise
