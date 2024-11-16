import torch
import importlib

import segmentation_robustness_framework.models as models
from segmentation_robustness_framework.config import ModelConfig


class ModelLoader:
    def __init__(self, model_config: ModelConfig):
        self.config = model_config
        self.origin = self.config.origin
        self.model_name = self.config.name
        self.encoder_name = self.config.encoder
        self.weights = self.config.weights
        self.num_classes = self.config.num_classes
        self.device = self.config.device

    def load_model(self) -> torch.nn.Module:
        if str.lower(self.origin) == "torchvision":
            model = self._load_torchvision_model()
        elif str.lower(self.origin) == "smp":
            model = self._load_smp_model()
        elif self.origin is None:
            model = self._load_local_model()
        else:
            raise ValueError(f'Unknown origin: "{self.origin}". Valid options are "torchvision", "smp", or "local"')

        model = model.to(self.device)
        return model

    def _load_torchvision_model(self) -> torch.nn.Module:
        if self.weights == "default":
            self.weights = "coco_with_voc_labels"

        if self.model_name == "FCN" and self.encoder_name in ["resnet50", "resnet101"]:
            model = models.TorchvisionFCN(
                encoder_name=self.encoder_name,
                encoder_weights=self.weights,
                num_classes=self.num_classes,
            )
        elif self.model_name == "DeepLabV3" and self.encoder_name in ["resnet50", "resnet101", "mobilenet_v3_large"]:
            model = models.TorchvisionDeepLabV3(
                encoder_name=self.encoder_name,
                encoder_weights=self.weights,
                num_classes=self.num_classes,
            )
        else:
            raise ValueError(f"Invalid encoder: {self.encoder}")

        return model

    def _load_smp_model(self) -> torch.nn.Module:
        try:
            smp = importlib.import_module("segmentation_models_pytorch")
        except ImportError:
            raise ImportError(
                "The 'segmentation_models_pytorch' module is not installed. "
                "Please install it to use models from smp."
            )

        if hasattr(smp, self.model_name):
            ModelClass = getattr(smp, self.model_name)
            model = ModelClass(
                encoder_name=self.encoder_name,
                encoder_weights="imagenet",  # Default encoder weights
                in_channels=3,
                classes=self.num_classes,
            )

            if self.weights:
                try:
                    state_dict = torch.load(self.weights, weights_only=True)
                    model.load_state_dict(state_dict)
                except Exception as e:
                    raise ValueError(f"Failed to load weights from {self.weights}: {e}")
        else:
            raise ValueError(f'Model "{self.model_name}" not found in segmentation_models_pytorch (smp).')

        return model.to(self.device)

    def _load_local_model(self) -> torch.nn.Module:
        try:
            ModelClass = getattr(models, self.model_name)
        except AttributeError:
            raise ValueError(f"Model '{self.model_name}' not found in segmentation_robustness_framework.models.")

        model = ModelClass(encoder=self.encoder_name, num_classes=self.num_classes)

        if self.weights and self.weights != "default":
            try:
                state_dict = torch.load(self.weights, weights_only=True)
                model.load_state_dict(state_dict)
            except Exception as e:
                raise ValueError(f"Failed to load weights from {self.weights}: {e}")

        return model.to(self.device)
