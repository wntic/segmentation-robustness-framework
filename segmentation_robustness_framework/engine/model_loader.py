import torch
import segmentation_robustness_framework.models as models
from segmentation_robustness_framework.config import ModelConfig


class ModelLoader:
    def __init__(self, model_config: ModelConfig):
        self.config = model_config
        self.model_name = model_config.name
        self.encoder = model_config.encoder
        self.weights = model_config.weights
        self.num_classes = model_config.num_classes
        self.device = model_config.device

    def load_model(self):
        if self.model_name in ["DeepLabV3", "FCN"]:
            model = self._load_torchvision_model()
        else:
            model = self._load_model()

        model = model.to(self.device)
        return model

    def _load_torchvision_model(self):
        if self.weights == "default":
            self.weights = "coco_with_voc_labels"

        if self.model_name == "FCN" and self.encoder in ["resnet50", "resnet101"]:
            model = models.FCN(
                encoder_name=self.encoder,
                encoder_weights=self.weights,
                num_classes=self.num_classes,
            )
        elif self.model_name == "DeepLabV3" and self.encoder in ["resnet50", "resnet101", "mobilenet_v3_large"]:
            model = models.DeepLabV3(
                encoder_name=self.encoder,
                encoder_weights=self.weights,
                num_classes=self.num_classes,
            )
        else:
            raise ValueError(f"Invalid encoder: {self.encoder}")

        if self.num_classes != 21:
            model.classifier[4] = torch.nn.Conv2d(model.classifier[4].in_channels, self.num_classes, kernel_size=1)

        return model

    def _load_model(self):
        try:
            model_class = getattr(models, self.model_name)
        except AttributeError:
            raise ValueError(f"Model '{self.model_name}' not found in segmentation_robustness_framework.models.")

        model = model_class(encoder=self.encoder, num_classes=self.num_classes)

        if self.weights and self.weights != "default":
            model.load_state_dict(torch.load(self.weights, weights_only=True))

        return model
