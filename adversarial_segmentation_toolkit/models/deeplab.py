import torchvision.models.segmentation as models

from .base_model import SegmentationModel

DEEPLABV3_ENCODERS = {
    "resnet50": models.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1,
    "resnet101": models.DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1,
    "mobilenet_v3_large": models.DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1,
}


class DeepLabV3(SegmentationModel):
    def __init__(
        self,
        encoder_name: str = "resnet50",
        encoder_weights: str = "coco_with_voc_labels",
        num_classes: int = 21,
    ):
        super().__init__(encoder_name, encoder_weights, num_classes)
        if encoder_name == "resnet50":
            self.model = models.deeplabv3_resnet50(
                weights=DEEPLABV3_ENCODERS["resnet50"],
                num_classes=self.num_classes,
            )
        elif encoder_name == "resnet101":
            self.model = models.deeplabv3_resnet101(
                weights=DEEPLABV3_ENCODERS["resnet101"],
                num_classes=self.num_classes,
            )
        elif encoder_name == "mobilenet_v3_large":
            self.model = models.deeplabv3_mobilenet_v3_large(
                weights=DEEPLABV3_ENCODERS["mobilenet_v3_large"],
                num_classes=self.num_classes,
            )
        else:
            raise ValueError(f"Encoder `{encoder_name}` is not supported for the DeepLabV3.")

    def forward(self, x):
        return self.model(x)["out"]
