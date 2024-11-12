import torchvision.models.segmentation as models
from torch import Tensor

from .base_model import SegmentationModel

DEEPLABV3_ENCODERS = {
    "resnet50": models.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1,
    "resnet101": models.DeepLabV3_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1,
    "mobilenet_v3_large": models.DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1,
}


class DeepLabV3(SegmentationModel):
    """DeepLabV3 segmentation model from "Rethinking Atrous Convolution for Semantic Image Segmentation".
    Paper: https://arxiv.org/abs/1706.05587v3

    This class implements the DeepLabV3 architecture using different backbone encoders,
    such as ResNet50, ResNet101, and MobileNetV3 Large. It extends the `SegmentationModel`
    base class and allows for initialization with various pretrained weights.

    Attributes:
        encoder_name (str, optional): The name of the backbone encoder to be used. Supported encoders
            are "resnet50", "resnet101", and "mobilenet_v3_large". Defaults to "resnet50".
        weights (str, optional): Encoder's pretrained weights. Typically, this is "coco_with_voc_labels".
            Defaults to "coco_with_voc_labels".
        num_classes (int, optional): Number of classes in the dataset used. Defaults to 21.
    """

    def __init__(
        self,
        encoder_name: str = "resnet50",
        weights: str = "coco_with_voc_labels",
        num_classes: int = 21,
    ):
        """Initializes the DeepLabV3 segmentation model with the given encoder name, encoder weights,
        and number of classes.

        Args:
            encoder_name (str, optional): The name of the backbone encoder to be used. Supported encoders
                are "resnet50", "resnet101", and "mobilenet_v3_large". Defaults to "resnet50".
            weights (str, optional): Encoder's pretrained weights. Typically, this is "coco_with_voc_labels".
                Defaults to "coco_with_voc_labels".
            num_classes (int, optional): Number of classes in the dataset used. Defaults to 21.

        Raises:
            ValueError: If the encoder is not supported
        """
        super().__init__(encoder_name, weights, num_classes)
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
            raise ValueError(f'Encoder "{encoder_name}" is not supported for the DeepLabV3.')

    def forward(self, x: Tensor) -> Tensor:
        """Defines the forward pass of the DeepLabV3 model.

        Args:
            x (torch.Tensor): 4D input image tensor with shape `[1, C, H, W]`.

        Returns:
            torch.Tensor: The output tensor after passing through the DeepLabV3.
        """
        return self.model(x)["out"]
