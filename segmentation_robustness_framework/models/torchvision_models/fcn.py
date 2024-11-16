import torchvision.models.segmentation as models

from ..base_model import SegmentationModel

FCN_ENCODERS = {
    "resnet50": models.FCN_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1,
    "resnet101": models.FCN_ResNet101_Weights.COCO_WITH_VOC_LABELS_V1,
}


class TorchvisionFCN(SegmentationModel):
    """Fully Convolutional Network (FCN) from "Fully Convolutional Networks for Semantic Segmentation".
    Paper: https://arxiv.org/abs/1605.06211v1
    Source: torchvision

    This class implements the FCN architecture using either a ResNet50 or ResNet101
    encoder. It initializes the model based on the specified encoder.

    The model implementation is based on the paper:
    "Fully Convolutional Networks for Semantic Segmentation" [https://arxiv.org/abs/1605.06211v1].

    Attributes:
        encoder_name (str): The name of the encoder used in the FCN model. Supported encoders
            are "resnet50" and "resnet101". Defaults to "resnet50".
        encoder_weights (str, optional): Encoder's pretrained weights. Typically, this is "coco_with_voc_labels".
            Defaults to "coco_with_voc_labels".
        num_classes (int): The number of output classes for the segmentation task. Defaults to 21.
    """

    def __init__(
        self,
        encoder_name: str = "resnet50",
        encoder_weights: str = "coco_with_voc_labels",
        num_classes: int = 21,
    ):
        """Initializes the FCN model with the specified encoder, weights and number of classes.

        Args:
            encoder_name (str, optional): The name of the encoder to be used in the model. Must be "resnet50" or "resnet101". Defaults to "resnet50".
            num_classes (int, optional): The number of classes for the segmentation task. Default is 21.

        Raises:
            ValueError: If the specified encoder is not supported.
        """
        super().__init__(encoder_name, encoder_weights, num_classes)
        if encoder_name == "resnet50":
            self.model = models.fcn_resnet50(weights=FCN_ENCODERS["resnet50"], num_classes=self.num_classes)
        elif encoder_name == "resnet101":
            self.model = models.fcn_resnet101(weights=FCN_ENCODERS["resnet101"], num_classes=self.num_classes)
        else:
            raise ValueError(f'Encoder "{encoder_name}" is not supported for the FCN.')

    def forward(self, x):
        """Forward pass of the FCN model.

        Args:
            x (torch.Tensor): An input tensor.

        Returns:
            torch.Tensor: The output tensor after passing through the segmentation model.
        """
        return self.model(x)["out"]
