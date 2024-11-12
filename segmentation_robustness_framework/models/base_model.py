from torch import Tensor, nn


class SegmentationModel(nn.Module):
    """Base class for all segmentation models.

    This class serves as a template for creating segmentation models with a specified encoder
    and a number of output classes.

    Attributes:
        encoder_name (str): The name of the encoder to be used in the segmentation model.
        weights (str): The pretrained weights for the encoder.
        num_classes (int): Number of classes in the dataset used.
        model (SegmentationModel): The segmentation model, to be defined by subclasses.
    """

    def __init__(self, encoder_name: str, weights: str, num_classes: int):
        """Initializes the SegmentationModel with the given encoder name, encoder weights, and number of classes.

        Args:
            encoder_name (str): The name of the backbone encoder to be used in the segmentation model.
            weights (str): Backbone encoder pretrained weights.
            num_classes (int): Number of classes in the dataset used.
        """
        super().__init__()
        self.encoder_name = encoder_name
        self.weights = weights
        self.num_classes = num_classes
        self.model = None

    def forward(self, x: Tensor):
        """Defines the forward pass of the model. Must be implemented by subclasses.

        Args:
            x (torch.Tensor): 4D input image tensor with shape `[1, C, H, W]`.

        Returns:
            torch.Tensor: The output tensor after passing through the segmentation model.

        Raises:
            NotImplementedError: If called directly from this base class.
        """
        raise NotImplementedError("This method should be implemented by subclasses.")
