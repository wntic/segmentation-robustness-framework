import torch


class SegmentationModel(torch.nn.Module):
    """Base class for all segmentation models.

    Classes implementing different segmentation model architectures must inherit this base class.

    Attributes:
        encoder_name (str): The name of the encoder to be used in the model.
        encoder_weights (str): The pretrained weights for the encoder model.
        num_classes (int): Number of classes in the dataset used.
    """

    def __init__(self, encoder_name: str, encoder_weights: str, num_classes: int):
        """Initializes the SegmentationModel with the given encoder name, encoder weights, and number of classes.

        Args:
            encoder_name (str): The name of the encoder to be used in the model.
            encoder_weights (str): The pretrained weights for the encoder.
            num_classes (int, optional): Number of classes in the dataset used.
        """
        super(SegmentationModel, self).__init__()
        self.encoder_name = encoder_name
        self.encoder_weights = encoder_weights
        self.num_classes = num_classes
        self.model = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass of the segmentation model.

        Args:
            x (torch.Tensor): An input tensor.

        Returns:
            torch.Tensor: The output tensor after passing through the segmentation model.

        Raises:
            NotImplementedError: If the segmentation model has not been defined in a subclass.
        """
        if self.model is None:
            raise NotImplementedError("This method should be implemented by subclasses.")
        return self.model(x)
