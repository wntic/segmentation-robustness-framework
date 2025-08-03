import torch

from segmentation_robustness_framework.adapters.base_protocol import SegmentationModelProtocol


class CustomAdapter(torch.nn.Module, SegmentationModelProtocol):
    """Provide a template adapter for custom user segmentation models.

    This class demonstrates how to implement an adapter for a user-defined segmentation model.
    Users should modify this template to fit their model's output structure and register it
    using the adapter registry if desired.

    Attributes:
        model (torch.nn.Module): The underlying custom model.
        num_classes (int): Number of output classes.
    """

    def __init__(self, model: torch.nn.Module, num_classes: int = 1):
        """Initialize the custom adapter.

        Args:
            model (torch.nn.Module): Custom segmentation model instance.
            num_classes (int): Number of output classes.
        """
        super().__init__()
        self.model = model
        self.num_classes = num_classes

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits for input images.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Logits tensor of shape (B, num_classes, H, W).
        """
        # Modify this line to match your model's output
        return self.model(x)

    def predictions(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted class labels for input images.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Predicted label tensor of shape (B, H, W).
        """
        logits = self.logits(x)
        return torch.argmax(logits, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass through the model.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Logits tensor of shape (B, num_classes, H, W).
        """
        return self.logits(x)


# To use this adapter:
# 1. Copy and modify as needed for your model's output.
# 2. Register it with the registry if you want a different name:
#    from segmentation_robustness_framework.adapters.registry import register_adapter
#    @register_adapter("my_custom")
#    class MyCustomAdapter(CustomAdapter):
#        ...
#
# Do not forget to add forward method.
