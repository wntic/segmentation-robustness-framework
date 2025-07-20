import torch

from segmentation_robustness_framework.adapters.base_protocol import SegmentationModelProtocol
from segmentation_robustness_framework.adapters.registry import register_adapter


@register_adapter("torchvision")
class TorchvisionAdapter(torch.nn.Module, SegmentationModelProtocol):
    """Adapter for Torchvision segmentation models.

    This adapter standardizes the interface for Torchvision models that return a dict with an 'out' key.

    Attributes:
        model (torch.nn.Module): The underlying Torchvision model.
        num_classes (int): Number of output classes.
    """

    def __init__(self, model: torch.nn.Module):
        """Initialize the adapter.

        Args:
            model (torch.nn.Module): Torchvision segmentation model instance.
        """
        super().__init__()
        self.model = model

        if hasattr(model, "classifier") and hasattr(model.classifier, "out_channels"):
            self.num_classes = model.classifier.out_channels
        else:
            self.num_classes = getattr(model, "num_classes", 21)

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits for input images.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Logits tensor of shape (B, num_classes, H, W).
        """
        return self.model(x)["out"]

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
        """Forward pass.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Logits tensor of shape (B, num_classes, H, W).
        """
        return self.logits(x)
