import torch

from segmentation_robustness_framework.adapters.base_protocol import SegmentationModelProtocol
from segmentation_robustness_framework.adapters.registry import register_adapter


@register_adapter("huggingface")
class HuggingFaceAdapter(torch.nn.Module, SegmentationModelProtocol):
    """Adapter for HuggingFace segmentation models.

    This adapter standardizes the interface for HuggingFace models that return an object with a 'logits' attribute.

    Attributes:
        model (torch.nn.Module): The underlying HuggingFace model.
        num_classes (int): Number of output classes.
    """

    def __init__(self, model: torch.nn.Module):
        """Initialize the adapter.

        Args:
            model (torch.nn.Module): HuggingFace segmentation model instance.
        """
        super().__init__()
        self.model = model

        if hasattr(model, "config") and hasattr(model.config, "num_labels"):
            self.num_classes = model.config.num_labels
        elif hasattr(model, "config") and hasattr(model.config, "num_classes"):
            self.num_classes = model.config.num_classes
        else:
            raise ValueError("Model config does not contain num_labels or num_classes")

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits for input images.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Logits tensor of shape (B, num_classes, H, W).
        """
        return self.model(pixel_values=x).logits

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
