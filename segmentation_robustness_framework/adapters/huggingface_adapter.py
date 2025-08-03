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
        try:
            device = next(self.model.parameters()).device
        except StopIteration:
            device = x.device

        if x.device != device:
            x = x.to(device, non_blocking=True)

        try:
            if x.requires_grad:
                self.model.train()
                output = self.model(pixel_values=x)
                logits = output.logits
                self.model.eval()
            else:
                with torch.no_grad():
                    output = self.model(pixel_values=x)
                    logits = output.logits

            if logits.device != device:
                logits = logits.to(device, non_blocking=True)

            return logits
        except AttributeError as e:
            raise e
        except Exception as e:
            if device.type == "cuda":
                torch.cuda.empty_cache()
            raise RuntimeError(f"HuggingFace model forward pass failed: {e}") from e

    def predictions(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted class labels for input images.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Predicted label tensor of shape (B, H, W).
        """
        logits = self.logits(x)
        predictions = torch.argmax(logits, dim=1)

        del logits
        try:
            if next(self.model.parameters()).device.type == "cuda":
                torch.cuda.empty_cache()
        except StopIteration:
            pass

        return predictions

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Logits tensor of shape (B, num_classes, H, W).
        """
        return self.logits(x)
