from typing import Protocol, runtime_checkable

import torch


@runtime_checkable
class SegmentationModelProtocol(Protocol):
    """Define the interface for segmentation model adapters.

    All segmentation model adapters must implement this interface, providing methods for
    obtaining logits and predictions, and exposing the number of output classes.

    Attributes:
        num_classes (int): Number of output classes for segmentation.
    """

    num_classes: int

    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits for input images.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Logits tensor of shape (B, num_classes, H, W).
        """
        ...

    def predictions(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted class labels for input images.

        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).

        Returns:
            torch.Tensor: Predicted label tensor of shape (B, H, W).
        """
        ...
