import logging
from abc import ABC, abstractmethod

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


class AdversarialAttack(ABC):
    """Define the base class for adversarial attacks.

    Attributes:
        model (nn.Module): Segmentation model to be attacked.
        device (str | torch.device): The device to use for the attack.
    """

    def __init__(self, model: nn.Module):
        """Initialize the adversarial attack.

        Args:
            model (nn.Module): Segmentation model to be attacked.
        """
        self.model = model

        try:
            self.device = next(model.parameters()).device
        except Exception:
            self.device = torch.device("cpu")
            logger.warning("Failed to detect model device. Using CPU. You can try `set_device()`")

    def set_device(self, device: str | torch.device) -> None:
        """Set the device for the attack.

        Args:
            device (str | torch.device): The device to use for the attack.
        """
        self.device = torch.device(device) if isinstance(device, str) else device
        logger.info(f"Attack device set to: {self.device}")

    @abstractmethod
    def apply(self, image: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Perform an attack on the segmentation model.

        This method should be implemented by subclasses to define the attack logic.

        Args:
            image (torch.Tensor): The input image tensor to be perturbed.
            labels (torch.Tensor): The true or target labels for the image.

        Returns:
            torch.Tensor: The perturbed image tensor.
        """
        pass
