import torch
import torch.nn as nn


class AdversarialAttack:
    """Base class for adversarial attacks.

    Attributes:
        model (SegmentationModel): Segmentation model to be attacked.
    """

    def __init__(self, model: nn.Module):
        """Initializes adversarial attack.

        Args:
            model (torch.nn.Module): Segmentation model to be attacked.
        """
        self.model = model

        try:
            self.device = next(model.parameters()).device
        except Exception:
            self.device = None
            print("Failed to set device. Try set_device().")

    def set_device(self, device):
        self.device = torch.device(device) if isinstance(device, str) else device

    def attack(self, image, labels):
        """Performs an attack on the segmentation model.

        This method should be implemented by subclasses to define the attack logic.

        Args:
            image (torch.Tensor): The input image tensor to be perturbed.
            labels (torch.Tensor): The true or target labels for the image.

        Raises:
            NotImplementedError
        """
        raise NotImplementedError("The method must be implemented in a subclass.")
