# This file contains code borrowed from the following repository:
# Original repository URL: https://github.com/Harry24k/adversarial-attacks-pytorch
# Original code author: Harry Kim
# Original code license: MIT

# The original code may have been modified to fit current requirements.

import torch
import torch.nn as nn

from .attack import AdversarialAttack
from .registry import register_attack


@register_attack("fgsm")
class FGSM(AdversarialAttack):
    """Fast Gradient Sign Method (FGSM) method from "Explaining and harnessing adversarial examples".
    Paper: https://arxiv.org/abs/1412.6572

    Attributes:
        model (nn.Module): The model that the adversarial attack will be applied to.
        eps (float): The magnitude of the perturbation.
    """

    def __init__(
        self,
        model: nn.Module,
        eps: float = 2 / 255,
    ):
        """Initialize FGSM attack.

        Args:
            model (nn.Module): The model that the adversarial attack will be applied to.
            eps (float): The magnitude of the perturbation. Defaults to 2/255.
        """
        super().__init__(model)
        self.eps = eps

    def __repr__(self) -> str:
        return f"FGSM attack: eps={self.eps}"

    def __call__(self, image: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Apply FGSM attack to input images.

        Args:
            image (torch.Tensor): Input image tensor [B, C, H, W].
            labels (torch.Tensor): Target labels tensor [B, H, W].

        Returns:
            torch.Tensor: Adversarial image tensor [B, C, H, W].
        """
        return self.apply(image, labels)

    def get_params(self) -> dict[str, float]:
        """Get attack parameters.

        Returns:
            dict[str, float]: Dictionary containing attack parameters.
        """
        return {"epsilon": self.eps}

    def apply(self, image: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Apply FGSM attack to input images.

        Args:
            image (torch.Tensor): Input image tensor [B, C, H, W].
            labels (torch.Tensor): Target labels tensor [B, H, W].

        Returns:
            torch.Tensor: Adversarial image tensor [B, C, H, W].
        """
        self.model.eval()

        image = image.to(self.device, non_blocking=True)
        image.requires_grad = True
        labels = labels.to(self.device, non_blocking=True)

        valid_mask = labels >= 0

        if not torch.any(valid_mask):
            return image.detach()

        outputs = self.model(image)
        self.model.zero_grad()

        # Reshape outputs and labels for loss computation
        # outputs: [B, C, H, W] -> [B*H*W, C]
        # labels: [B, H, W] -> [B*H*W]
        B, C, H, W = outputs.shape
        outputs_flat = outputs.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
        labels_flat = labels.reshape(-1)  # [B*H*W]

        # Only compute loss on valid pixels
        valid_indices = valid_mask.reshape(-1)  # [B*H*W]
        valid_outputs = outputs_flat[valid_indices]  # [N_valid, C]
        valid_labels = labels_flat[valid_indices]  # [N_valid]

        # Compute loss only on valid pixels
        loss = torch.nn.CrossEntropyLoss()
        cost = loss(valid_outputs, valid_labels)
        cost.backward()

        adv_image = image + self.eps * image.grad.sign()
        adv_image = torch.clamp(adv_image, 0, 1)

        # Memory cleanup
        del outputs, outputs_flat, labels_flat, valid_outputs, valid_labels, cost
        if self.device == "cuda":
            torch.cuda.empty_cache()

        return adv_image
