# This file contains code borrowed from the following repository:
# Original repository URL: https://github.com/Harry24k/adversarial-attacks-pytorch
# Original code author: Harry Kim
# Original code license: MIT

# The original code may have been modified to fit current requirements.

import torch
import torch.nn as nn

from .attack import AdversarialAttack
from .registry import register_attack


@register_attack("rfgsm")
class RFGSM(AdversarialAttack):
    """Random Fast Gradient Sign Method (R+FGSM) from the paper "Ensemble Adversarial Training : Attacks and Defences".
    Paper: https://arxiv.org/abs/1705.07204

    Attributes:
        model (nn.Module): The model that the adversarial attack will be applied to.
        eps (float): Strength of the attack or maximum perturbation.
        alpha (float): Step size.
        iters (int): Number of iterations.
        targeted (bool): Indicates whether the attack is targeted or not.
    """

    def __init__(
        self,
        model: nn.Module,
        eps: float = 8 / 255,
        alpha: float = 2 / 255,
        iters: int = 10,
        targeted: bool = False,
    ):
        """Initializes R+FGSM attack.

        Args:
            model (nn.Module): The model that the adversarial attack will be applied to.
            eps (float, optional): Strength of the attack or maximum perturbation. Defaults to 8/255.
            alpha (float, optional): Step size. Defaults to 2/255.
            iters (int, optional): Number of iterations. Defaults to 10.
            targeted (bool, optional): If True, performs a targeted attack; otherwise, performs
                an untargeted attack. Defaults to False.
        """
        super().__init__(model)
        self.eps = eps
        self.alpha = alpha
        self.iters = iters
        self.targeted = targeted

    def __repr__(self) -> str:
        return f"R+FGSM attack: eps={self.eps}, alpha={self.alpha}, iters={self.iters}, targeted={self.targeted}"

    def __call__(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Apply R+FGSM attack to a batch of images.

        Args:
            images (torch.Tensor): Batch of input images [B, C, H, W].
            labels (torch.Tensor): Batch of target labels [B, H, W].

        Returns:
            torch.Tensor: Batch of adversarial images [B, C, H, W].
        """
        return self.apply(images, labels)

    def get_params(self) -> dict[str, float]:
        """Get attack parameters.

        Returns:
            dict[str, float]: Dictionary containing attack parameters.
        """
        return {"epsilon": self.eps, "alpha": self.alpha, "iters": self.iters, "targeted": self.targeted}

    def apply(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Apply R+FGSM attack to a batch of images.

        Args:
            images (torch.Tensor): Batch of input images [B, C, H, W].
            labels (torch.Tensor): Batch of target labels [B, H, W].

        Returns:
            torch.Tensor: Batch of adversarial images [B, C, H, W].
        """
        self.model.eval()

        images = images.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)

        if labels.dim() == 4 and labels.shape[1] == 1:
            labels = labels.squeeze(1)  # [B, H, W]

        valid_mask = labels >= 0

        if not torch.any(valid_mask):
            return images.detach()

        adv_images = images + (self.eps - self.alpha) * torch.randn_like(images).sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        loss = torch.nn.CrossEntropyLoss()

        for _ in range(self.iters):
            adv_images.requires_grad = True

            outputs = self.model(adv_images)  # [B, num_classes, H, W]

            # Reshape outputs and labels for loss computation
            # outputs: [B, C, H, W] -> [B*H*W, C]
            # labels: [B, H, W] -> [B*H*W]
            B, C, H, W = outputs.shape
            outputs_flat = outputs.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
            labels_flat = labels.reshape(-1)  # [B*H*W]

            valid_indices = valid_mask.reshape(-1)  # [B*H*W]
            valid_outputs = outputs_flat[valid_indices]  # [N_valid, C]
            valid_labels = labels_flat[valid_indices]  # [N_valid]

            if self.targeted:
                cost = -loss(valid_outputs, valid_labels)
            else:
                cost = loss(valid_outputs, valid_labels)

            self.model.zero_grad()
            grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

            # Memory cleanup
            del outputs, outputs_flat, labels_flat, valid_outputs, valid_labels, cost, grad
            if self.device == "cuda":
                torch.cuda.empty_cache()

        return adv_images
