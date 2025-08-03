# This file contains code borrowed from the following repository:
# Original repository URL: https://github.com/Harry24k/adversarial-attacks-pytorch
# Original code author: Harry Kim
# Original code license: MIT

# The original code may have been modified to fit current requirements.

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attack import AdversarialAttack
from .registry import register_attack


@register_attack("tpgd")
class TPGD(AdversarialAttack):
    """PGD based on KL-Divergence loss from the paper "Theoretically Principled Trade-off between Robustness and Accuracy".
    Paper: https://arxiv.org/abs/1901.08573

    Attributes:
        model (nn.Module): The model that the adversarial attack will be applied to.
        eps (float): Strength of the attack or maximum perturbation.
        alpha (float): Step size.
        iters (int): Number of iterations.
    """

    def __init__(
        self,
        model: nn.Module,
        eps: float = 8 / 255,
        alpha: float = 2 / 255,
        iters: int = 10,
    ):
        """Initializes TPGD attack.

        Args:
            model (nn.Module): The model that the adversarial attack will be applied to.
            eps (float): Strength of the attack or maximum perturbation.
            alpha (float): Step size.
            iters (int): Number of iterations.
        """
        super().__init__(model)
        self.eps = eps
        self.alpha = alpha
        self.iters = iters

    def __repr__(self) -> str:
        return f"TPGD attack: eps={self.eps}, alpha={self.alpha}, iters={self.iters}"

    def __call__(self, images: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        """Apply TPGD attack to a batch of images.

        Args:
            images (torch.Tensor): Batch of input images [B, C, H, W].
            labels (torch.Tensor, optional): Batch of target labels [B, H, W]. Not used in TPGD.

        Returns:
            torch.Tensor: Batch of adversarial images [B, C, H, W].
        """
        return self.apply(images, labels)

    def get_params(self) -> dict[str, float]:
        return {"epsilon": self.eps, "alpha": self.alpha, "iters": self.iters}

    def apply(self, images: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        """Apply TPGD attack to a batch of images.

        Args:
            images (torch.Tensor): Batch of input images [B, C, H, W].
            labels (torch.Tensor, optional): Batch of target labels [B, H, W]. Not used in TPGD.

        Returns:
            torch.Tensor: Batch of adversarial images [B, C, H, W].
        """
        self.model.eval()

        images = images.clone().detach().to(self.device, non_blocking=True)

        with torch.no_grad():
            logit_ori = self.model(images).detach()

        adv_images = images + 0.001 * torch.randn_like(images)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        loss = torch.nn.KLDivLoss(reduction="sum")

        for _ in range(self.iters):
            adv_images.requires_grad = True

            logit_adv = self.model(adv_images)  # [B, num_classes, H, W]

            cost = loss(F.log_softmax(logit_adv, dim=1), F.softmax(logit_ori, dim=1))

            self.model.zero_grad()
            grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]

            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

            # Memory cleanup
            del logit_adv, cost, grad
            if self.device == "cuda":
                torch.cuda.empty_cache()

        return adv_images
