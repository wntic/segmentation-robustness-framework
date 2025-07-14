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
        model (SegmentationModel): The model that the adversarial attack will be applied to.
        eps (float): Strength of the attack or maximum perturbation.
        alpha (float): Step size.
        iters: (int): Number of iters.
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
            model (SegmentationModel): The model that the adversarial attack will be applied to.
            eps (float, optional): Strength of the attack or maximum perturbation. Defaults to 8/255.
            alpha (float, optional): Step size. Defaults to 2/255.
            iters (int, optional): Number of iters. Defaults to 10.
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

    def __call__(self, image: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Allows the object to be called like a function to perform the attack."""
        return self.attack(image, labels)

    def get_params(self) -> dict[str, float]:
        return {"epsilon": self.eps, "alpha": self.alpha, "iters": self.iters, "targeted": self.targeted}

    def attack(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Overriden.
        """
        self.model.eval()

        images = images.to(self.device)
        labels = labels.to(self.device)

        adv_images = images + (self.eps - self.alpha) * torch.randn_like(images).sign()
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

        loss = torch.nn.CrossEntropyLoss()

        for _ in range(self.iters):
            adv_images.requires_grad = True
            outputs = self.model(adv_images)

            if self.targeted:
                cost = -loss(outputs, labels)
            else:
                cost = loss(outputs, labels)

            grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()

        return adv_images
