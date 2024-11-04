import torch

from ..models import SegmentationModel
from .attack import AdversarialAttack


class FGSM(AdversarialAttack):
    """Fast Gradient Sign Method (FGSM) method from "Explaining and harnessing adversarial examples".
    Paper: https://arxiv.org/abs/1412.6572

    Attributes:
        model (SegmentationModel): The model that the adversarial attack will be applied to.
        eps (float): The magnitude of the perturbation.
        targeted (bool): Indicates whether the attack is targeted or not.
    """

    def __init__(
        self,
        model: SegmentationModel,
        eps: float = 2 / 255,
    ):
        """Initializes FGSM attack.

        Args:
            model (SegmentationModel): The model that the adversarial attack will be applied to.
            eps (float, optional): The magnitude of the perturbation. Defaults to 2/255.
            targeted (bool, optional): If True, performs a targeted attack; otherwise, performs
                an untargeted attack. Defaults to False.
        """
        super().__init__(model)
        self.eps = eps

    def __repr__(self) -> str:
        return f"FGSM attack: eps={self.eps}"

    def __call__(self, image: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Allows the object to be called like a function to perform the attack."""
        return self.attack(image, labels)

    def get_params(self) -> dict[str, float]:
        return {"epsilon": self.eps}

    def attack(self, image: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Overriden.
        """
        self.model.eval()

        image = image.to(self.device)
        image.requires_grad = True
        labels = labels.to(self.device)

        loss = torch.nn.CrossEntropyLoss()

        outputs = self.model(image)
        self.model.zero_grad()

        cost = loss(outputs, labels)
        cost.backward()

        adv_image = image + self.eps * image.grad.sign()
        adv_image = torch.clamp(adv_image, 0, 1)
        return adv_image
