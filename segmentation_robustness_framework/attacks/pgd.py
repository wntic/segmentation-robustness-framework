import torch

from ..models import SegmentationModel
from .attack import AdversarialAttack


class PGD(AdversarialAttack):
    """Projected Gradient Descent (PGD) method from "Towards Deep Learning Models Resistant to Adversarial Attacks".
    Paper: https://arxiv.org/abs/1706.06083

    Attributes:
        model (SegmentationModel): The model that the adversarial attack will be applied to.
        eps (float): The magnitude of the perturbation.
        alpha (float): The step size for each iteration.
        iters (int): The number of iterations.
        targeted (bool): Indicates whether the attack is targeted or not.
    """

    def __init__(
        self,
        model: SegmentationModel,
        eps: float = 2 / 255,
        alpha: float = 2 / 255,
        iters: int = 10,
        targeted: bool = False,
    ):
        """Initializes PGD attack.

        Args:
            model (SegmentationModel): The model that the adversarial attack will be applied to.
            eps (float, optional): The magnitude of the perturbation. Defaults to 2/255.
            alpha (float, optional): The step size for each iteration. Defaults to 2/255.
            iters (int, optional): The number of iterations. Defaults to 10.
            targeted (bool, optional): If True, performs a targeted attack; otherwise, performs
                an untargeted attack. Defaults to False.
        """
        super().__init__(model)
        self.eps = eps
        self.alpha = alpha
        self.iters = iters
        self.targeted = targeted

    def __repr__(self) -> str:
        return f"PGD attack: eps={self.eps}, alpha={self.alpha}, iters={self.iters}, targeted={self.targeted}"

    def __call__(self, image: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Allows the object to be called like a function to perform the attack."""
        return self.attack(image, labels)

    def get_params(self) -> dict[str, float]:
        return {"epsilon": self.eps, "alpha": self.alpha, "iters": self.iters, "targeted": self.targeted}

    def attack(self, image: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """
        Overriden.
        """
        self.model.eval()
        
        image = image.to(self.device)
        labels = labels.to(self.device)

        loss = torch.nn.CrossEntropyLoss()
        adv_image = image.clone().detach()

        # Adv image random start point
        adv_image = adv_image + torch.empty_like(adv_image).uniform_(-self.eps, self.eps)
        adv_image = torch.clamp(adv_image, min=0, max=1).detach()

        for _ in range(self.iters):
            adv_image.requires_grad = True
            outputs = self.model(adv_image)

            # Calculate loss
            if self.targeted:
                cost = -loss(outputs, labels)
            else:
                cost = loss(outputs, labels)

            grad = torch.autograd.grad(cost, adv_image, retain_graph=False, create_graph=False)[0]

            adv_image = adv_image.detach() + self.alpha * grad.sign()
            delta = torch.clamp(adv_image - image, min=-self.eps, max=self.eps)
            adv_image = torch.clamp(image + delta, min=0, max=1).detach()

        return adv_image
