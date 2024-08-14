import torch

from ..models import SegmentationModel
from .attack import AdversarialAttack


class PGD(AdversarialAttack):
    def __init__(
        self,
        model: SegmentationModel,
        eps: float = 2 / 255,
        alpha: float = 2 / 255,
        iters: int = 10,
        targeted: bool = False,
    ):
        super(PGD, self).__init__(model)
        self.eps = eps
        self.alpha = alpha
        self.iters = iters
        self.targeted = targeted

    def attack(self, image: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        device = next(self.model.parameters()).device

        image.to(device)
        labels.to(device)

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
