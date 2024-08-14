import torch

from ..models import SegmentationModel
from .attack import AdversarialAttack


class FGSM(AdversarialAttack):
    def __init__(
        self,
        model: SegmentationModel,
        eps: float = 2 / 255,
        targeted: bool = False,
    ):
        super(FGSM, self).__init__(model)
        self.eps = eps
        self.targeted = targeted

    def attack(self, image: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        loss = torch.nn.CrossEntropyLoss()
        device = next(self.model.parameters()).device

        image.to(device)
        image.requires_grad = True
        labels.to(device)

        outputs = self.model(image)

        self.model.zero_grad()
        cost = loss(outputs, labels)
        cost.backward()

        if self.targeted:
            adv_image = image - self.eps * image.grad.sign()
        else:
            adv_image = image + self.eps * image.grad.sign()

        adv_image = torch.clamp(adv_image, 0, 1)
        return adv_image
