from ..models import SegmentationModel


class AdversarialAttack:
    def __init__(self, model: SegmentationModel):
        self.model = model

    def attack(self, image, labels):
        raise NotImplementedError("The method must be implemented in a subclass.")
