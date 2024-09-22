from ..models import SegmentationModel


class AdversarialAttack:
    """Base class for adversarial attacks.

    Attributes:
        model (SegmentationModel): Segmentation model to be attacked.
    """

    def __init__(self, model: SegmentationModel):
        """Initializes adversarial attack.

        Args:
            model (SegmentationModel): Segmentation model to be attacked.
        """
        self.model = model

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
