import pytest
import torch
from adversarial_segmentation_toolkit.attacks import FGSM, PGD, AdversarialAttack
from adversarial_segmentation_toolkit.models import SegmentationModel


class TestBaseAttack:
    def test_base_attack(self):
        model = SegmentationModel(encoder_name="resnet50", encoder_weights="imagenet", num_classes=21)
        atk = AdversarialAttack(model)
        assert atk.model == model

    def test_attack(self):
        model = SegmentationModel(encoder_name="resnet50", encoder_weights="imagenet", num_classes=21)
        atk = AdversarialAttack(model)
        image = torch.rand(1, 3, 16, 16)
        labels = torch.rand(1, 3, 16, 16)

        with pytest.raises(NotImplementedError, match="The method must be implemented in a subclass."):
            atk.attack(image, labels)


class TestFGSM:
    def test_initialization(self):
        model = SegmentationModel(encoder_name="resnet50", encoder_weights="imagenet", num_classes=21)
        atk = FGSM(model=model, eps=0.05, targeted=False)

        assert atk.model == model
        assert atk.eps == 0.05
        assert not atk.targeted


class TestPGD:
    def test_initialization(self):
        model = SegmentationModel(encoder_name="resnet50", encoder_weights="imagenet", num_classes=21)
        atk = PGD(model=model, eps=0.05, alpha=2 / 255, iters=10, targeted=False)

        assert atk.model == model
        assert atk.eps == 0.05
        assert atk.alpha == 2 / 255
        assert atk.iters == 10
        assert not atk.targeted
