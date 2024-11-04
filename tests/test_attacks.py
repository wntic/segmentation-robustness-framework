import pytest
import torch
from segmentation_robustness_framework import attacks, models


DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def segmentation_model():
    """Fixture to create base segmentation model."""
    model = models.SegmentationModel(encoder_name="resnet50", encoder_weights="imagenet", num_classes=21)
    model.eval()
    return model.to(DEVICE)


class TestBaseAttack:
    def test_base_attack(self):
        atk = attacks.AdversarialAttack(model=segmentation_model)
        assert atk.model == segmentation_model

    def test_attack(self):
        atk = attacks.AdversarialAttack(segmentation_model)
        image = torch.rand(1, 3, 16, 16)
        labels = torch.rand(1, 3, 16, 16)

        with pytest.raises(NotImplementedError, match="The method must be implemented in a subclass."):
            atk.attack(image, labels)

    def test_set_device(self):
        atk = attacks.AdversarialAttack(model=segmentation_model)
        assert atk.device is None
        atk.set_device("cuda")
        assert atk.device == torch.device("cuda")


class TestFGSM:
    def test_initialization(self):
        atk = attacks.FGSM(model=segmentation_model, eps=0.05)
        assert atk.model == segmentation_model
        assert atk.eps == 0.05

    def test_get_params(self):
        atk = attacks.FGSM(model=segmentation_model, eps=0.05)
        params = atk.get_params()

        assert isinstance(params, dict)
        assert "epsilon" in params
        assert params["epsilon"] == 0.05

    def test_repr(self):
        atk = attacks.FGSM(model=segmentation_model, eps=0.05)
        assert repr(atk) == "FGSM attack: eps=0.05"


class TestPGD:
    def test_initialization(self):
        atk = attacks.PGD(model=segmentation_model, eps=0.05, alpha=2 / 255, iters=10, targeted=False)

        assert atk.model == segmentation_model
        assert atk.eps == 0.05
        assert atk.alpha == 2 / 255
        assert atk.iters == 10
        assert not atk.targeted

    def test_get_params(self):
        atk = attacks.PGD(model=segmentation_model, eps=2 / 255, alpha=2 / 255, iters=10, targeted=False)
        params = atk.get_params()

        assert isinstance(params, dict)

        assert "epsilon" in params
        assert "alpha" in params
        assert "iters" in params

        assert params["epsilon"] == 2 / 255
        assert params["alpha"] == 2 / 255
        assert params["iters"] == 10

    def test_repr(self):
        atk = attacks.PGD(model=segmentation_model, eps=0.1, alpha=0.07, iters=10, targeted=False)
        assert repr(atk) == "PGD attack: eps=0.1, alpha=0.07, iters=10, targeted=False"


class TestRFGSM:
    def test_initialization(self):
        atk = attacks.RFGSM(model=segmentation_model, eps=0.05, alpha=2 / 255, iters=10, targeted=False)

        assert atk.model == segmentation_model
        assert atk.eps == 0.05
        assert atk.alpha == 2 / 255
        assert atk.iters == 10
        assert not atk.targeted

    def test_get_params(self):
        atk = attacks.RFGSM(model=segmentation_model, eps=8 / 255, alpha=2 / 255, iters=10, targeted=False)
        params = atk.get_params()

        assert isinstance(params, dict)

        assert "epsilon" in params
        assert "alpha" in params
        assert "iters" in params

        assert params["epsilon"] == 8 / 255
        assert params["alpha"] == 2 / 255
        assert params["iters"] == 10

    def test_repr(self):
        atk = attacks.RFGSM(model=segmentation_model, eps=0.1, alpha=0.07, iters=10, targeted=False)
        assert repr(atk) == "R+FGSM attack: eps=0.1, alpha=0.07, iters=10, targeted=False"


class TestTPGD:
    def test_initialization(self):
        atk = attacks.TPGD(segmentation_model, eps=8 / 255, alpha=2 / 255, iters=10)

        assert atk.model == segmentation_model
        assert atk.eps == 8 / 255
        assert atk.alpha == 2 / 255
        assert atk.iters == 10

    def test_get_params(self):
        atk = attacks.TPGD(segmentation_model, eps=8 / 255, alpha=2 / 255, iters=10)
        params = atk.get_params()

        assert isinstance(params, dict)

        assert "epsilon" in params
        assert "alpha" in params
        assert "iters" in params

        assert params["epsilon"] == 8 / 255
        assert params["alpha"] == 2 / 255
        assert params["iters"] == 10

    def test_repr(self):
        atk = attacks.TPGD(segmentation_model, eps=0.1, alpha=0.07, iters=10)
        assert repr(atk) == "TPGD attack: eps=0.1, alpha=0.07, iters=10"
