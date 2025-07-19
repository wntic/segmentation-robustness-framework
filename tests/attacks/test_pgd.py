from unittest.mock import patch

import pytest
import torch
from segmentation_robustness_framework.attacks.attack import AdversarialAttack
from segmentation_robustness_framework.attacks.pgd import PGD


@pytest.fixture
def attack(model):
    return PGD(model, eps=0.1, alpha=0.01, iters=5)


def test_initialization(model):
    attack = PGD(model, eps=0.1, alpha=0.01, iters=5, targeted=False)
    assert attack.model == model
    assert attack.eps == 0.1
    assert attack.alpha == 0.01
    assert attack.iters == 5
    assert attack.targeted is False
    assert isinstance(attack, AdversarialAttack)


def test_initialization_defaults(model):
    attack = PGD(model)
    assert attack.eps == 2 / 255
    assert attack.alpha == 2 / 255
    assert attack.iters == 10
    assert attack.targeted is False


def test_initialization_targeted(model):
    attack = PGD(model, targeted=True)
    assert attack.targeted is True


def test_repr(attack):
    repr_str = repr(attack)
    assert "PGD attack" in repr_str
    assert "eps=0.1" in repr_str
    assert "alpha=0.01" in repr_str
    assert "iters=5" in repr_str
    assert "targeted=False" in repr_str


def test_get_params(attack):
    params = attack.get_params()
    assert isinstance(params, dict)
    assert "epsilon" in params
    assert "alpha" in params
    assert "iters" in params
    assert "targeted" in params
    assert params["epsilon"] == 0.1
    assert params["alpha"] == 0.01
    assert params["iters"] == 5
    assert params["targeted"] is False


def test_call_method(attack):
    device = next(attack.model.parameters()).device
    image = torch.randn(1, 3, 32, 32, device=device)
    labels = torch.randint(0, 10, (1, 32, 32), device=device)

    with patch.object(attack, "apply") as mock_apply:
        mock_apply.return_value = image + 0.1
        result = attack(image, labels)

        mock_apply.assert_called_once_with(image, labels)
        assert torch.allclose(result, image + 0.1)


def test_apply_basic_functionality(attack, model):
    device = next(model.parameters()).device
    image = torch.randn(1, 3, 32, 32, device=device)
    labels = torch.randint(0, 10, (1, 32, 32), device=device)

    model.eval()
    result = attack.apply(image, labels)

    assert result.shape == image.shape
    assert result.dtype == image.dtype
    assert not torch.allclose(result, image)


def test_apply_with_valid_mask(attack, model):
    device = next(model.parameters()).device
    image = torch.randn(1, 3, 32, 32, device=device)
    labels = torch.randint(0, 10, (1, 32, 32), device=device)
    labels[0, 0, 0] = -1
    labels[0, 1, 1] = -1

    model.eval()
    result = attack.apply(image, labels)

    assert result.shape == image.shape
    assert not torch.allclose(result, image)


def test_apply_with_all_invalid_labels(attack, model):
    device = next(model.parameters()).device
    image = torch.randn(1, 3, 32, 32, device=device)
    labels = torch.full((1, 32, 32), -1, dtype=torch.long, device=device)

    model.eval()
    result = attack.apply(image, labels)

    assert torch.allclose(result, image)


def test_apply_with_mixed_valid_invalid_labels(attack, model):
    device = next(model.parameters()).device
    image = torch.randn(1, 3, 32, 32, device=device)
    labels = torch.randint(0, 10, (1, 32, 32), device=device)
    labels[0, :16, :16] = -1

    model.eval()
    result = attack.apply(image, labels)

    assert result.shape == image.shape
    assert not torch.allclose(result, image)


def test_apply_with_different_eps_values(model):
    device = next(model.parameters()).device
    eps_values = [0.01, 0.05, 0.1, 0.2]

    image = torch.randn(1, 3, 32, 32, device=device)
    labels = torch.randint(0, 10, (1, 32, 32), device=device)

    for eps in eps_values:
        attack = PGD(model, eps=eps, alpha=0.01, iters=3)
        model.eval()

        result = attack.apply(image, labels)

        assert result.shape == image.shape
        assert not torch.allclose(result, image)


def test_apply_with_different_alpha_values(model):
    device = next(model.parameters()).device
    alpha_values = [0.005, 0.01, 0.02, 0.05]

    image = torch.randn(1, 3, 32, 32, device=device)
    labels = torch.randint(0, 10, (1, 32, 32), device=device)

    for alpha in alpha_values:
        attack = PGD(model, eps=0.1, alpha=alpha, iters=3)
        model.eval()

        result = attack.apply(image, labels)

        assert result.shape == image.shape
        assert not torch.allclose(result, image)


def test_apply_with_different_iterations(model):
    device = next(model.parameters()).device
    iterations = [1, 3, 5, 10]

    image = torch.randn(1, 3, 32, 32, device=device)
    labels = torch.randint(0, 10, (1, 32, 32), device=device)

    for iters in iterations:
        attack = PGD(model, eps=0.1, alpha=0.01, iters=iters)
        model.eval()

        result = attack.apply(image, labels)

        assert result.shape == image.shape
        assert not torch.allclose(result, image)


def test_apply_targeted_vs_untargeted(model):
    device = next(model.parameters()).device
    image = torch.randn(1, 3, 32, 32, device=device)
    labels = torch.randint(0, 10, (1, 32, 32), device=device)

    attack_untargeted = PGD(model, eps=0.1, alpha=0.01, iters=3, targeted=False)
    attack_targeted = PGD(model, eps=0.1, alpha=0.01, iters=3, targeted=True)

    model.eval()
    result_untargeted = attack_untargeted.apply(image, labels)
    result_targeted = attack_targeted.apply(image, labels)

    assert result_untargeted.shape == image.shape
    assert result_targeted.shape == image.shape
    assert not torch.allclose(result_untargeted, image)
    assert not torch.allclose(result_targeted, image)


def test_apply_with_4d_labels(attack, model):
    device = next(model.parameters()).device
    image = torch.randn(1, 3, 32, 32, device=device)
    labels = torch.randint(0, 10, (1, 1, 32, 32), device=device)

    model.eval()
    result = attack.apply(image, labels)

    assert result.shape == image.shape
    assert not torch.allclose(result, image)
