from unittest.mock import patch

import pytest
import torch
from segmentation_robustness_framework.attacks.attack import AdversarialAttack
from segmentation_robustness_framework.attacks.fgsm import FGSM


@pytest.fixture
def attack(model):
    return FGSM(model, eps=0.1)


def test_initialization(model):
    attack = FGSM(model, eps=0.1)
    assert attack.model == model
    assert attack.eps == 0.1
    assert isinstance(attack, AdversarialAttack)


def test_initialization_default_eps(model):
    attack = FGSM(model)
    assert attack.eps == 2 / 255


def test_initialization_custom_eps(model):
    custom_eps = 0.05
    attack = FGSM(model, eps=custom_eps)
    assert attack.eps == custom_eps


def test_repr(attack):
    repr_str = repr(attack)
    assert "FGSM attack" in repr_str
    assert "eps=0.1" in repr_str


def test_get_params(attack):
    params = attack.get_params()
    assert isinstance(params, dict)
    assert "epsilon" in params
    assert params["epsilon"] == 0.1


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


def test_apply_gradient_flow(attack, model):
    device = next(model.parameters()).device
    image = torch.randn(1, 3, 32, 32, device=device, requires_grad=True)
    labels = torch.randint(0, 10, (1, 32, 32), device=device)

    model.eval()
    result = attack.apply(image, labels)

    loss = result.sum()
    loss.backward()

    assert image.grad is not None
    assert image.grad.shape == image.shape


def test_apply_deterministic(attack, model):
    device = next(model.parameters()).device
    image = torch.randn(1, 3, 32, 32, device=device)
    labels = torch.randint(0, 10, (1, 32, 32), device=device)

    model.eval()
    result1 = attack.apply(image, labels)
    result2 = attack.apply(image, labels)

    assert torch.allclose(result1, result2)


def test_apply_with_different_eps_values(model):
    device = next(model.parameters()).device
    eps_values = [0.01, 0.05, 0.1, 0.2]

    image = torch.randn(1, 3, 32, 32, device=device)
    labels = torch.randint(0, 10, (1, 32, 32), device=device)

    for eps in eps_values:
        attack = FGSM(model, eps=eps)
        model.eval()

        result = attack.apply(image, labels)

        assert result.shape == image.shape
        assert not torch.allclose(result, image)
