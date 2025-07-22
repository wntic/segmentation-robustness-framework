from unittest.mock import patch

import pytest
import torch
from segmentation_robustness_framework.attacks.attack import AdversarialAttack
from segmentation_robustness_framework.attacks.tpgd import TPGD
from tests.attacks.base_attack_common import model  # noqa: F401


@pytest.fixture
def attack(model):  # noqa: F811
    return TPGD(model, eps=0.1, alpha=0.01, iters=5)


def test_initialization(model):  # noqa: F811
    attack = TPGD(model, eps=0.1, alpha=0.01, iters=5)
    assert attack.model == model
    assert attack.eps == 0.1
    assert attack.alpha == 0.01
    assert attack.iters == 5
    assert isinstance(attack, AdversarialAttack)


def test_initialization_defaults(model):  # noqa: F811
    attack = TPGD(model)
    assert attack.eps == 8 / 255
    assert attack.alpha == 2 / 255
    assert attack.iters == 10


def test_repr(attack):
    repr_str = repr(attack)
    assert "TPGD attack" in repr_str
    assert "eps=0.1" in repr_str
    assert "alpha=0.01" in repr_str
    assert "iters=5" in repr_str


def test_get_params(attack):
    params = attack.get_params()
    assert isinstance(params, dict)
    assert "epsilon" in params
    assert "alpha" in params
    assert "iters" in params
    assert params["epsilon"] == 0.1
    assert params["alpha"] == 0.01
    assert params["iters"] == 5


def test_call_method(attack):
    device = next(attack.model.parameters()).device
    image = torch.randn(1, 3, 32, 32, device=device)

    with patch.object(attack, "apply") as mock_apply:
        mock_apply.return_value = image + 0.1
        result = attack(image)

        mock_apply.assert_called_once_with(image, None)
        assert torch.allclose(result, image + 0.1)


def test_call_method_with_labels(attack):
    device = next(attack.model.parameters()).device
    image = torch.randn(1, 3, 32, 32, device=device)
    labels = torch.randint(0, 10, (1, 32, 32), device=device)

    with patch.object(attack, "apply") as mock_apply:
        mock_apply.return_value = image + 0.1
        result = attack(image, labels)

        mock_apply.assert_called_once_with(image, labels)
        assert torch.allclose(result, image + 0.1)


def test_apply_basic_functionality(attack, model):  # noqa: F811
    device = next(model.parameters()).device
    image = torch.randn(1, 3, 32, 32, device=device)

    model.eval()
    result = attack.apply(image)

    assert result.shape == image.shape
    assert result.dtype == image.dtype
    assert not torch.allclose(result, image)


def test_apply_with_labels_ignored(attack, model):  # noqa: F811
    device = next(model.parameters()).device
    image = torch.randn(1, 3, 32, 32, device=device)
    labels = torch.randint(0, 10, (1, 32, 32), device=device)

    model.eval()
    result_with_labels = attack.apply(image, labels)
    result_without_labels = attack.apply(image)

    assert result_with_labels.shape == image.shape
    assert result_without_labels.shape == image.shape
    assert not torch.allclose(result_with_labels, image)
    assert not torch.allclose(result_without_labels, image)


def test_apply_with_different_eps_values(model):  # noqa: F811
    device = next(model.parameters()).device
    eps_values = [0.01, 0.05, 0.1, 0.2]

    image = torch.randn(1, 3, 32, 32, device=device)

    for eps in eps_values:
        attack = TPGD(model, eps=eps, alpha=0.01, iters=3)
        model.eval()

        result = attack.apply(image)

        assert result.shape == image.shape
        assert not torch.allclose(result, image)


def test_apply_with_different_alpha_values(model):  # noqa: F811
    device = next(model.parameters()).device
    alpha_values = [0.005, 0.01, 0.02, 0.05]

    image = torch.randn(1, 3, 32, 32, device=device)

    for alpha in alpha_values:
        attack = TPGD(model, eps=0.1, alpha=alpha, iters=3)
        model.eval()

        result = attack.apply(image)

        assert result.shape == image.shape
        assert not torch.allclose(result, image)


def test_apply_with_different_iterations(model):  # noqa: F811
    device = next(model.parameters()).device
    iterations = [1, 3, 5, 10]

    image = torch.randn(1, 3, 32, 32, device=device)

    for iters in iterations:
        attack = TPGD(model, eps=0.1, alpha=0.01, iters=iters)
        model.eval()

        result = attack.apply(image)

        assert result.shape == image.shape
        assert not torch.allclose(result, image)


def test_apply_with_edge_case_images(attack, model):  # noqa: F811
    device = next(model.parameters()).device

    image = torch.zeros(1, 3, 32, 32, device=device)
    model.eval()
    result = attack.apply(image)
    assert result.shape == image.shape

    image = torch.ones(1, 3, 32, 32, device=device)
    model.eval()
    result = attack.apply(image)
    assert result.shape == image.shape


def test_apply_with_negative_values(attack, model):  # noqa: F811
    device = next(model.parameters()).device
    image = torch.randn(1, 3, 32, 32, device=device)
    image[image < 0] = -1.0

    model.eval()
    result = attack.apply(image)

    assert result.shape == image.shape
    assert torch.all(result >= 0)
    assert torch.all(result <= 1)


def test_apply_with_values_above_one(attack, model):  # noqa: F811
    device = next(model.parameters()).device
    image = torch.randn(1, 3, 32, 32, device=device)
    image[image > 0] = 2.0

    model.eval()
    result = attack.apply(image)

    assert result.shape == image.shape
    assert torch.all(result >= 0)
    assert torch.all(result <= 1)


def test_apply_with_different_channels(model):  # noqa: F811
    device = next(model.parameters()).device
    channels = [3, 3, 3]

    for num_channels in channels:
        attack = TPGD(model, eps=0.1, alpha=0.01, iters=3)
        image = torch.randn(1, num_channels, 32, 32, device=device)

        model.eval()
        result = attack.apply(image)

        assert result.shape == image.shape
        assert not torch.allclose(result, image)
