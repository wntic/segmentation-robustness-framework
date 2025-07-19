from unittest.mock import Mock

import pytest
import torch
import torch.nn as nn
from segmentation_robustness_framework.attacks.attack import AdversarialAttack


class ConcreteAttack(AdversarialAttack):
    def apply(self, image: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return image + 0.1


@pytest.fixture
def mock_model():
    model = Mock(spec=nn.Module)
    model.parameters.return_value = [torch.randn(10, 10)]
    return model


def test_set_device_method(mock_model):
    attack = ConcreteAttack(mock_model)

    attack.set_device("cuda:0")
    assert attack.device == torch.device("cuda:0")

    cpu_device = torch.device("cpu")
    attack.set_device(cpu_device)
    assert attack.device == cpu_device

    assert attack.model == mock_model


def test_abstract_method_enforcement(mock_model):
    with pytest.raises(TypeError):
        AdversarialAttack(mock_model)


def test_concrete_attack_instantiation(mock_model):
    attack = ConcreteAttack(mock_model)

    assert isinstance(attack, AdversarialAttack)
    assert isinstance(attack, ConcreteAttack)
    assert attack.model == mock_model
    assert attack.device == torch.device("cpu")


def test_apply_method_implementation(mock_model):
    attack = ConcreteAttack(mock_model)
    image = torch.randn(1, 3, 32, 32)
    labels = torch.randint(0, 10, (1, 32, 32))

    result = attack.apply(image, labels)

    assert isinstance(result, torch.Tensor)
    assert result.shape == image.shape
    assert torch.allclose(result, image + 0.1)
