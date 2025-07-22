from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn
from segmentation_robustness_framework.attacks import AdversarialAttack
from segmentation_robustness_framework.attacks.registry import ATTACK_REGISTRY
from segmentation_robustness_framework.loaders.attack_loader import AttackLoader


class MockAttack(AdversarialAttack):
    def __init__(self, model: nn.Module, eps: float = 0.1, **kwargs):
        super().__init__(model)
        self.eps = eps
        self.kwargs = kwargs

    def apply(self, image: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        return image + self.eps


def MockAttackFactory(model: nn.Module, **kwargs):
    return [MockAttack(model, **kwargs)]


class TestAttackLoader:
    def setup_method(self):
        self.mock_model = Mock(spec=nn.Module)
        self.mock_model.parameters.return_value = [torch.randn(10, 10)]

        ATTACK_REGISTRY.clear()

    def test_initialization(self):
        attack_config = [{"name": "test_attack", "eps": 0.1}, {"name": "test_attack2", "eps": 0.2}]

        loader = AttackLoader(self.mock_model, attack_config)

        assert loader.model == self.mock_model
        assert loader.config == attack_config

    def test_initialization_with_empty_config(self):
        attack_config = []

        loader = AttackLoader(self.mock_model, attack_config)

        assert loader.model == self.mock_model
        assert loader.config == []

    def test_load_attacks_single_config(self):
        ATTACK_REGISTRY["test_attack"] = MockAttackFactory

        attack_config = [{"name": "test_attack", "eps": 0.1}]
        loader = AttackLoader(self.mock_model, attack_config)

        attacks = loader.load_attacks()

        assert len(attacks) == 1
        assert len(attacks[0]) == 1
        assert isinstance(attacks[0][0], MockAttack)
        assert attacks[0][0].eps == 0.1

    def test_load_attacks_multiple_configs(self):
        ATTACK_REGISTRY["test_attack"] = MockAttackFactory

        attack_config = [{"name": "test_attack", "eps": 0.1}, {"name": "test_attack", "eps": 0.2}]
        loader = AttackLoader(self.mock_model, attack_config)

        attacks = loader.load_attacks()

        assert len(attacks) == 2
        assert len(attacks[0]) == 1
        assert len(attacks[1]) == 1
        assert attacks[0][0].eps == 0.1
        assert attacks[1][0].eps == 0.2

    def test_load_attacks_with_additional_parameters(self):
        ATTACK_REGISTRY["test_attack"] = MockAttackFactory

        attack_config = [{"name": "test_attack", "eps": 0.1, "alpha": 0.01, "steps": 10}]
        loader = AttackLoader(self.mock_model, attack_config)

        attacks = loader.load_attacks()

        assert len(attacks) == 1
        assert len(attacks[0]) == 1
        attack = attacks[0][0]
        assert attack.eps == 0.1
        assert attack.kwargs["alpha"] == 0.01
        assert attack.kwargs["steps"] == 10

    def test_load_attacks_empty_config(self):
        attack_config = []
        loader = AttackLoader(self.mock_model, attack_config)

        attacks = loader.load_attacks()

        assert attacks == []

    def test_create_attack_instances_unknown_attack(self):
        attack_config = {"name": "unknown_attack", "eps": 0.1}
        loader = AttackLoader(self.mock_model, [])

        with pytest.raises(ValueError, match="Unknown attack type: unknown_attack"):
            loader._create_attack_instances(attack_config)

    def test_create_attack_instances_missing_name(self):
        attack_config = {"eps": 0.1}
        loader = AttackLoader(self.mock_model, [])

        with pytest.raises(KeyError):
            loader._create_attack_instances(attack_config)

    def test_load_attacks_with_registered_attack(self):
        def attack_factory(model, **kwargs):
            return [MockAttack(model, **kwargs)]

        ATTACK_REGISTRY["real_attack"] = attack_factory

        attack_config = [{"name": "real_attack", "eps": 0.15}]
        loader = AttackLoader(self.mock_model, attack_config)

        attacks = loader.load_attacks()

        assert len(attacks) == 1
        assert len(attacks[0]) == 1
        assert isinstance(attacks[0][0], MockAttack)
        assert attacks[0][0].eps == 0.15

    def test_load_attacks_multiple_attacks_per_config(self):
        def multi_attack_factory(model, **kwargs):
            return [MockAttack(model, eps=kwargs.get("eps", 0.1)), MockAttack(model, eps=kwargs.get("eps2", 0.2))]

        ATTACK_REGISTRY["multi_attack"] = multi_attack_factory

        attack_config = [{"name": "multi_attack", "eps": 0.1, "eps2": 0.3}]
        loader = AttackLoader(self.mock_model, attack_config)

        attacks = loader.load_attacks()

        assert len(attacks) == 1
        assert len(attacks[0]) == 2
        assert attacks[0][0].eps == 0.1
        assert attacks[0][1].eps == 0.3

    def test_load_attacks_with_model_parameter(self):
        captured_model = None
        captured_kwargs = None

        def capture_factory(model, **kwargs):
            nonlocal captured_model, captured_kwargs
            captured_model = model
            captured_kwargs = kwargs
            return [MockAttack(model, **kwargs)]

        ATTACK_REGISTRY["capture_attack"] = capture_factory

        attack_config = [{"name": "capture_attack", "eps": 0.1}]
        loader = AttackLoader(self.mock_model, attack_config)

        loader.load_attacks()

        assert captured_model == self.mock_model
        assert captured_kwargs["eps"] == 0.1

    def test_load_attacks_preserves_config_order(self):
        ATTACK_REGISTRY["attack1"] = lambda model, **kwargs: [MockAttack(model, eps=0.1)]
        ATTACK_REGISTRY["attack2"] = lambda model, **kwargs: [MockAttack(model, eps=0.2)]

        attack_config = [{"name": "attack1", "eps": 0.1}, {"name": "attack2", "eps": 0.2}]
        loader = AttackLoader(self.mock_model, attack_config)

        attacks = loader.load_attacks()

        assert len(attacks) == 2
        assert attacks[0][0].eps == 0.1
        assert attacks[1][0].eps == 0.2

    def test_load_attacks_with_complex_parameters(self):
        ATTACK_REGISTRY["complex_attack"] = lambda model, **kwargs: [MockAttack(model, **kwargs)]

        attack_config = [
            {
                "name": "complex_attack",
                "eps": 0.1,
                "alpha": 0.01,
                "steps": 10,
                "targeted": True,
                "target_label": 5,
                "random_start": False,
            }
        ]
        loader = AttackLoader(self.mock_model, attack_config)

        attacks = loader.load_attacks()

        assert len(attacks) == 1
        attack = attacks[0][0]
        assert attack.eps == 0.1
        assert attack.kwargs["alpha"] == 0.01
        assert attack.kwargs["steps"] == 10
        assert attack.kwargs["targeted"] is True
        assert attack.kwargs["target_label"] == 5
        assert attack.kwargs["random_start"] is False

    def test_load_attacks_error_handling(self):
        def failing_factory(model, **kwargs):
            raise RuntimeError("Factory failed")

        ATTACK_REGISTRY["failing_attack"] = failing_factory

        attack_config = [{"name": "failing_attack", "eps": 0.1}]
        loader = AttackLoader(self.mock_model, attack_config)

        with pytest.raises(RuntimeError, match="Factory failed"):
            loader.load_attacks()

    def test_load_attacks_with_none_model(self):
        ATTACK_REGISTRY["test_attack"] = MockAttackFactory

        attack_config = [{"name": "test_attack", "eps": 0.1}]
        loader = AttackLoader(None, attack_config)

        attacks = loader.load_attacks()
        assert len(attacks) == 1

    def test_attack_loader_repr(self):
        attack_config = [{"name": "test_attack", "eps": 0.1}]
        loader = AttackLoader(self.mock_model, attack_config)

        repr_str = repr(loader)
        assert "AttackLoader" in repr_str
        assert isinstance(repr_str, str)

    def test_attack_loader_with_real_attack_registry(self):
        with patch("segmentation_robustness_framework.loaders.attack_loader.ATTACK_REGISTRY") as mock_registry:
            mock_registry.__contains__.return_value = True
            mock_registry.__getitem__.return_value = MockAttackFactory

            attack_config = [{"name": "fgsm", "eps": 0.1}]
            loader = AttackLoader(self.mock_model, attack_config)

            attacks = loader.load_attacks()

            assert len(attacks) == 1
            mock_registry.__getitem__.assert_called_once_with("fgsm")

    def test_attack_loader_config_immutability(self):
        original_config = [{"name": "test_attack", "eps": 0.1}]
        config_copy = [{"name": "test_attack", "eps": 0.1}]

        ATTACK_REGISTRY["test_attack"] = MockAttackFactory

        loader = AttackLoader(self.mock_model, config_copy)
        loader.load_attacks()

        assert config_copy == original_config

    def test_attack_loader_with_empty_attack_list(self):
        def empty_factory(model, **kwargs):
            return []

        ATTACK_REGISTRY["empty_attack"] = empty_factory

        attack_config = [{"name": "empty_attack", "eps": 0.1}]
        loader = AttackLoader(self.mock_model, attack_config)

        attacks = loader.load_attacks()

        assert len(attacks) == 1
        assert len(attacks[0]) == 0
