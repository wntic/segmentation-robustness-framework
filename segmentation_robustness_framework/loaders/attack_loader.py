import inspect
from typing import Any

import torch.nn as nn

from segmentation_robustness_framework.attacks import AdversarialAttack
from segmentation_robustness_framework.attacks.registry import ATTACK_REGISTRY


class AttackLoader:
    """Loads adversarial attacks for a given segmentation model.

    The `AttackLoader` class initializes adversarial attacks based on the provided configuration.
    It supports multiple attack types and parameters, allowing the evaluation of model robustness.

    Attributes:
        model (nn.Module): The segmentation model to be attacked.
        config (list[dict[str, Any]]): List of configurations specifying the attack types and parameters.
    """

    def __init__(self, model: nn.Module, attack_config: list[dict[str, Any]]) -> None:
        """Initializes the `AttackLoader` with the given model and attack configuration.

        Args:
            model (nn.Module): The segmentation model to be attacked.
            attack_config (list[dict[str, Any]]): List of configurations specifying the attack types and parameters.
        """

        self.model = model
        self.config = attack_config

    def load_attacks(self) -> list[list[AdversarialAttack]]:
        """Creates and loads adversarial attack instances based on the configuration.

        Returns:
            list[list[AdversarialAttack]]: A nested list where each sublist contains instances
            of adversarial attacks for a specific attack configuration.
        """
        attacks = [self._create_attack_instances(attack_config) for attack_config in self.config]
        return attacks

    def _create_attack_instances(self, attack_config: dict[str, Any]) -> list[AdversarialAttack]:
        """Generates a list of adversarial attacks based on the configuration.

        Args:
            attack_config (dict[str, Any]): Configuration specifying the attack type and parameters.

        Returns:
            list[AdversarialAttack]: A list of adversarial attack instances.

        Raises:
            ValueError: If the specified attack type is not recognized.
        """
        attack_name = attack_config["name"]

        if attack_name not in ATTACK_REGISTRY:
            raise ValueError(f"Unknown attack type: {attack_name}")

        attack_params = {k: v for k, v in attack_config.items() if k != "name"}

        attack_class_or_factory = ATTACK_REGISTRY[attack_name]
        is_class = inspect.isclass(attack_class_or_factory)

        if is_class:

            def factory(model, **kwargs):
                return [attack_class_or_factory(model, **kwargs)]

            attack_instances = factory(model=self.model, **attack_params)
        else:
            attack_instances = attack_class_or_factory(model=self.model, **attack_params)

        return attack_instances
