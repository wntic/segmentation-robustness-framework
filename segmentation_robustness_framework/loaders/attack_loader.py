import torch.nn as nn

from segmentation_robustness_framework import attacks
from segmentation_robustness_framework.config import AttackConfig


class AttackLoader:
    def __init__(self, model: nn.Module, attack_config: list[AttackConfig]) -> None:
        self.model = model
        self.config = attack_config

    def load_attacks(self) -> list[list[attacks.AdversarialAttack]]:
        attacks = [self._create_attack_instances(attack_config) for attack_config in self.config]
        return attacks

    def _create_attack_instances(self, attack_config: AttackConfig) -> list[attacks.AdversarialAttack]:
        """Generates a list of adversarial attacks based on the configuration.

        Args:
            model (nn.Module): The segmentation model to attack.
            attack_config (validator.AttackConfig): Configuration specifying the attack type and parameters.

        Returns:
            list[attacks.AdversarialAttack]: A list of adversarial attack instances.

        Raises:
            ValueError: If the specified attack type is not recognized.
        """
        attack_name = attack_config.name

        if attack_name == "FGSM":
            epsilon_values = attack_config.epsilon
            return [attacks.FGSM(model=self.model, eps=epsilon) for epsilon in epsilon_values]
        elif attack_name == "PGD":
            epsilon_values = attack_config.epsilon
            alpha_values = attack_config.alpha
            iters = attack_config.steps
            targeted = attack_config.targeted
            return [
                attacks.PGD(model=self.model, eps=epsilon, alpha=alpha, iters=iters, targeted=targeted)
                for epsilon in epsilon_values
                for alpha in alpha_values
            ]
        elif attack_name == "RFGSM":
            epsilon_values = attack_config.epsilon
            alpha_values = attack_config.alpha
            iters = attack_config.steps
            targeted = attack_config.targeted
            return [
                attacks.RFGSM(model=self.model, eps=epsilon, alpha=alpha, iters=iters, targeted=targeted)
                for epsilon in epsilon_values
                for alpha in alpha_values
            ]
        elif attack_name == "TPGD":
            epsilon_values = attack_config.epsilon
            alpha_values = attack_config.alpha
            iters = attack_config.steps
            return [
                attacks.TPGD(model=self.model, eps=epsilon, alpha=alpha, iters=iters)
                for epsilon in epsilon_values
                for alpha in alpha_values
            ]
        raise ValueError(f"Unknown attack type: {attack_name}")
