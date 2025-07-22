import pytest
from segmentation_robustness_framework.attacks import AdversarialAttack
from segmentation_robustness_framework.attacks.registry import register_attack


def test_register_attack_raises_error_if_attack_already_registered():
    @register_attack("test_attack")
    class TestAttack(AdversarialAttack):
        pass

    with pytest.raises(ValueError):

        @register_attack("test_attack")
        class TestAttack2(AdversarialAttack):
            pass
