import pytest
from segmentation_robustness_framework.config.model_config import (
    CustomConfig,
    HuggingFaceConfig,
    SMPConfig,
    TorchvisionConfig,
)


def test_torchvision_config_validation():
    cfg = TorchvisionConfig(type="torchvision", name="deeplabv3_resnet50")
    assert cfg.num_classes == 21
    assert cfg.pretrained
    assert cfg.device == "cpu"
    assert cfg.weights_path is None


def test_smp_config_validation():
    cfg = SMPConfig(
        type="smp",
        architecture="Unet",
        encoder_name="resnet50",
        encoder_weights=None,
        classes=2,
        activation="softmax",
        weights_path="weights.pth",
        device="cuda",
    )
    assert cfg.architecture == "Unet"
    assert cfg.encoder_name == "resnet50"
    assert cfg.encoder_weights is None
    assert cfg.classes == 2
    assert cfg.activation == "softmax"
    assert cfg.weights_path == "weights.pth"
    assert cfg.device == "cuda"


def test_huggingface_config_validation():
    cfg = HuggingFaceConfig(
        type="huggingface",
        model_name="nvidia/segformer-b0-finetuned-ade-512-512",
        num_labels=150,
        trust_remote_code=True,
        weights_path="hf_weights.pth",
        device="cuda",
    )
    assert cfg.model_name == "nvidia/segformer-b0-finetuned-ade-512-512"
    assert cfg.num_labels == 150
    assert cfg.trust_remote_code
    assert cfg.weights_path == "hf_weights.pth"
    assert cfg.device == "cuda"


def test_custom_config_defaults():
    def dummy_model():
        pass

    cfg = CustomConfig(type="custom", model_class=dummy_model)
    assert cfg.model_class is dummy_model
    assert cfg.model_args == []
    assert cfg.model_kwargs == {}
    assert cfg.device == "cpu"
    assert cfg.weights_path is None


def test_invalid_type_raises():
    from pydantic import ValidationError

    with pytest.raises(ValidationError):
        TorchvisionConfig(type="invalid", name="deeplabv3_resnet50")
