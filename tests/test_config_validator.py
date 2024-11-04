import pytest
from unittest.mock import patch
from pydantic import ValidationError
from segmentation_robustness_framework.config import ModelConfig, AttackConfig, DatasetConfig, Config


class TestModelConfig:
    def test_valid_config(self):
        config = ModelConfig(name="FCN", encoder="resnet50", weights="some_weights", num_classes=3, device="cuda")
        assert config.name == "FCN"
        assert config.encoder == "resnet50"
        assert config.weights == "some_weights"
        assert config.num_classes == 3
        assert config.device == "cuda"

    def test_config_invalid_encoder(self):
        with pytest.raises(ValidationError, match="Encoder mobilenet_v3_large is not supported for FCN"):
            ModelConfig(name="FCN", encoder="mobilenet_v3_large", weights="some_weights", num_classes=3, device="cpu")


class TestAttackConfig:
    def test_attack_config_fgsm_valid(self):
        config = AttackConfig(name="FGSM", epsilon=[0.03])
        assert config.name == "FGSM"
        assert config.epsilon == [0.03]

    def test_attack_config_fgsm_invalid_params(self):
        with pytest.raises(
            ValidationError, match="FGSM attack got unexpected parameter. Valid parameter is 'epsilon' only"
        ):
            AttackConfig(name="FGSM", epsilon=[0.03], alpha=[0.01])

    def test_attack_config_rfgsm_valid(self):
        config = AttackConfig(name="RFGSM", epsilon=[0.03], alpha=[0.01], steps=10, targeted=True, target_label=2)
        assert config.name == "RFGSM"
        assert config.epsilon == [0.03]
        assert config.alpha == [0.01]
        assert config.steps == 10

    def test_attack_config_rfgsm_missing_params(self):
        with pytest.raises(
            ValidationError,
            match=r"(parameters 'epsilon', 'alpha', 'steps' and 'targeted' should not be None|Input should be a valid list)",
        ):
            # Passing None to `alpha` to trigger the list-type validation error
            AttackConfig(name="RFGSM", epsilon=[0.03], alpha=None, steps=10, targeted=True)


class TestDatasetConfig:
    @patch("os.path.exists", return_value=True)  # Mocking the filesystem check
    def test_dataset_config_voc_valid(self, mock_exists):
        # Creating a valid configuration with required fields only
        config = DatasetConfig(name="VOC", root="/valid/path", split="val", image_shape=[512, 512])
        assert config.name == "VOC"
        assert config.split == "val"
        assert config.root == "/valid/path"

    @patch("os.path.exists", return_value=True)  # Mock the filesystem check for root path
    def test_dataset_config_voc_invalid_split(self, mock_exists):
        with pytest.raises(
            ValidationError, match="Input should be 'train', 'val', 'test', 'train_extra' or 'trainval'"
        ):
            DatasetConfig(name="VOC", root="/valid/path", split="invalid_split", image_shape=[512, 512])

    @patch("os.path.exists", return_value=True)  # Mock the filesystem check for root path
    def test_dataset_config_cityscapes_missing_mode(self, mock_exists):
        with pytest.raises(ValidationError, match=r"mode.*'fine' or 'coarse'"):
            DatasetConfig(
                name="Cityscapes", root="/valid/path", split="train", target_type=["semantic"], image_shape=[512, 512]
            )


class TestConfig:
    @patch("os.path.exists", return_value=True)  # Mock the filesystem check for root path
    def test_config_overall(self, mock_exists):
        model = ModelConfig(
            name="DeepLabV3", encoder="resnet101", weights="path/to/weights.pth", num_classes=3, device="cuda"
        )
        attack = AttackConfig(
            name="PGD", epsilon=[0.03, 0.05], alpha=[0.01, 0.02], steps=40, targeted=True, target_label=1
        )
        dataset = DatasetConfig(
            name="Cityscapes",
            root="/valid/path",
            split="train",
            mode="fine",
            target_type=["semantic", "instance"],
            image_shape=[512, 512],
            max_images=1000,
        )
        config = Config(model=model, attacks=[attack], dataset=dataset)
        assert config.model == model
        assert config.attacks == [attack]
        assert config.dataset == dataset
