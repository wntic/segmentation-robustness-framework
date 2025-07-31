import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
import torch
import yaml
from segmentation_robustness_framework.pipeline.config import PipelineConfig


class MockSegmentationModel:
    def __init__(self, num_classes: int = 21):
        self.num_classes = num_classes
        self.training = False

    def to(self, device):
        self.device = device
        return self

    def eval(self):
        self.training = False

    def logits(self, x):
        batch_size, _, height, width = x.shape
        return torch.randn(batch_size, self.num_classes, height, width)

    def predictions(self, x):
        batch_size, _, height, width = x.shape
        return torch.randint(0, self.num_classes, (batch_size, height, width))


class MockDataset:
    def __init__(self, num_classes: int = 21):
        self.num_classes = num_classes
        self.size = 100

    def __getitem__(self, idx):
        return torch.randn(3, 256, 256), torch.randint(0, self.num_classes, (256, 256))


class MockAttack:
    def __init__(self, name: str = "mock_attack"):
        self.name = name

    def __call__(self, images, targets):
        return images


@pytest.fixture
def valid_config_dict():
    return {
        "model": {
            "type": "torchvision",
            "config": {"name": "deeplabv3_resnet50", "num_classes": 21},
            "weights_path": None,
            "weight_type": "full",
        },
        "dataset": {
            "name": "voc",
            "root": "./data/VOCdevkit/VOC2012",
            "split": "val",
            "image_shape": [256, 256],
            "download": False,
        },
        "attacks": [
            {"name": "fgsm", "eps": 0.02},
            {"name": "pgd", "eps": 0.02, "alpha": 0.01, "iters": 10, "targeted": False},
        ],
        "pipeline": {
            "batch_size": 8,
            "device": "cpu",
            "output_dir": "./runs",
            "auto_resize_masks": True,
            "output_formats": ["json", "csv"],
        },
        "metrics": {
            "ignore_index": 255,
            "selected_metrics": ["mean_iou", "pixel_accuracy", {"name": "dice_score", "average": "micro"}],
        },
    }


@pytest.fixture
def minimal_config_dict():
    return {
        "model": {"type": "torchvision", "config": {"name": "deeplabv3_resnet50", "num_classes": 21}},
        "dataset": {"name": "voc", "image_shape": [256, 256]},
        "attacks": [{"name": "fgsm", "eps": 0.02}],
        "pipeline": {},
    }


class TestPipelineConfig:
    def test_initialization(self, valid_config_dict):
        config = PipelineConfig(valid_config_dict)
        assert config.config == valid_config_dict

    def test_from_dict(self, valid_config_dict):
        config = PipelineConfig.from_dict(valid_config_dict)
        assert config.config == valid_config_dict

    def test_from_yaml(self, valid_config_dict):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(valid_config_dict, f)
            yaml_path = f.name

        try:
            config = PipelineConfig.from_yaml(yaml_path)
            assert config.config == valid_config_dict
        finally:
            Path(yaml_path).unlink()

    def test_from_json(self, valid_config_dict):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(valid_config_dict, f)
            json_path = f.name

        try:
            config = PipelineConfig.from_json(json_path)
            assert config.config == valid_config_dict
        finally:
            Path(json_path).unlink()

    def test_from_yaml_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            PipelineConfig.from_yaml("nonexistent.yaml")

    def test_from_json_file_not_found(self):
        with pytest.raises(FileNotFoundError, match="Configuration file not found"):
            PipelineConfig.from_json("nonexistent.json")

    def test_validation_missing_sections(self):
        invalid_config = {
            "model": {"type": "torchvision", "config": {"name": "test"}},
            "dataset": {"name": "voc", "image_shape": [256, 256]},
        }

        with pytest.raises(ValueError, match="Missing required configuration section"):
            PipelineConfig(invalid_config)

    def test_validation_missing_model_type(self):
        invalid_config = {
            "model": {"config": {"name": "test"}},
            "dataset": {"name": "voc", "image_shape": [256, 256]},
            "attacks": [{"name": "fgsm"}],
            "pipeline": {},
        }

        with pytest.raises(ValueError, match="Model configuration must specify 'type'"):
            PipelineConfig(invalid_config)

    def test_validation_missing_model_config(self):
        invalid_config = {
            "model": {"type": "torchvision"},
            "dataset": {"name": "voc", "image_shape": [256, 256]},
            "attacks": [{"name": "fgsm"}],
            "pipeline": {},
        }

        with pytest.raises(ValueError, match="Model configuration must specify 'config'"):
            PipelineConfig(invalid_config)

    def test_validation_torchvision_missing_name(self):
        invalid_config = {
            "model": {"type": "torchvision", "config": {"num_classes": 21}},
            "dataset": {"name": "voc", "image_shape": [256, 256]},
            "attacks": [{"name": "fgsm"}],
            "pipeline": {},
        }

        with pytest.raises(ValueError, match="Torchvision model config must specify 'name'"):
            PipelineConfig(invalid_config)

    def test_validation_huggingface_missing_model_name(self):
        invalid_config = {
            "model": {"type": "huggingface", "config": {"num_classes": 21}},
            "dataset": {"name": "voc", "image_shape": [256, 256]},
            "attacks": [{"name": "fgsm"}],
            "pipeline": {},
        }

        with pytest.raises(ValueError, match="HuggingFace model config must specify 'model_name'"):
            PipelineConfig(invalid_config)

    def test_validation_smp_missing_architecture(self):
        invalid_config = {
            "model": {"type": "smp", "config": {"encoder_name": "resnet50", "num_classes": 21}},
            "dataset": {"name": "voc", "image_shape": [256, 256]},
            "attacks": [{"name": "fgsm"}],
            "pipeline": {},
        }

        with pytest.raises(ValueError, match="SMP model config must specify 'architecture'"):
            PipelineConfig(invalid_config)

    def test_validation_smp_missing_encoder_name(self):
        invalid_config = {
            "model": {"type": "smp", "config": {"architecture": "unet", "num_classes": 21}},
            "dataset": {"name": "voc", "image_shape": [256, 256]},
            "attacks": [{"name": "fgsm"}],
            "pipeline": {},
        }

        with pytest.raises(ValueError, match="SMP model config must specify 'encoder_name'"):
            PipelineConfig(invalid_config)

    def test_validation_dataset_missing_name(self):
        invalid_config = {
            "model": {"type": "torchvision", "config": {"name": "test"}},
            "dataset": {"image_shape": [256, 256]},
            "attacks": [{"name": "fgsm"}],
            "pipeline": {},
        }

        with pytest.raises(ValueError, match="Dataset configuration must specify 'name'"):
            PipelineConfig(invalid_config)

    def test_validation_dataset_missing_image_shape(self):
        invalid_config = {
            "model": {"type": "torchvision", "config": {"name": "test"}},
            "dataset": {"name": "voc"},
            "attacks": [{"name": "fgsm"}],
            "pipeline": {},
        }

        with pytest.raises(ValueError, match="Dataset configuration must specify 'image_shape'"):
            PipelineConfig(invalid_config)

    def test_validation_attacks_empty_list(self):
        invalid_config = {
            "model": {"type": "torchvision", "config": {"name": "test"}},
            "dataset": {"name": "voc", "image_shape": [256, 256]},
            "attacks": [],
            "pipeline": {},
        }

        with pytest.raises(ValueError, match="Attacks configuration must be a non-empty list"):
            PipelineConfig(invalid_config)

    def test_validation_attacks_not_list(self):
        invalid_config = {
            "model": {"type": "torchvision", "config": {"name": "test"}},
            "dataset": {"name": "voc", "image_shape": [256, 256]},
            "attacks": "not_a_list",
            "pipeline": {},
        }

        with pytest.raises(ValueError, match="Attacks configuration must be a non-empty list"):
            PipelineConfig(invalid_config)

    def test_validation_attack_missing_name(self):
        invalid_config = {
            "model": {"type": "torchvision", "config": {"name": "test"}},
            "dataset": {"name": "voc", "image_shape": [256, 256]},
            "attacks": [{"eps": 0.02}],
            "pipeline": {},
        }

        with pytest.raises(ValueError, match="Each attack configuration must specify 'name'"):
            PipelineConfig(invalid_config)

    @patch("segmentation_robustness_framework.pipeline.config.UniversalModelLoader")
    @patch("segmentation_robustness_framework.pipeline.config.DatasetLoader")
    @patch("segmentation_robustness_framework.pipeline.config.AttackLoader")
    @patch("segmentation_robustness_framework.pipeline.config.MetricsCollection")
    def test_create_pipeline(
        self, mock_metrics_collection, mock_attack_loader, mock_dataset_loader, mock_model_loader, valid_config_dict
    ):
        mock_model = MockSegmentationModel()
        mock_dataset = MockDataset()
        mock_attacks = [MockAttack("fgsm"), MockAttack("pgd")]
        mock_metrics = [lambda x, y: 0.85, lambda x, y: 0.92]

        mock_model_loader_instance = Mock()
        mock_model_loader_instance.load_model.return_value = mock_model
        mock_model_loader.return_value = mock_model_loader_instance

        mock_dataset_loader_instance = Mock()
        mock_dataset_loader_instance.load_dataset.return_value = mock_dataset
        mock_dataset_loader.return_value = mock_dataset_loader_instance

        mock_attack_loader_instance = Mock()
        mock_attack_loader_instance.load_attacks.return_value = [mock_attacks]
        mock_attack_loader.return_value = mock_attack_loader_instance

        mock_metrics_collection_instance = Mock()
        mock_metrics_collection_instance.get_metric_with_averaging.return_value = lambda x, y: 0.85
        mock_metrics_collection_instance.pixel_accuracy = lambda x, y: 0.92
        mock_metrics_collection.return_value = mock_metrics_collection_instance

        config = PipelineConfig(valid_config_dict)
        pipeline = config.create_pipeline()

        assert pipeline.model == mock_model
        assert pipeline.dataset == mock_dataset
        assert pipeline.batch_size == 8
        assert pipeline.device == "cpu"
        assert "run_" in str(pipeline.output_dir)
        assert pipeline.auto_resize_masks is True
        assert set(pipeline.output_formats) == {"json", "csv"}

    @patch("segmentation_robustness_framework.pipeline.config.UniversalModelLoader")
    @patch("segmentation_robustness_framework.pipeline.config.DatasetLoader")
    @patch("segmentation_robustness_framework.pipeline.config.AttackLoader")
    @patch("segmentation_robustness_framework.pipeline.config.MetricsCollection")
    def test_create_pipeline_minimal_config(
        self, mock_metrics_collection, mock_attack_loader, mock_dataset_loader, mock_model_loader, minimal_config_dict
    ):
        mock_model = MockSegmentationModel()
        mock_dataset = MockDataset()
        mock_attacks = [MockAttack("fgsm")]

        mock_model_loader_instance = Mock()
        mock_model_loader_instance.load_model.return_value = mock_model
        mock_model_loader.return_value = mock_model_loader_instance

        mock_dataset_loader_instance = Mock()
        mock_dataset_loader_instance.load_dataset.return_value = mock_dataset
        mock_dataset_loader.return_value = mock_dataset_loader_instance

        mock_attack_loader_instance = Mock()
        mock_attack_loader_instance.load_attacks.return_value = [mock_attacks]
        mock_attack_loader.return_value = mock_attack_loader_instance

        mock_metrics_collection_instance = Mock()
        mock_metrics_collection_instance.get_all_metrics_with_averaging.return_value = (
            [lambda x, y: 0.85],
            ["metric1"],
        )
        mock_metrics_collection.return_value = mock_metrics_collection_instance

        config = PipelineConfig(minimal_config_dict)
        pipeline = config.create_pipeline()

        assert pipeline.model == mock_model
        assert pipeline.dataset == mock_dataset
        assert pipeline.batch_size == 8
        assert pipeline.device == "cpu"
        assert "run_" in str(pipeline.output_dir)
        assert pipeline.auto_resize_masks is True

    @patch("segmentation_robustness_framework.pipeline.config.UniversalModelLoader")
    def test_load_model(self, mock_model_loader, valid_config_dict):
        mock_model = MockSegmentationModel()
        mock_loader_instance = Mock()
        mock_loader_instance.load_model.return_value = mock_model
        mock_model_loader.return_value = mock_loader_instance

        config = PipelineConfig(valid_config_dict)
        model = config._load_model()

        assert model == mock_model
        mock_loader_instance.load_model.assert_called_once_with(
            model_type="torchvision",
            model_config={"name": "deeplabv3_resnet50", "num_classes": 21},
            weights_path=None,
            weight_type="full",
            adapter_cls=None,
        )

    @patch("segmentation_robustness_framework.pipeline.config.DatasetLoader")
    def test_load_dataset(self, mock_dataset_loader, valid_config_dict):
        mock_dataset = MockDataset()
        mock_loader_instance = Mock()
        mock_loader_instance.load_dataset.return_value = mock_dataset
        mock_dataset_loader.return_value = mock_loader_instance

        config = PipelineConfig(valid_config_dict)
        dataset = config._load_dataset()

        assert dataset == mock_dataset
        mock_dataset_loader.assert_called_once_with(valid_config_dict["dataset"])

    @patch("segmentation_robustness_framework.pipeline.config.AttackLoader")
    def test_load_attacks(self, mock_attack_loader, valid_config_dict):
        mock_attacks = [MockAttack("fgsm"), MockAttack("pgd")]
        mock_loader_instance = Mock()
        mock_loader_instance.load_attacks.return_value = [mock_attacks]
        mock_attack_loader.return_value = mock_loader_instance

        mock_model = MockSegmentationModel()
        config = PipelineConfig(valid_config_dict)
        attacks = config._load_attacks(mock_model)

        assert attacks == mock_attacks
        mock_attack_loader.assert_called_once_with(mock_model, valid_config_dict["attacks"])

    @patch("segmentation_robustness_framework.pipeline.config.MetricsCollection")
    @patch("segmentation_robustness_framework.pipeline.config.get_custom_metric")
    def test_load_metrics_with_selected_metrics(
        self, mock_get_custom_metric, mock_metrics_collection, valid_config_dict
    ):
        mock_dataset = MockDataset(num_classes=21)
        mock_metrics_collection_instance = Mock()
        mock_metrics_collection_instance.get_metric_with_averaging.return_value = lambda x, y: 0.85
        mock_metrics_collection_instance.pixel_accuracy = lambda x, y: 0.92
        mock_metrics_collection.return_value = mock_metrics_collection_instance

        mock_get_custom_metric.return_value = lambda x, y: 0.95

        config = PipelineConfig(valid_config_dict)
        metrics, metric_names = config._load_metrics(mock_dataset)

        assert len(metrics) == 3
        assert len(metric_names) == 3
        assert "mean_iou" in metric_names
        assert "pixel_accuracy" in metric_names
        assert "dice_score_micro" in metric_names

    @patch("segmentation_robustness_framework.pipeline.config.MetricsCollection")
    def test_load_metrics_default(self, mock_metrics_collection, minimal_config_dict):
        mock_dataset = MockDataset(num_classes=21)
        mock_metrics_collection_instance = Mock()
        mock_metrics_collection_instance.get_all_metrics_with_averaging.return_value = (
            [lambda x, y: 0.85, lambda x, y: 0.92],
            ["metric1", "metric2"],
        )
        mock_metrics_collection.return_value = mock_metrics_collection_instance

        config = PipelineConfig(minimal_config_dict)
        metrics, metric_names = config._load_metrics(mock_dataset)

        assert len(metrics) == 2
        assert len(metric_names) == 2
        assert metric_names == ["metric1", "metric2"]

    @patch("segmentation_robustness_framework.pipeline.config.MetricsCollection")
    def test_load_metrics_custom_metric_not_found(self, mock_metrics_collection, valid_config_dict):
        mock_dataset = MockDataset(num_classes=21)
        mock_metrics_collection_instance = Mock()
        mock_metrics_collection_instance.get_metric_with_averaging.return_value = lambda x, y: 0.85
        mock_metrics_collection_instance.pixel_accuracy = lambda x, y: 0.92
        mock_metrics_collection.return_value = mock_metrics_collection_instance

        with patch("segmentation_robustness_framework.pipeline.config.get_custom_metric") as mock_get_custom_metric:
            mock_get_custom_metric.side_effect = KeyError("Metric not found")

            config = PipelineConfig(valid_config_dict)
            metrics, metric_names = config._load_metrics(mock_dataset)

            assert len(metrics) == 3
            assert "mean_iou_macro" in metric_names

    def test_load_metrics_invalid_metric_spec(self, valid_config_dict):
        mock_dataset = MockDataset(num_classes=21)
        config_dict = valid_config_dict.copy()
        config_dict["metrics"]["selected_metrics"] = [123]

        config = PipelineConfig(config_dict)
        with pytest.raises(ValueError, match="Invalid metric specification"):
            config._load_metrics(mock_dataset)

    @patch("segmentation_robustness_framework.pipeline.config.PipelineConfig.create_pipeline")
    def test_run_pipeline(self, mock_create_pipeline, valid_config_dict):
        mock_pipeline = Mock()
        mock_pipeline.run.return_value = {"clean": {"metric1": 0.85}}
        mock_create_pipeline.return_value = mock_pipeline

        config = PipelineConfig(valid_config_dict)
        results = config.run_pipeline(save=True, show=False)

        assert results == {"clean": {"metric1": 0.85}}
        mock_create_pipeline.assert_called_once()
        mock_pipeline.run.assert_called_once_with(save=True, show=False)

    def test_get_config_summary(self, valid_config_dict):
        config = PipelineConfig(valid_config_dict)
        summary = config.get_config_summary()

        expected_summary = {
            "model": {
                "type": "torchvision",
                "config": {"name": "deeplabv3_resnet50", "num_classes": 21},
            },
            "dataset": {
                "name": "voc",
                "split": "val",
                "image_shape": [256, 256],
            },
            "attacks": ["fgsm", "pgd"],
            "pipeline": {
                "batch_size": 8,
                "device": "cpu",
                "output_dir": "./runs",
            },
        }

        assert summary == expected_summary

    def test_get_config_summary_minimal(self, minimal_config_dict):
        config = PipelineConfig(minimal_config_dict)
        summary = config.get_config_summary()

        expected_summary = {
            "model": {
                "type": "torchvision",
                "config": {"name": "deeplabv3_resnet50", "num_classes": 21},
            },
            "dataset": {
                "name": "voc",
                "split": None,
                "image_shape": [256, 256],
            },
            "attacks": ["fgsm"],
            "pipeline": {
                "batch_size": 8,
                "device": "cuda",
                "output_dir": "./runs",
            },
        }

        assert summary == expected_summary

    def test_yaml_malformed(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            f.write("invalid: yaml: content: [")
            yaml_path = f.name

        try:
            with pytest.raises(yaml.YAMLError):
                PipelineConfig.from_yaml(yaml_path)
        finally:
            Path(yaml_path).unlink()

    def test_json_malformed(self):
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write('{"invalid": json}')
            json_path = f.name

        try:
            with pytest.raises(json.JSONDecodeError):
                PipelineConfig.from_json(json_path)
        finally:
            Path(json_path).unlink()

    def test_metrics_with_pixel_accuracy_string(self, valid_config_dict):
        config_dict = valid_config_dict.copy()
        config_dict["metrics"]["selected_metrics"] = ["pixel_accuracy"]

        mock_dataset = MockDataset(num_classes=21)
        with patch("segmentation_robustness_framework.pipeline.config.MetricsCollection") as mock_metrics_collection:
            mock_metrics_collection_instance = Mock()
            mock_metrics_collection_instance.pixel_accuracy = lambda x, y: 0.92
            mock_metrics_collection.return_value = mock_metrics_collection_instance

            config = PipelineConfig(config_dict)
            metrics, metric_names = config._load_metrics(mock_dataset)

            assert len(metrics) == 1
            assert metric_names == ["pixel_accuracy"]

    def test_metrics_with_pixel_accuracy_dict(self, valid_config_dict):
        config_dict = valid_config_dict.copy()
        config_dict["metrics"]["selected_metrics"] = [{"name": "pixel_accuracy"}]

        mock_dataset = MockDataset(num_classes=21)
        with patch("segmentation_robustness_framework.pipeline.config.MetricsCollection") as mock_metrics_collection:
            mock_metrics_collection_instance = Mock()
            mock_metrics_collection_instance.pixel_accuracy = lambda x, y: 0.92
            mock_metrics_collection.return_value = mock_metrics_collection_instance

            config = PipelineConfig(config_dict)
            metrics, metric_names = config._load_metrics(mock_dataset)

            assert len(metrics) == 1
            assert metric_names == ["pixel_accuracy"]

    def test_metrics_with_custom_metric(self, valid_config_dict):
        config_dict = valid_config_dict.copy()
        config_dict["metrics"]["selected_metrics"] = ["custom_metric"]

        mock_dataset = MockDataset(num_classes=21)
        with patch("segmentation_robustness_framework.pipeline.config.MetricsCollection") as mock_metrics_collection:
            mock_metrics_collection_instance = Mock()
            mock_metrics_collection.return_value = mock_metrics_collection_instance

            with patch("segmentation_robustness_framework.pipeline.config.get_custom_metric") as mock_get_custom_metric:
                mock_get_custom_metric.return_value = lambda x, y: 0.95

                config = PipelineConfig(config_dict)
                metrics, metric_names = config._load_metrics(mock_dataset)

                assert len(metrics) == 1
                assert metric_names == ["custom_metric"]

    def test_dataset_without_num_classes(self, minimal_config_dict):
        mock_dataset = Mock()
        del mock_dataset.num_classes

        with patch("segmentation_robustness_framework.pipeline.config.MetricsCollection") as mock_metrics_collection:
            mock_metrics_collection_instance = Mock()
            mock_metrics_collection_instance.get_all_metrics_with_averaging.return_value = (
                [lambda x, y: 0.85],
                ["metric1"],
            )
            mock_metrics_collection.return_value = mock_metrics_collection_instance

            config = PipelineConfig(minimal_config_dict)
            metrics, metric_names = config._load_metrics(mock_dataset)

            assert len(metrics) == 1
            assert metric_names == ["metric1"]

    def test_attacks_flattening(self, valid_config_dict):
        with patch("segmentation_robustness_framework.pipeline.config.AttackLoader") as mock_attack_loader:
            mock_attacks_group1 = [MockAttack("fgsm")]
            mock_attacks_group2 = [MockAttack("pgd"), MockAttack("cw")]
            mock_loader_instance = Mock()
            mock_loader_instance.load_attacks.return_value = [mock_attacks_group1, mock_attacks_group2]
            mock_attack_loader.return_value = mock_loader_instance

            mock_model = MockSegmentationModel()
            config = PipelineConfig(valid_config_dict)
            attacks = config._load_attacks(mock_model)

            assert len(attacks) == 3
            assert attacks[0].name == "fgsm"
            assert attacks[1].name == "pgd"
            assert attacks[2].name == "cw"
