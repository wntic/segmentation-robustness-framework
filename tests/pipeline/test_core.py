import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pandas as pd
import pytest
import torch
from segmentation_robustness_framework.pipeline.core import SegmentationRobustnessPipeline
from torch.utils.data import Dataset


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


class MockDataset(Dataset):
    def __init__(self, size: int = 10, image_shape: tuple = (3, 256, 256), num_classes: int = 21):
        self.size = size
        self.image_shape = image_shape
        self.num_classes = num_classes
        self.target_transform = None

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        image = torch.randn(*self.image_shape)
        target = torch.randint(0, self.num_classes, (self.image_shape[1], self.image_shape[2]))
        return image, target


class MockAttack:
    def __init__(self, name: str = "mock_attack", eps: float = 0.02):
        self.name = name
        self.eps = eps

    def __call__(self, images, targets):
        return images + torch.randn_like(images) * self.eps

    def get_params(self):
        return {"eps": self.eps, "name": self.name}


def mock_metric(targets, preds):
    return 0.85


def mock_metric_with_params(targets, preds):
    return 0.92


@pytest.fixture
def mock_model():
    return MockSegmentationModel(num_classes=21)


@pytest.fixture
def mock_dataset():
    return MockDataset(size=10, image_shape=(3, 256, 256), num_classes=21)


@pytest.fixture
def mock_attacks():
    return [
        MockAttack("fgsm", eps=0.02),
        MockAttack("pgd", eps=0.03),
    ]


@pytest.fixture
def mock_metrics():
    return [mock_metric, mock_metric_with_params]


@pytest.fixture
def temp_output_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        yield temp_dir


class TestSegmentationRobustnessPipeline:
    def test_initialization(self, mock_model, mock_dataset, mock_attacks, mock_metrics, temp_output_dir):
        pipeline = SegmentationRobustnessPipeline(
            model=mock_model,
            dataset=mock_dataset,
            attacks=mock_attacks,
            metrics=mock_metrics,
            batch_size=4,
            device="cpu",
            output_dir=temp_output_dir,
            auto_resize_masks=False,
        )

        assert pipeline.model == mock_model
        assert pipeline.dataset == mock_dataset
        assert pipeline.attacks == mock_attacks
        assert pipeline.metrics == mock_metrics
        assert pipeline.batch_size == 4
        assert pipeline.device == "cpu"
        assert pipeline.auto_resize_masks is False
        assert pipeline.metric_precision == 4
        assert pipeline.num_workers == 0
        assert pipeline.pin_memory is False
        assert pipeline.persistent_workers is False

    def test_initialization_defaults(self, mock_model, mock_dataset, mock_attacks, mock_metrics):
        pipeline = SegmentationRobustnessPipeline(
            model=mock_model,
            dataset=mock_dataset,
            attacks=mock_attacks,
            metrics=mock_metrics,
        )

        assert pipeline.batch_size == 8
        assert pipeline.device == "cpu"
        assert pipeline.base_output_dir == "./runs"
        assert pipeline.auto_resize_masks is True
        assert set(pipeline.output_formats) == {"json", "csv"}

    def test_run_id_generation(self, mock_model, mock_dataset, mock_attacks, mock_metrics):
        pipeline = SegmentationRobustnessPipeline(
            model=mock_model,
            dataset=mock_dataset,
            attacks=mock_attacks,
            metrics=mock_metrics,
        )

        run_id = pipeline.run_id
        assert isinstance(run_id, str)
        assert len(run_id) > 0
        assert "_" in run_id

    def test_attack_name_generation(self, mock_model, mock_dataset, mock_attacks, mock_metrics):
        pipeline = SegmentationRobustnessPipeline(
            model=mock_model,
            dataset=mock_dataset,
            attacks=mock_attacks,
            metrics=mock_metrics,
        )

        attack_name = pipeline._generate_attack_name(mock_attacks[0])
        assert "MockAttack" in attack_name
        assert "eps_0p020" in attack_name

        simple_attack = Mock()
        simple_attack.__class__.__name__ = "SimpleAttack"
        simple_attack.eps = 0.01
        attack_name = pipeline._generate_attack_name(simple_attack)
        assert "SimpleAttack" in attack_name
        assert "eps_0p010" in attack_name

    def test_metric_names_setup(self, mock_model, mock_dataset, mock_attacks, mock_metrics):
        pipeline = SegmentationRobustnessPipeline(
            model=mock_model,
            dataset=mock_dataset,
            attacks=mock_attacks,
            metrics=mock_metrics,
        )

        assert len(pipeline.metric_names) == 2
        assert "mock_metric" in pipeline.metric_names[0]
        assert "mock_metric_with_params" in pipeline.metric_names[1]

        custom_names = ["metric1", "metric2"]
        pipeline = SegmentationRobustnessPipeline(
            model=mock_model,
            dataset=mock_dataset,
            attacks=mock_attacks,
            metrics=mock_metrics,
            metric_names=custom_names,
        )
        assert pipeline.metric_names == custom_names

    def test_metric_names_mismatch(self, mock_model, mock_dataset, mock_attacks, mock_metrics):
        with pytest.raises(ValueError, match="Number of metric names"):
            SegmentationRobustnessPipeline(
                model=mock_model,
                dataset=mock_dataset,
                attacks=mock_attacks,
                metrics=mock_metrics,
                metric_names=["metric1"],
            )

    def test_output_formats_setup(self, mock_model, mock_dataset, mock_attacks, mock_metrics):
        pipeline = SegmentationRobustnessPipeline(
            model=mock_model,
            dataset=mock_dataset,
            attacks=mock_attacks,
            metrics=mock_metrics,
            output_formats=["json", "csv"],
        )
        assert set(pipeline.output_formats) == {"json", "csv"}

        with pytest.raises(ValueError, match="Invalid output format"):
            SegmentationRobustnessPipeline(
                model=mock_model,
                dataset=mock_dataset,
                attacks=mock_attacks,
                metrics=mock_metrics,
                output_formats=["invalid_format"],
            )

        pipeline = SegmentationRobustnessPipeline(
            model=mock_model,
            dataset=mock_dataset,
            attacks=mock_attacks,
            metrics=mock_metrics,
            output_formats=[],
        )
        assert pipeline.output_formats == ["json"]

    @patch("segmentation_robustness_framework.pipeline.core.get_model_output_size")
    @patch("segmentation_robustness_framework.pipeline.core.image_preprocessing")
    def test_auto_resize_masks_setup(
        self, mock_image_preprocessing, mock_get_output_size, mock_model, mock_dataset, mock_attacks, mock_metrics
    ):
        mock_get_output_size.return_value = (128, 128)
        mock_image_preprocessing.get_preprocessing_fn.return_value = (None, lambda x: x)

        pipeline = SegmentationRobustnessPipeline(
            model=mock_model,
            dataset=mock_dataset,
            attacks=mock_attacks,
            metrics=mock_metrics,
            auto_resize_masks=True,
        )

        mock_get_output_size.assert_called_once()
        mock_image_preprocessing.get_preprocessing_fn.assert_called_once()

    def test_dataset_name_detection(self, mock_model, mock_dataset, mock_attacks, mock_metrics):
        pipeline = SegmentationRobustnessPipeline(
            model=mock_model,
            dataset=mock_dataset,
            attacks=mock_attacks,
            metrics=mock_metrics,
        )

        mock_dataset.name = "voc"
        dataset_name = pipeline._detect_dataset_name()
        assert dataset_name == "voc"

        del mock_dataset.name
        mock_dataset.dataset_name = "ade20k"
        dataset_name = pipeline._detect_dataset_name()
        assert dataset_name == "ade20k"

        del mock_dataset.dataset_name
        mock_dataset._name = "cityscapes"
        dataset_name = pipeline._detect_dataset_name()
        assert dataset_name == "cityscapes"

        del mock_dataset._name
        dataset_name = pipeline._detect_dataset_name()
        assert dataset_name is None

    @patch("segmentation_robustness_framework.pipeline.core.tqdm")
    def test_evaluate_clean(self, mock_tqdm, mock_model, mock_dataset, mock_attacks, mock_metrics, temp_output_dir):
        mock_tqdm.return_value = [(torch.randn(2, 3, 256, 256), torch.randint(0, 21, (2, 256, 256)))]

        pipeline = SegmentationRobustnessPipeline(
            model=mock_model,
            dataset=mock_dataset,
            attacks=mock_attacks,
            metrics=mock_metrics,
            output_dir=temp_output_dir,
        )

        results = pipeline.evaluate_clean(Mock())
        assert isinstance(results, list)
        assert len(results) == 1

    @patch("segmentation_robustness_framework.pipeline.core.tqdm")
    def test_evaluate_attack(self, mock_tqdm, mock_model, mock_dataset, mock_attacks, mock_metrics, temp_output_dir):
        mock_tqdm.return_value = [(torch.randn(2, 3, 256, 256), torch.randint(0, 21, (2, 256, 256)))]

        pipeline = SegmentationRobustnessPipeline(
            model=mock_model,
            dataset=mock_dataset,
            attacks=mock_attacks,
            metrics=mock_metrics,
            output_dir=temp_output_dir,
        )

        attack = mock_attacks[0]
        results = pipeline.evaluate_attack(Mock(), attack)
        assert isinstance(results, list)
        assert len(results) == 1

    def test_compute_metrics(self, mock_model, mock_dataset, mock_attacks, mock_metrics):
        pipeline = SegmentationRobustnessPipeline(
            model=mock_model,
            dataset=mock_dataset,
            attacks=mock_attacks,
            metrics=mock_metrics,
        )

        targets = torch.randint(0, 21, (2, 256, 256))
        preds = torch.randint(0, 21, (2, 256, 256))

        results = pipeline.compute_metrics(targets, preds)
        assert isinstance(results, dict)
        assert len(results) == 2
        assert all(isinstance(v, (float, type(None))) for v in results.values())

    def test_compute_metrics_with_error(self, mock_model, mock_dataset, mock_attacks):
        def failing_metric(targets, preds):
            raise ValueError("Test error")

        metrics = [failing_metric]
        pipeline = SegmentationRobustnessPipeline(
            model=mock_model,
            dataset=mock_dataset,
            attacks=mock_attacks,
            metrics=metrics,
        )

        targets = torch.randint(0, 21, (2, 256, 256))
        preds = torch.randint(0, 21, (2, 256, 256))

        results = pipeline.compute_metrics(targets, preds)
        assert isinstance(results, dict)
        assert len(results) == 1
        assert results["failing_metric"] is None

    def test_aggregate_metrics(self, mock_model, mock_dataset, mock_attacks, mock_metrics):
        pipeline = SegmentationRobustnessPipeline(
            model=mock_model,
            dataset=mock_dataset,
            attacks=mock_attacks,
            metrics=mock_metrics,
        )

        batch_metrics = [
            {"metric1": 0.8, "metric2": 0.9},
            {"metric1": 0.9, "metric2": 0.85},
            {"metric1": 0.85, "metric2": 0.95},
        ]

        aggregated = pipeline._aggregate_metrics(batch_metrics)
        assert isinstance(aggregated, dict)
        assert len(aggregated) == 2
        assert "metric1" in aggregated
        assert "metric2" in aggregated
        assert aggregated["metric1"] == pytest.approx(0.85, abs=1e-2)
        assert aggregated["metric2"] == pytest.approx(0.9, abs=1e-2)

    def test_aggregate_metrics_with_none_values(self, mock_model, mock_dataset, mock_attacks, mock_metrics):
        pipeline = SegmentationRobustnessPipeline(
            model=mock_model,
            dataset=mock_dataset,
            attacks=mock_attacks,
            metrics=mock_metrics,
        )

        batch_metrics = [
            {"metric1": 0.8, "metric2": None},
            {"metric1": None, "metric2": 0.85},
            {"metric1": 0.85, "metric2": 0.95},
        ]

        aggregated = pipeline._aggregate_metrics(batch_metrics)
        assert aggregated["metric1"] == pytest.approx(0.825, abs=1e-2)
        assert aggregated["metric2"] == pytest.approx(0.9, abs=1e-2)

    def test_aggregate_metrics_empty(self, mock_model, mock_dataset, mock_attacks, mock_metrics):
        pipeline = SegmentationRobustnessPipeline(
            model=mock_model,
            dataset=mock_dataset,
            attacks=mock_attacks,
            metrics=mock_metrics,
        )

        aggregated = pipeline._aggregate_metrics([])
        assert aggregated == {}

    def test_save_results(self, mock_model, mock_dataset, mock_attacks, mock_metrics, temp_output_dir):
        pipeline = SegmentationRobustnessPipeline(
            model=mock_model,
            dataset=mock_dataset,
            attacks=mock_attacks,
            metrics=mock_metrics,
            output_dir=temp_output_dir,
            output_formats=["json", "csv"],
        )

        batch_metrics = [
            {"metric1": 0.8, "metric2": 0.9},
            {"metric1": 0.9, "metric2": 0.85},
        ]

        pipeline.save_results(batch_metrics, "test_evaluation")

        json_file = pipeline.output_dir / "test_evaluation_detailed.json"
        csv_file = pipeline.output_dir / "test_evaluation_detailed.csv"

        assert json_file.exists()
        assert csv_file.exists()

        with open(json_file) as f:
            data = json.load(f)
        assert isinstance(data, list)
        assert len(data) == 2

        df = pd.read_csv(csv_file)
        assert len(df) == 2
        assert "metric1" in df.columns
        assert "metric2" in df.columns

    def test_save_results_json_only(self, mock_model, mock_dataset, mock_attacks, mock_metrics, temp_output_dir):
        pipeline = SegmentationRobustnessPipeline(
            model=mock_model,
            dataset=mock_dataset,
            attacks=mock_attacks,
            metrics=mock_metrics,
            output_dir=temp_output_dir,
            output_formats=["json"],
        )

        batch_metrics = [{"metric1": 0.8}]
        pipeline.save_results(batch_metrics, "test_evaluation")

        json_file = pipeline.output_dir / "test_evaluation_detailed.json"
        csv_file = pipeline.output_dir / "test_evaluation_detailed.csv"

        assert json_file.exists()
        assert not csv_file.exists()

    def test_save_summary_results(self, mock_model, mock_dataset, mock_attacks, mock_metrics, temp_output_dir):
        pipeline = SegmentationRobustnessPipeline(
            model=mock_model,
            dataset=mock_dataset,
            attacks=mock_attacks,
            metrics=mock_metrics,
            output_dir=temp_output_dir,
            output_formats=["json", "csv"],
        )

        pipeline.results = {
            "clean": {"metric1": 0.85, "metric2": 0.9},
            "attack_fgsm": {"metric1": 0.7, "metric2": 0.8},
        }

        pipeline._save_summary_results()

        summary_file = pipeline.output_dir / "summary_results.json"
        comparison_file = pipeline.output_dir / "comparison_table.csv"

        assert summary_file.exists()
        assert comparison_file.exists()

        with open(summary_file) as f:
            data = json.load(f)
        assert "clean" in data
        assert "attack_fgsm" in data

        df = pd.read_csv(comparison_file)
        assert len(df) == 2
        assert "evaluation" in df.columns
        assert "metric1" in df.columns
        assert "metric2" in df.columns

    def test_get_summary(self, mock_model, mock_dataset, mock_attacks, mock_metrics):
        pipeline = SegmentationRobustnessPipeline(
            model=mock_model,
            dataset=mock_dataset,
            attacks=mock_attacks,
            metrics=mock_metrics,
        )

        summary = pipeline.get_summary()
        assert "error" in summary

        pipeline.results = {
            "clean": {"metric1": 0.85, "metric2": 0.9},
            "attack_fgsm": {"metric1": 0.7, "metric2": 0.8},
        }

        summary = pipeline.get_summary()
        assert "total_evaluations" in summary
        assert "evaluations" in summary
        assert "clean_performance" in summary
        assert "attack_performance" in summary
        assert "robustness_analysis" in summary

        robustness = summary["robustness_analysis"]["attack_fgsm"]
        assert "metric1_degradation" in robustness
        assert "metric2_degradation" in robustness

    def test_get_run_info(self, mock_model, mock_dataset, mock_attacks, mock_metrics):
        pipeline = SegmentationRobustnessPipeline(
            model=mock_model,
            dataset=mock_dataset,
            attacks=mock_attacks,
            metrics=mock_metrics,
        )

        run_info = pipeline.get_run_info()
        assert "run_id" in run_info
        assert "output_directory" in run_info
        assert "device" in run_info
        assert "batch_size" in run_info
        assert "auto_resize_masks" in run_info
        assert "model_num_classes" in run_info
        assert "dataset_size" in run_info
        assert "num_attacks" in run_info
        assert "num_metrics" in run_info
        assert "output_formats" in run_info

    @patch("segmentation_robustness_framework.pipeline.core.plt")
    def test_create_visualizations(
        self, mock_plt, mock_model, mock_dataset, mock_attacks, mock_metrics, temp_output_dir
    ):
        pipeline = SegmentationRobustnessPipeline(
            model=mock_model,
            dataset=mock_dataset,
            attacks=mock_attacks,
            metrics=mock_metrics,
            output_dir=temp_output_dir,
        )

        pipeline.results = {
            "clean": {"metric1": 0.85, "metric2": 0.9},
            "attack_fgsm": {"metric1": 0.7, "metric2": 0.8},
        }

        pipeline._create_visualizations()

        metric1_file = Path(temp_output_dir) / "metric1_comparison.png"
        metric2_file = Path(temp_output_dir) / "metric2_comparison.png"
        heatmap_file = Path(temp_output_dir) / "performance_heatmap.png"

        assert mock_plt.figure.called
        assert mock_plt.savefig.called

    def test_print_summary(self, mock_model, mock_dataset, mock_attacks, mock_metrics, capsys):
        pipeline = SegmentationRobustnessPipeline(
            model=mock_model,
            dataset=mock_dataset,
            attacks=mock_attacks,
            metrics=mock_metrics,
        )

        pipeline.print_summary()
        captured = capsys.readouterr()
        assert "Error: No results available" in captured.out

        pipeline.results = {
            "clean": {"metric1": 0.85, "metric2": 0.9},
            "attack_fgsm": {"metric1": 0.7, "metric2": 0.8},
        }

        pipeline.print_summary()
        captured = capsys.readouterr()
        assert "SEGMENTATION ROBUSTNESS EVALUATION SUMMARY" in captured.out
        assert "CLEAN PERFORMANCE" in captured.out
        assert "ROBUSTNESS ANALYSIS" in captured.out

    @patch("segmentation_robustness_framework.pipeline.core.tqdm")
    def test_run_pipeline(self, mock_tqdm, mock_model, mock_dataset, mock_attacks, mock_metrics, temp_output_dir):
        mock_tqdm.return_value = [(torch.randn(2, 3, 256, 256), torch.randint(0, 21, (2, 256, 256)))]

        pipeline = SegmentationRobustnessPipeline(
            model=mock_model,
            dataset=mock_dataset,
            attacks=mock_attacks,
            metrics=mock_metrics,
            output_dir=temp_output_dir,
        )

        results = pipeline.run(save=True, show=False)

        assert isinstance(results, dict)
        assert "clean" in results
        assert len([k for k in results.keys() if k.startswith("attack_")]) == 2

        assert pipeline.output_dir.exists()

    def test_target_clamping(self, mock_model, mock_dataset, mock_attacks, mock_metrics):
        pipeline = SegmentationRobustnessPipeline(
            model=mock_model,
            dataset=mock_dataset,
            attacks=mock_attacks,
            metrics=mock_metrics,
        )

        targets = torch.randint(0, 30, (2, 256, 256))
        preds = torch.randint(0, 21, (2, 256, 256))

        results = pipeline.compute_metrics(targets, preds)
        assert isinstance(results, dict)

    def test_ignore_index_handling(self, mock_model, mock_dataset, mock_attacks, mock_metrics):
        pipeline = SegmentationRobustnessPipeline(
            model=mock_model,
            dataset=mock_dataset,
            attacks=mock_attacks,
            metrics=mock_metrics,
        )

        targets = torch.randint(0, 21, (2, 256, 256))
        targets[0, 0, 0] = -1
        preds = torch.randint(0, 21, (2, 256, 256))

        results = pipeline.compute_metrics(targets, preds)
        assert isinstance(results, dict)

    def test_metric_precision(self, mock_model, mock_dataset, mock_attacks, mock_metrics):
        pipeline = SegmentationRobustnessPipeline(
            model=mock_model,
            dataset=mock_dataset,
            attacks=mock_attacks,
            metrics=mock_metrics,
            metric_precision=2,
        )

        targets = torch.randint(0, 21, (2, 256, 256))
        preds = torch.randint(0, 21, (2, 256, 256))

        results = pipeline.compute_metrics(targets, preds)
        for value in results.values():
            if value is not None:
                decimal_str = str(value).split(".")[-1]
                assert len(decimal_str) <= 2

    def test_device_handling(self, mock_model, mock_dataset, mock_attacks, mock_metrics):
        pipeline = SegmentationRobustnessPipeline(
            model=mock_model,
            dataset=mock_dataset,
            attacks=mock_attacks,
            metrics=mock_metrics,
            device="cpu",
        )

        assert pipeline.device == "cpu"
        assert mock_model.device == "cpu"

    def test_output_directory_creation(self, mock_model, mock_dataset, mock_attacks, mock_metrics, temp_output_dir):
        pipeline = SegmentationRobustnessPipeline(
            model=mock_model,
            dataset=mock_dataset,
            attacks=mock_attacks,
            metrics=mock_metrics,
            output_dir=temp_output_dir,
        )

        assert pipeline.output_dir.exists()
        assert pipeline.output_dir.is_dir()

    def test_attack_with_different_device(self, mock_model, mock_dataset, mock_attacks, mock_metrics):
        pipeline = SegmentationRobustnessPipeline(
            model=mock_model,
            dataset=mock_dataset,
            attacks=mock_attacks,
            metrics=mock_metrics,
            device="cpu",
        )

        class DeviceMismatchAttack:
            def __call__(self, images, targets):
                return images.to("cpu") if images.device != "cpu" else images

        attack = DeviceMismatchAttack()

        images = torch.randn(2, 3, 256, 256)
        targets = torch.randint(0, 21, (2, 256, 256))
        adv_images = attack(images, targets)

        assert adv_images.device == images.device
