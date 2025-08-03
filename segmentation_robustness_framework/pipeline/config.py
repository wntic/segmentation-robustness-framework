import json
import logging
from pathlib import Path
from typing import Any, Callable, Union

import torch
import yaml

from segmentation_robustness_framework.adapters.base_protocol import SegmentationModelProtocol
from segmentation_robustness_framework.loaders.attack_loader import AttackLoader
from segmentation_robustness_framework.loaders.dataset_loader import DatasetLoader
from segmentation_robustness_framework.loaders.models.universal_loader import UniversalModelLoader
from segmentation_robustness_framework.metrics import MetricsCollection, get_custom_metric

from .core import SegmentationRobustnessPipeline

logger = logging.getLogger(__name__)


class PipelineConfig:
    """Configuration parser and pipeline factory for segmentation robustness evaluation.

    This class loads configuration from YAML/JSON files and creates ready-to-run
    segmentation robustness pipelines.

    **Configuration File Structure:**

    ```yaml
    # Model configuration
    model:
      type: "torchvision"  # torchvision, smp, huggingface, custom
      config:
        name: "deeplabv3_resnet50"
        num_classes: 21
      weights_path: null  # optional
      weight_type: "full"  # full or encoder
      adapter: null  # optional custom adapter class

    # Dataset configuration
    dataset:
      name: "ade20k"
      root: null  # will use cache directory
      split: "val"
      image_shape: [256, 256]
      download: true

    # Attack configurations
    attacks:
      - name: "fgsm"
        eps: 0.02
      - name: "pgd"
        eps: 0.02
        alpha: 0.01
        iters: 10
        targeted: false

    # Pipeline configuration
    pipeline:
      batch_size: 8
      device: "cuda"
      output_dir: "./runs"
      auto_resize_masks: true
      output_formats: ["json", "csv"]

    # Metrics configuration
    metrics:
      ignore_index: 255
      selected_metrics:
        - "mean_iou"
        - "pixel_accuracy"
        - {"name": "dice_score", "average": "micro"}
        - "name_of_custom_metric"
    ```

    **Usage Examples:**

    ```python
    # From YAML file
    config = PipelineConfig.from_yaml("config.yaml")
    pipeline = config.create_pipeline()
    results = pipeline.run(save=True)

    # From dictionary
    config_dict = {
        "model": {
            "type": "torchvision",
            "config": {"name": "deeplabv3_resnet50", "num_classes": 21}
        },
        "dataset": {
            "name": "voc",
            "root": "./data/VOCdevkit/VOC2012",
            "split": "val",
            "image_shape": [256, 256],
            "download": false
        },
        "attacks": [{"name": "fgsm", "eps": 0.02}],
        "pipeline": {"batch_size": 4, "device": "cuda", "auto_resize_masks": true}
    }
    config = PipelineConfig.from_dict(config_dict)
    pipeline = config.create_pipeline()
    results = pipeline.run()
    ```
    """

    def __init__(self, config: dict[str, Any]):
        """Initialize configuration parser.

        Args:
            config (dict[str, Any]): Configuration dictionary.
        """
        self.config = config
        self._validate_config()

    @classmethod
    def from_yaml(cls, config_path: Union[str, Path]) -> "PipelineConfig":
        """Create configuration from YAML file.

        Args:
            config_path (str | Path): Path to YAML configuration file.

        Returns:
            PipelineConfig: Configuration instance.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            yaml.YAMLError: If YAML is malformed.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        return cls(config)

    @classmethod
    def from_json(cls, config_path: Union[str, Path]) -> "PipelineConfig":
        """Create configuration from JSON file.

        Args:
            config_path (str | Path): Path to JSON configuration file.

        Returns:
            PipelineConfig: Configuration instance.

        Raises:
            FileNotFoundError: If config file doesn't exist.
            json.JSONDecodeError: If JSON is malformed.
        """
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {config_path}")

        with open(config_path) as f:
            config = json.load(f)

        return cls(config)

    @classmethod
    def from_dict(cls, config: dict[str, Any]) -> "PipelineConfig":
        """Create configuration from dictionary.

        Args:
            config (dict[str, Any]): Configuration dictionary.

        Returns:
            PipelineConfig: Configuration instance.
        """
        return cls(config)

    def _validate_config(self) -> None:
        """Validate configuration structure and required fields.

        Raises:
            ValueError: If configuration is invalid.
        """
        required_sections = ["model", "dataset", "attacks", "pipeline"]
        for section in required_sections:
            if section not in self.config:
                raise ValueError(f"Missing required configuration section: {section}")

        model_config = self.config["model"]
        if "type" not in model_config:
            raise ValueError("Model configuration must specify 'type'")
        if "config" not in model_config:
            raise ValueError("Model configuration must specify 'config'")

        model_type = model_config["type"]
        model_config_dict = model_config["config"]

        if model_type == "torchvision":
            if "name" not in model_config_dict:
                raise ValueError("Torchvision model config must specify 'name'")
        elif model_type == "huggingface":
            if "model_name" not in model_config_dict:
                raise ValueError("HuggingFace model config must specify 'model_name'")
        elif model_type == "smp":
            if "architecture" not in model_config_dict:
                raise ValueError("SMP model config must specify 'architecture'")
            if "encoder_name" not in model_config_dict:
                raise ValueError("SMP model config must specify 'encoder_name'")

        dataset_config = self.config["dataset"]
        if "name" not in dataset_config:
            raise ValueError("Dataset configuration must specify 'name'")
        if "image_shape" not in dataset_config:
            raise ValueError("Dataset configuration must specify 'image_shape'")

        attacks_config = self.config["attacks"]
        if not isinstance(attacks_config, list) or len(attacks_config) == 0:
            raise ValueError("Attacks configuration must be a non-empty list")

        for attack_config in attacks_config:
            if "name" not in attack_config:
                raise ValueError("Each attack configuration must specify 'name'")

    def create_pipeline(self) -> SegmentationRobustnessPipeline:
        """Create and configure a segmentation robustness pipeline.

        Returns:
            SegmentationRobustnessPipeline: Configured pipeline ready to run.
        """
        logger.info("Creating pipeline from configuration...")

        model = self._load_model()
        logger.info(f"Loaded model: {type(model).__name__}")

        device = self.config["pipeline"].get("device", "cpu")
        model.to(device)
        model.eval()

        dataset = self._load_dataset()
        logger.info(f"Loaded dataset: {type(dataset).__name__}")

        attacks = self._load_attacks(model)
        logger.info(f"Loaded {len(attacks)} attack instances")

        metrics, metric_names = self._load_metrics(dataset)
        logger.info(f"Loaded {len(metrics)} metrics: {', '.join(metric_names)}")

        pipeline_config = self.config["pipeline"]

        pipeline = SegmentationRobustnessPipeline(
            model=model,
            dataset=dataset,
            attacks=attacks,
            metrics=metrics,
            batch_size=pipeline_config.get("batch_size", 8),
            device=pipeline_config.get("device", "cpu"),
            output_dir=pipeline_config.get("output_dir", "./runs"),
            auto_resize_masks=pipeline_config.get("auto_resize_masks", True),
            metric_names=metric_names,
            output_formats=pipeline_config.get("output_formats", ["json"]),
            metric_precision=pipeline_config.get("metric_precision", 4),
            num_workers=pipeline_config.get("num_workers", 0),
            pin_memory=pipeline_config.get("pin_memory", False),
            persistent_workers=pipeline_config.get("persistent_workers", False),
        )

        logger.info("Pipeline created successfully")
        return pipeline

    def _load_model(self) -> SegmentationModelProtocol:
        """Load and configure the segmentation model.

        Returns:
            SegmentationModelProtocol: Loaded and adapted model.
        """
        model_config = self.config["model"]
        loader = UniversalModelLoader()

        model = loader.load_model(
            model_type=model_config["type"],
            model_config=model_config["config"],
            weights_path=model_config.get("weights_path", None),
            weight_type=model_config.get("weight_type", "full"),
            adapter_cls=model_config.get("adapter", None),
        )

        return model

    def _load_dataset(self) -> torch.utils.data.Dataset:
        """Load and configure the dataset.

        Returns:
            torch.utils.data.Dataset: Configured dataset.
        """
        dataset_config = self.config["dataset"]
        loader = DatasetLoader(dataset_config)
        dataset = loader.load_dataset()

        return dataset

    def _load_attacks(self, model: SegmentationModelProtocol) -> list[Any]:
        """Load and configure adversarial attacks.

        Args:
            model (SegmentationModelProtocol): Model to attack.

        Returns:
            list[Any]: Flattened list of attack instances.
        """
        attacks_config = self.config["attacks"]
        loader = AttackLoader(model, attacks_config)
        attacks = loader.load_attacks()

        flat_attacks = []
        for attack_group in attacks:
            flat_attacks.extend(attack_group)

        return flat_attacks

    def _load_metrics(self, dataset: torch.utils.data.Dataset) -> tuple[list[Callable], list[str]]:
        """Load and configure evaluation metrics.

        Args:
            dataset (torch.utils.data.Dataset): Dataset to get number of classes from.

        Returns:
            tuple[list[Callable], list[str]]: List of metric functions and their names.
                The functions accept (targets, predictions) and return float values.
        """
        metrics_config = self.config.get("metrics", {})

        num_classes = getattr(dataset, "num_classes", 1)
        logger.info(f"Detected {num_classes} classes from dataset: {type(dataset).__name__}")

        ignore_index = metrics_config.get("ignore_index", 255)

        selected_metrics = metrics_config.get("selected_metrics", None)
        include_pixel_accuracy = metrics_config.get("include_pixel_accuracy", True)

        metrics_collection = MetricsCollection(
            num_classes=num_classes,
            ignore_index=ignore_index,
        )

        if selected_metrics is not None:
            metric_functions = []
            metric_names = []

            for metric_spec in selected_metrics:
                if isinstance(metric_spec, str):
                    try:
                        custom_metric = get_custom_metric(metric_spec)
                        metric_functions.append(custom_metric)
                        metric_names.append(metric_spec)
                        logger.info(f"Loaded custom metric: {metric_spec}")
                        continue
                    except KeyError:
                        pass

                    if metric_spec == "pixel_accuracy":
                        metric_functions.append(metrics_collection.pixel_accuracy)
                        metric_names.append("pixel_accuracy")
                    else:
                        metric_functions.append(metrics_collection.get_metric_with_averaging(metric_spec, "macro"))
                        metric_names.append(f"{metric_spec}_macro")
                elif isinstance(metric_spec, dict):
                    metric_name = metric_spec["name"]
                    average = metric_spec.get("average", "macro")

                    if metric_name == "pixel_accuracy":
                        metric_functions.append(metrics_collection.pixel_accuracy)
                        metric_names.append("pixel_accuracy")
                    else:
                        metric_functions.append(metrics_collection.get_metric_with_averaging(metric_name, average))
                        metric_names.append(f"{metric_name}_{average}")
                else:
                    raise ValueError(f"Invalid metric specification: {metric_spec}")
        else:
            metric_functions, metric_names = metrics_collection.get_all_metrics_with_averaging(
                include_pixel_accuracy=include_pixel_accuracy
            )

        return metric_functions, metric_names

    def run_pipeline(self, save: bool = True, show: bool = False) -> dict[str, Any]:
        """Create and run the pipeline.

        Args:
            save (bool): Whether to save results. Defaults to True.
            show (bool): Whether to show visualizations. Defaults to False.

        Returns:
            dict[str, Any]: Pipeline results.
        """
        pipeline = self.create_pipeline()
        return pipeline.run(save=save, show=show)

    def get_config_summary(self) -> dict[str, Any]:
        """Get a summary of the configuration.

        Returns:
            dict[str, Any]: Configuration summary.
        """
        return {
            "model": {
                "type": self.config["model"]["type"],
                "config": self.config["model"]["config"],
            },
            "dataset": {
                "name": self.config["dataset"]["name"],
                "split": self.config["dataset"].get("split"),
                "image_shape": self.config["dataset"]["image_shape"],
            },
            "attacks": [attack["name"] for attack in self.config["attacks"]],
            "pipeline": {
                "batch_size": self.config["pipeline"].get("batch_size", 8),
                "device": self.config["pipeline"].get("device", "cuda"),
                "output_dir": self.config["pipeline"].get("output_dir", "./runs"),
            },
        }
