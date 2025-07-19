import json
import logging
import os
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from segmentation_robustness_framework.adapters.base_protocol import SegmentationModelProtocol
from segmentation_robustness_framework.utils import image_preprocessing
from segmentation_robustness_framework.utils.model_utils import get_model_output_size

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

STOP_AFTER_N_BATCHES = 5


class SegmentationRobustnessPipeline:
    """Pipeline for evaluating segmentation models under adversarial attacks.

    This pipeline evaluates a segmentation model on clean and adversarial images,
    computes metrics, and provides hooks for saving results.

    Attributes:
        model (SegmentationModelProtocol): Adapter-wrapped segmentation model.
        dataset (torch.utils.data.Dataset): Dataset for evaluation.
        attacks (list): List of attack instances (must implement __call__(images, targets)).
        metrics (list[Callable]): List of metric functions/classes (accepting (targets, preds)).
        batch_size (int): Batch size for evaluation.
        device (str): Device to use for computation.
        output_dir (str): Directory to save results.
        auto_resize_masks (bool): Whether to automatically resize masks to model output size.
    """

    def __init__(
        self,
        model: SegmentationModelProtocol,
        dataset: torch.utils.data.Dataset,
        attacks: list,
        metrics: list[Callable],
        batch_size: int = 8,
        device: str = "cuda",
        output_dir: Optional[str] = None,
        auto_resize_masks: bool = True,
        metric_names: Optional[list[str]] = None,
    ):
        """Initialize the pipeline.

        Args:
            model (SegmentationModelProtocol): Segmentation model (adapter-wrapped).
            dataset (Dataset): Dataset object.
            attacks (list): List of attack instances.
            metrics (list[Callable]): List of metric functions or classes.
            batch_size (int): Batch size for evaluation.
            device (str): Device to use.
            output_dir (str, optional): Directory to save results.
            auto_resize_masks (bool): Whether to automatically resize masks to model output size.
        """
        self.model = model.to(device)
        self.dataset = dataset
        self.attacks = attacks
        self.metrics = metrics
        self.batch_size = batch_size
        self.device = device
        self.base_output_dir = output_dir or "./runs"
        self.auto_resize_masks = auto_resize_masks

        self.metric_names = self._setup_metric_names(metric_names)

        self.run_id = self._generate_run_id()
        self.output_dir = Path(self.base_output_dir) / f"run_{self.run_id}"
        os.makedirs(self.output_dir, exist_ok=True)

        logger.info(f"Pipeline initialized. Run ID: {self.run_id}")
        logger.info(f"Output directory: {self.output_dir}")

        if self.auto_resize_masks:
            self._setup_automatic_mask_resizing()

        self.clean_counter = 0
        self.adv_counter = 0
        self.results = {}

    def _generate_run_id(self) -> str:
        """Generate a unique run ID for this evaluation.

        Returns:
            str: Unique run identifier combining timestamp and UUID.
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"{timestamp}_{unique_id}"

    def _setup_metric_names(self, metric_names: Optional[list[str]] = None) -> list[str]:
        """Setup metric names for proper identification.

        Args:
            metric_names (Optional[list[str]]): Custom metric names. If None, auto-generate.

        Returns:
            list[str]: List of metric names corresponding to self.metrics.
        """
        if metric_names is not None:
            if len(metric_names) != len(self.metrics):
                raise ValueError(
                    f"Number of metric names ({len(metric_names)}) must match number of metrics ({len(self.metrics)})"
                )
            return metric_names

        names = []
        for i, metric in enumerate(self.metrics):
            if hasattr(metric, "__name__") and metric.__name__ != "<lambda>":
                names.append(metric.__name__)
            elif hasattr(metric, "__class__") and hasattr(metric.__class__, "__name__"):
                class_name = metric.__class__.__name__
                if class_name != "function":
                    names.append(class_name)
                else:
                    names.append(f"metric_{i}")
            else:
                names.append(f"metric_{i}")

        return names

    def _setup_automatic_mask_resizing(self) -> None:
        """Automatically detect model output size and update dataset mask transforms."""
        try:
            sample_image, _ = self.dataset[0]
            if hasattr(sample_image, "shape"):
                input_shape = sample_image.shape  # [C, H, W]
            else:
                input_shape = (3, 512, 512)
                logger.warning("Could not determine input shape from dataset, using default (3, 512, 512)")

            output_size = get_model_output_size(self.model, input_shape, self.device)
            self._update_dataset_mask_transform(output_size)
            logger.info(
                f"Automatically configured mask resizing to model output size: {output_size[0]}x{output_size[1]}"
            )

        except Exception as e:
            logger.warning(f"Failed to setup automatic mask resizing: {e}. Using original dataset transforms.")
            self.auto_resize_masks = False

    def _update_dataset_mask_transform(self, output_size: tuple[int, int]) -> None:
        """Update the dataset's target transform to resize masks to model output size."""
        try:
            dataset_name = self._detect_dataset_name()

            _, new_target_transform = image_preprocessing.get_preprocessing_fn(
                list(output_size), dataset_name=dataset_name
            )

            self.dataset.target_transform = new_target_transform
            logger.info(
                f"Updated dataset mask transform to resize to {output_size[0]}x{output_size[1]} (dataset: {dataset_name or 'unknown'})"
            )

        except Exception as e:
            logger.error(f"Failed to update dataset mask transform: {e}")
            raise

    def _detect_dataset_name(self) -> Optional[str]:
        """Detect dataset name using multiple strategies for color mapping.

        Returns:
            Optional[str]: Dataset name for color mapping, or None if not detected.
        """
        # Strategy 1: Check if dataset has a name attribute
        if hasattr(self.dataset, "name"):
            return getattr(self.dataset, "name")

        # Strategy 2: Check if dataset has a dataset_name attribute
        if hasattr(self.dataset, "dataset_name"):
            return getattr(self.dataset, "dataset_name")

        # Strategy 3: Check if dataset has a _name attribute (private)
        if hasattr(self.dataset, "_name"):
            return getattr(self.dataset, "_name")

        # Strategy 4: Try to infer from class name (fallback for built-in datasets)
        if hasattr(self.dataset, "__class__"):
            class_name = self.dataset.__class__.__name__.lower()
            if "voc" in class_name:
                return "voc"
            elif "ade20k" in class_name or "ade" in class_name:
                return "ade20k"
            elif "stanford" in class_name:
                return "stanford_background"
            elif "cityscapes" in class_name:
                return "cityscapes"

        # Strategy 5: Check if dataset has a registered name in the registry
        try:
            from segmentation_robustness_framework.datasets.registry import DATASET_REGISTRY

            for name, cls in DATASET_REGISTRY.items():
                if isinstance(self.dataset, cls):
                    return name
        except ImportError:
            pass

        # Strategy 6: Check if dataset has a root path that might indicate the dataset
        if hasattr(self.dataset, "root"):
            root = str(self.dataset.root).lower()
            if "voc" in root:
                return "voc"
            elif "ade20k" in root or "ade" in root:
                return "ade20k"
            elif "stanford" in root:
                return "stanford_background"
            elif "cityscapes" in root:
                return "cityscapes"

        logger.debug(
            f"Could not detect dataset name for {self.dataset.__class__.__name__}, using generic mask processing"
        )
        return None

    def run(self, save: bool = False, show: bool = False) -> dict[str, Any]:
        """Run the evaluation pipeline: clean and adversarial evaluation.

        Args:
            save (bool): Whether to save results (images, metrics, etc.).
            show (bool): Whether to show visualizations.

        Returns:
            dict[str, Any]: Dictionary containing all evaluation results.
        """
        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        self.model.eval()

        logger.info("Starting clean evaluation...")
        clean_metrics = self.evaluate_clean(loader)
        self.results["clean"] = self._aggregate_metrics(clean_metrics)
        if save:
            self.save_results(clean_metrics, "clean")

        for attack in self.attacks:
            logger.info(f"Starting evaluation for attack: {attack}")
            adv_metrics = self.evaluate_attack(loader, attack)
            attack_name = attack.__class__.__name__
            self.results[f"attack_{attack_name}"] = self._aggregate_metrics(adv_metrics)
            if save:
                self.save_results(adv_metrics, f"attack_{attack_name}")

        if save:
            self._save_summary_results()
            if show:
                self._create_visualizations()

        return self.results

    def get_summary(self) -> dict[str, Any]:
        """Get a summary of the evaluation results.

        Returns:
            dict[str, Any]: Summary containing key statistics and comparisons.
        """
        if not self.results:
            return {"error": "No results available. Run the pipeline first."}

        summary = {
            "total_evaluations": len(self.results),
            "evaluations": list(self.results.keys()),
            "clean_performance": self.results.get("clean", {}),
            "attack_performance": {k: v for k, v in self.results.items() if k.startswith("attack_")},
        }

        if "clean" in self.results:
            clean_metrics = self.results["clean"]
            attack_results = summary["attack_performance"]

            robustness_analysis = {}
            for attack_name, attack_metrics in attack_results.items():
                robustness = {}
                for metric in clean_metrics.keys():
                    if metric in attack_metrics and attack_metrics[metric] is not None:
                        clean_val = clean_metrics[metric]
                        attack_val = attack_metrics[metric]
                        if clean_val is not None and clean_val != 0:
                            robustness[f"{metric}_degradation"] = (clean_val - attack_val) / clean_val * 100
                        else:
                            robustness[f"{metric}_degradation"] = None
                robustness_analysis[attack_name] = robustness

            summary["robustness_analysis"] = robustness_analysis

        return summary

    def print_summary(self) -> None:
        """Print a formatted summary of the evaluation results."""
        summary = self.get_summary()

        print("\n" + "=" * 60)
        print("SEGMENTATION ROBUSTNESS EVALUATION SUMMARY")
        print("=" * 60)

        if "error" in summary:
            print(f"Error: {summary['error']}")
            return

        print(f"Total evaluations: {summary['total_evaluations']}")
        print(f"Evaluations: {', '.join(summary['evaluations'])}")

        if "clean" in self.results:
            print("\n" + "-" * 40)
            print("CLEAN PERFORMANCE")
            print("-" * 40)
            for metric, value in self.results["clean"].items():
                if value is not None:
                    print(f"{metric}: {value:.4f}")

        if "robustness_analysis" in summary:
            print("\n" + "-" * 40)
            print("ROBUSTNESS ANALYSIS")
            print("-" * 40)
            for attack_name, robustness in summary["robustness_analysis"].items():
                print(f"\n{attack_name}:")
                for metric, degradation in robustness.items():
                    if degradation is not None:
                        print(f"  {metric}: {degradation:.2f}%")
                    else:
                        print(f"  {metric}: N/A")

        print("\n" + "=" * 60)

    def get_run_info(self) -> dict[str, Any]:
        """Get information about the current run.

        Returns:
            dict[str, Any]: Dictionary containing run information.
        """
        return {
            "run_id": self.run_id,
            "output_directory": str(self.output_dir),
            "base_output_directory": str(self.base_output_dir),
            "device": self.device,
            "batch_size": self.batch_size,
            "auto_resize_masks": self.auto_resize_masks,
            "model_num_classes": self.model.num_classes,
            "dataset_size": len(self.dataset),
            "num_attacks": len(self.attacks),
            "num_metrics": len(self.metrics),
        }

    def evaluate_clean(self, loader: DataLoader) -> list[dict[str, Any]]:
        """Evaluate model on clean images.

        Args:
            loader (DataLoader): DataLoader for the dataset.

        Returns:
            list[dict[str, Any]]: List of metric results for each batch.
        """
        all_metrics = []
        for images, targets in tqdm(loader, desc="Clean Evaluation"):
            images, targets = images.to(self.device), targets.to(self.device)

            valid_mask = targets >= 0  # Exclude ignore_index (-1)
            if torch.any(valid_mask):
                max_valid_value = torch.max(targets[valid_mask])
                if max_valid_value >= self.model.num_classes:
                    targets = torch.clamp(targets, -1, self.model.num_classes - 1)
                    logger.debug("Clamped mask values to valid range")

            with torch.no_grad():
                preds = self.model.predictions(images)
            batch_metrics = self.compute_metrics(targets, preds)
            all_metrics.append(batch_metrics)
            self.clean_counter += 1
            if self.clean_counter > STOP_AFTER_N_BATCHES:
                break
        logger.info("Clean evaluation complete.")
        return all_metrics

    def evaluate_attack(self, loader: DataLoader, attack: Callable) -> list[dict[str, Any]]:
        """Evaluate model on adversarial images for a given attack.

        Args:
            loader (DataLoader): DataLoader for the dataset.
            attack (Callable): Attack instance.

        Returns:
            list[dict[str, Any]]: List of metric results for each batch.
        """
        all_metrics = []
        for images, targets in tqdm(loader, desc=f"Attack: {attack}"):
            images, targets = images.to(self.device), targets.to(self.device)

            valid_mask = targets >= 0  # Exclude ignore_index (-1)
            if torch.any(valid_mask):
                max_valid_value = torch.max(targets[valid_mask])
                if max_valid_value >= self.model.num_classes:
                    targets = torch.clamp(targets, -1, self.model.num_classes - 1)
                    logger.debug("Clamped mask values to valid range")

            adv_images = attack(images, targets)
            with torch.no_grad():
                adv_preds = self.model.predictions(adv_images)
            batch_metrics = self.compute_metrics(targets, adv_preds)
            all_metrics.append(batch_metrics)
            self.adv_counter += 1
            if self.adv_counter > STOP_AFTER_N_BATCHES:
                break
        logger.info(f"Evaluation for attack {attack} complete.")
        return all_metrics

    def compute_metrics(self, targets: torch.Tensor | np.ndarray, preds: torch.Tensor | np.ndarray) -> dict[str, Any]:
        """Compute all metrics for a batch.

        Args:
            targets (torch.Tensor | np.ndarray): Ground truth labels.
            preds (torch.Tensor | np.ndarray): Predicted labels.

        Returns:
            dict[str, Any]: Dictionary of metric results.
        """
        results = {}
        for i, metric in enumerate(self.metrics):
            metric_name = self.metric_names[i]
            try:
                results[metric_name] = metric(targets, preds)
            except Exception as e:
                logger.error(f"Metric {metric_name} failed: {e}")
                results[metric_name] = None
        return results

    def _aggregate_metrics(self, metrics: list[dict[str, Any]]) -> dict[str, float]:
        """Aggregate batch metrics into summary statistics.

        Args:
            metrics (list[dict[str, Any]]): List of batch metric results.

        Returns:
            dict[str, float]: Aggregated metrics with mean values.
        """
        if not metrics:
            return {}

        aggregated = {}
        metric_names = metrics[0].keys()

        for metric_name in metric_names:
            values = [batch[metric_name] for batch in metrics if batch[metric_name] is not None]
            if values:
                aggregated[metric_name] = float(np.mean(values))
            else:
                aggregated[metric_name] = None

        return aggregated

    def save_results(self, metrics: list[dict[str, Any]], name: str) -> None:
        """Save detailed batch metrics to disk.

        Args:
            metrics (list[dict[str, Any]]): List of metric results for each batch.
            name (str): Name for the result set (e.g., 'clean', 'attack_FGSM').
        """
        results_file = Path(self.output_dir) / f"{name}_detailed.json"
        with open(results_file, "w") as f:
            json.dump(metrics, f, indent=2, default=str)

        csv_file = Path(self.output_dir) / f"{name}_detailed.csv"
        df = pd.DataFrame(metrics)
        df.to_csv(csv_file, index=False)

        logger.info(f"Detailed results for {name} saved to {results_file} and {csv_file}")

    def _save_summary_results(self) -> None:
        """Save aggregated summary results."""
        summary_file = Path(self.output_dir) / "summary_results.json"
        with open(summary_file, "w") as f:
            json.dump(self.results, f, indent=2, default=str)

        comparison_data = []
        for name, metrics in self.results.items():
            row = {"evaluation": name}
            row.update(metrics)
            comparison_data.append(row)

        comparison_df = pd.DataFrame(comparison_data)
        comparison_file = Path(self.output_dir) / "comparison_table.csv"
        comparison_df.to_csv(comparison_file, index=False)

        logger.info(f"Summary results saved to {summary_file}")
        logger.info(f"Comparison table saved to {comparison_file}")

    def _create_visualizations(self) -> None:
        """Create and save visualization plots."""
        if not self.results:
            logger.warning("No results available for visualization")
            return

        all_metrics = set()
        for metrics in self.results.values():
            all_metrics.update(metrics.keys())

        for metric in all_metrics:
            if metric == "evaluation":
                continue

            values = []
            labels = []

            for name, metrics_dict in self.results.items():
                if metric in metrics_dict and metrics_dict[metric] is not None:
                    values.append(metrics_dict[metric])
                    labels.append(name)

            if values:
                plt.figure(figsize=(10, 6))
                plt.bar(labels, values)
                plt.title(f"{metric} Comparison")
                plt.ylabel(metric)
                plt.xticks(rotation=45, ha="right")
                plt.tight_layout()

                plot_file = Path(self.output_dir) / f"{metric}_comparison.png"
                plt.savefig(plot_file, dpi=300, bbox_inches="tight")
                plt.close()

                logger.info(f"Plot for {metric} saved to {plot_file}")

        if len(self.results) > 1:
            plt.figure(figsize=(12, 8))

            eval_names = list(self.results.keys())
            metric_names = list(all_metrics - {"evaluation"})

            if metric_names:
                data_matrix = []
                for eval_name in eval_names:
                    row = []
                    for metric in metric_names:
                        value = self.results[eval_name].get(metric, None)
                        row.append(value if value is not None else 0)
                    data_matrix.append(row)

                plt.imshow(data_matrix, cmap="viridis", aspect="auto")
                plt.colorbar(label="Metric Value")
                plt.xticks(range(len(metric_names)), metric_names, rotation=45, ha="right")
                plt.yticks(range(len(eval_names)), eval_names)
                plt.title("Performance Heatmap")
                plt.tight_layout()

                heatmap_file = Path(self.output_dir) / "performance_heatmap.png"
                plt.savefig(heatmap_file, dpi=300, bbox_inches="tight")
                plt.close()

                logger.info(f"Performance heatmap saved to {heatmap_file}")
