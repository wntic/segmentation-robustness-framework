import logging
import os
from typing import Any, Callable, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from segmentation_robustness_framework.adapters.base_protocol import SegmentationModelProtocol
from segmentation_robustness_framework.utils import image_preprocessing
from segmentation_robustness_framework.utils.model_utils import get_model_output_size

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

STOP_AFTER_N_BATCHES = 1


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
        self.output_dir = output_dir or "./runs"
        self.auto_resize_masks = auto_resize_masks

        os.makedirs(self.output_dir, exist_ok=True)
        logger.info(f"Pipeline initialized. Output dir: {self.output_dir}")

        if self.auto_resize_masks:
            self._setup_automatic_mask_resizing()

        self.clean_counter = 0
        self.adv_counter = 0

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

    def run(self, save: bool = False, show: bool = False) -> None:
        """Run the evaluation pipeline: clean and adversarial evaluation.

        Args:
            save (bool): Whether to save results (images, metrics, etc.).
            show (bool): Whether to show visualizations (not implemented here).
        """
        loader = DataLoader(self.dataset, batch_size=self.batch_size, shuffle=False)
        self.model.eval()

        logger.info("Starting clean evaluation...")
        clean_metrics = self.evaluate_clean(loader)
        self.save_results(clean_metrics, "clean")

        for attack in self.attacks:
            logger.info(f"Starting evaluation for attack: {attack}")
            adv_metrics = self.evaluate_attack(loader, attack)
            self.save_results(adv_metrics, f"attack_{attack.__class__.__name__}")

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
        for metric in self.metrics:
            metric_name = metric.__name__ if hasattr(metric, "__name__") else metric.__class__.__name__
            try:
                results[metric_name] = metric(targets, preds)
            except Exception as e:
                logger.error(f"Metric {metric_name} failed: {e}")
                results[metric_name] = None
        return results

    def save_results(self, metrics: list[dict[str, Any]], name: str) -> None:
        """Save metrics or results to disk.

        This is a hook method that can be overridden to implement custom saving logic.
        Currently logs the results but does not save to disk.

        Args:
            metrics (list[dict[str, Any]]): List of metric results for each batch.
            name (str): Name for the result set (e.g., 'clean', 'attack_FGSM').
        """
        # Implement saving as JSON, CSV, etc. as needed.
        logger.info(f"Results for {name} ready to be saved. (Saving not implemented in this hook.)")
        pass
