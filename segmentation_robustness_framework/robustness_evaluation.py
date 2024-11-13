import json
import os
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
import yaml
from torch.utils.data import Dataset

from segmentation_robustness_framework import attacks, datasets, models, utils, validator, engine


class RobustnessEvaluation:
    """Evaluates the robustness of segmentation models against adversarial attacks.

    This class is responsible for loading a segmentation model, dataset, and adversarial attacks,
    and evaluating the model's robustness by generating adversarial examples, computing predictions,
    and calculating performance metrics.

    Attributes:
        model_config (validator.ModelConfig): Configuration for the model, including the architecture,
            encoder type, weights, and number of classes.
        device (torch.device): The device on which the model will be executed (either 'cpu' or 'cuda').
        attack_config (validator.AttackConfig): Configuration for the adversarial attacks to be applied.
        dataset_config (validator.DatasetConfig): Configuration for the dataset used for evaluation,
            including dataset type, image shape, and transformations.
        output_dir (Path): Directory path where the evaluation results and images will be saved.
        save (bool): Flag indicating whether generated adversarial images and evaluation results should be saved.
        show (bool): Flag indicating whether generated adversarial images and evaluation results should be showed.
    """

    VALID_METRICS = [
        "mean_iou",
        "pixel_accuracy",
        "precision_macro",
        "precision_micro",
        "recall_macro",
        "recall_micro",
        "dice_macro",
        "dice_micro",
    ]

    def __init__(
        self,
        config_path: Union[Path, str],
        output_dir: Union[Path, str] = None,
    ) -> None:
        """Initializes RobustnessEvaluation with the given configuration file.

        The method loads the YAML configuration file that specifies the model, dataset, and
        attack configurations, and prepares the necessary attributes for evaluation.

        Args:
            config_path (Union[Path, str]): Path to the YAML configuration file.
            output_dir (Union[Path, str]): Path to the output directory. Defaults to None.
        """
        with open(config_path) as f:
            data = yaml.load(f, yaml.SafeLoader)
        config = validator.Config(**data)

        # Configs
        self.model_config = config.model
        self.device = torch.device(config.model.device)
        self.attack_config = config.attacks
        self.dataset_config = config.dataset

        # Loaders
        self.model_loader = engine.ModelLoader(self.model_config)

        # Metrics
        self.metrics = None
        self.metrics_collection = None

        # Output settings
        self.output_dir = Path("./runs/") if output_dir is None else Path(output_dir)

        self.show = None
        self.save = None

    def _validate_metrics(self, metrics: list[str]) -> None:
        """Validates metrics.

        Args:
            metrics (list[str]): Metrics for validation.

        Raises:
            ValueError: If the given metric is not correct.
        """
        for metric in metrics:
            if metric not in self.VALID_METRICS:
                raise ValueError(f"Got unexpected metric '{metric}'. Valid metrics: {self.VALID_METRICS}")

    def run(self, show: bool = False, save: bool = False, metrics: list[str] = ["mean_iou"]) -> None:
        """Executes the robustness evaluation by applying adversarial attacks and calculating metrics.

        This method loads the model, dataset, and adversarial attacks, evaluates the model on clean and
        adversarial images, and computes performance metrics for each attack. Optionally, visualizes the images.

        Args:
            show (bool): If True, visualizes images and adversarial examples during evaluation.
            save (bool): If true, saves final images.
            metrics (list[str]): Metrics to compute for evaluating image segmentation. Defaults to ["mean_iou"].
        """
        self.show = show
        self.save = save

        self._validate_metrics(metrics)
        self.metrics = metrics

        # Load model, dataset, attacks
        model = self.model_loader.load_model()
        model.eval()

        dataset = self._load_dataset()
        attacks_list = [self._get_attacks(model.to(self.device), attack) for attack in self.attack_config]
        num_images = self.dataset_config.max_images if len(dataset) > self.dataset_config.max_images else len(dataset)

        # Create save path if not exists
        self._prepare_output_dir()

        # Initialize metrics collection and structure to store metrics
        self.metrics_collection = utils.metrics.MetricsCollection(num_classes=dataset.num_classes)
        self.metrics_storage = {}

        # Step 1: Evaluate clean metrics
        clean_metrics = self._evaluate_clean_images(model, dataset, num_images)
        self.metrics_storage["clean_metrics"] = clean_metrics

        # Step 2: Apply adversarial attacks and compute metrics
        for group_idx, attack_group in enumerate(attacks_list):
            attack_group_name = self.attack_config[group_idx].name
            self.metrics_storage[attack_group_name] = {"attacks": []}

            self._evaluate_attacks(model, dataset, num_images, group_idx, attack_group)

        # Save metrics to JSON
        self._save_metrics()

    def _load_dataset(self) -> Dataset:
        """Loads the dataset based on the dataset configuration and applies necessary preprocessing.

        This method loads the dataset specified in the `dataset_config` and prepares it with transformations.

        Returns:
            Dataset: The dataset to be used for evaluation.

        Raises:
            ValueError: If the dataset name specified in the configuration is not supported.
        """
        preprocess, target_preprocess = utils.get_preprocessing_fn(self.dataset_config.image_shape)

        if self.dataset_config.name == "VOC":
            ds = datasets.VOCSegmentation(
                root=self.dataset_config.root,
                split=self.dataset_config.split,
                transform=preprocess,
                target_transform=target_preprocess,
            )
        elif self.dataset_config.name == "StanfordBackground":
            ds = datasets.StanfordBackground(
                root=self.dataset_config.root,
                transform=preprocess,
                target_transform=target_preprocess,
            )
        elif self.dataset_config.name == "ADE20K":
            ds = datasets.ADE20K(
                root=self.dataset_config.root,
                split=self.dataset_config.split,
                transform=preprocess,
                target_transform=target_preprocess,
            )
        elif self.dataset_config.name == "Cityscapes":
            ds = datasets.Cityscapes(
                root=self.dataset_config.root,
                split=self.dataset_config.split,
                mode=self.dataset_config.mode,
                target_type=self.dataset_config.target_type,
                transform=preprocess,
                target_transform=target_preprocess,
            )
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_config.name}")

        return ds

    def _get_attacks(self, model: nn.Module, attack_config: validator.AttackConfig) -> list[attacks.AdversarialAttack]:
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
            return [attacks.FGSM(model=model, eps=epsilon) for epsilon in epsilon_values]
        elif attack_name == "PGD":
            epsilon_values = attack_config.epsilon
            alpha_values = attack_config.alpha
            iters = attack_config.steps
            targeted = attack_config.targeted
            return [
                attacks.PGD(model=model, eps=epsilon, alpha=alpha, iters=iters, targeted=targeted)
                for epsilon in epsilon_values
                for alpha in alpha_values
            ]
        elif attack_name == "RFGSM":
            epsilon_values = attack_config.epsilon
            alpha_values = attack_config.alpha
            iters = attack_config.steps
            targeted = attack_config.targeted
            return [
                attacks.RFGSM(model=model, eps=epsilon, alpha=alpha, iters=iters, targeted=targeted)
                for epsilon in epsilon_values
                for alpha in alpha_values
            ]
        elif attack_name == "TPGD":
            epsilon_values = attack_config.epsilon
            alpha_values = attack_config.alpha
            iters = attack_config.steps
            return [
                attacks.TPGD(model=model, eps=epsilon, alpha=alpha, iters=iters)
                for epsilon in epsilon_values
                for alpha in alpha_values
            ]
        raise ValueError(f"Unknown attack type: {attack_name}")

    def _prepare_output_dir(self) -> None:
        """Prepares the directory for saving evaluation results and adversarial images."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.run_dir = os.path.join(self.output_dir, f"run_{len(os.listdir(self.output_dir))}")
        os.makedirs(self.run_dir)

    def _initialize_metrics_storage(self) -> dict[str, list[float]]:
        """Initializes storage for performance metrics.

        Returns:
            dict[str, list[float]]: A dictionary to store lists of performance metrics.
        """
        return {key: [] for key in self.metrics}

    def _get_model_predictions(self, model: nn.Module, image: torch.Tensor) -> torch.Tensor:
        """Obtains predictions from the segmentation model for a given image.

        Args:
            model (nn.Module): The segmentation model to use for predictions.
            image (torch.Tensor): The input image tensor.

        Returns:
            torch.Tensor: The predicted labels for the input image.
        """
        with torch.no_grad():
            output = model(image)
            preds = torch.argmax(output, dim=1)

        return preds

    def _compute_metrics(self, targets: torch.Tensor, preds: torch.Tensor) -> dict[str, float]:
        """Computes performance metrics for segmentation.

        This method calculates various metrics such as mean IoU, pixel accuracy, precision, recall and Dice score.

        Args:
            targets (torch.Tensor): The ground truth labels.
            preds (torch.Tensor): The predicted labels from the model.

        Raises:
            ValueError: If metrics collection is not initialized.
            ValueError: If the given metric is not valid.

        Returns:
            dict[str, float]: A dictionary containing calculated metrics.
        """
        if self.metrics_collection is None:
            raise ValueError("Metrics Collection is empty (None)")
        results = {}

        for metric in self.metrics:
            if metric == "mean_iou":
                results["mean_iou"] = self.metrics_collection.mean_iou(targets, preds)
            elif metric == "pixel_accuracy":
                results["pixel_accuracy"] = self.metrics_collection.pixel_accuracy(targets, preds)
            elif metric == "precision_macro":
                results["precision_macro"] = self.metrics_collection.precision(targets, preds, average="macro")
            elif metric == "precision_micro":
                results["precision_micro"] = self.metrics_collection.precision(targets, preds, average="micro")
            elif metric == "recall_macro":
                results["recall_macro"] = self.metrics_collection.recall(targets, preds, average="macro")
            elif metric == "recall_micro":
                results["recall_micro"] = self.metrics_collection.recall(targets, preds, average="micro")
            elif metric == "dice_macro":
                results["dice_macro"] = self.metrics_collection.dice_score(targets, preds, average="macro")
            elif metric == "dice_micro":
                results["dice_micro"] = self.metrics_collection.dice_score(targets, preds, average="micro")
            else:
                raise ValueError(f"Metric '{metric}' is not valid.")

        return results

    def _append_metrics(self, storage: dict[str, list[float]], metrics: dict[str, float]) -> None:
        """Appends calculated metrics to the storage dictionary.

        Args:
            storage (dict[str, list[float]]): The dictionary where metrics are stored.
            metrics (dict[str, float]): The metrics to append to the storage.
        """
        for key, value in metrics.items():
            storage[key].append(value)

    def _evaluate_clean_images(self, model: nn.Module, dataset: Dataset, num_images: int) -> dict[str, list[float]]:
        """Evaluates the model on clean images and computes metrics.

        Args:
            model (nn.Module): The segmentation model to evaluate.
            dataset (Dataset): The dataset containing images and ground truth labels.
            num_images (int): The number of images to evaluate.

        Returns:
            dict[str, list[float]]: A dictionary of clean evaluation metrics.
        """
        clean_metrics_storage = self._initialize_metrics_storage()

        for idx in range(num_images):
            image, ground_truth = dataset[idx]
            image = image.to(self.device)
            preds = self._get_model_predictions(model, image)

            clean_metrics = self._compute_metrics(targets=ground_truth, preds=preds)
            self._append_metrics(clean_metrics_storage, clean_metrics)

        return clean_metrics_storage

    def _evaluate_attacks(
        self,
        model: nn.Module,
        dataset: Dataset,
        num_images: int,
        group_idx: int,
        attack_group: int,
    ) -> None:
        """Evaluates the model on adversarial examples generated by attacks.

        Args:
            model (nn.Module): The segmentation model to be evaluated.
            dataset (Dataset): The dataset used for generating adversarial examples.
            num_images (int): The number of images to evaluate.
            group_idx (int): The index of the attack group being evaluated.
            attack_group (list[attacks.AdversarialAttack]): The list of attacks to apply.
        """
        for attack_idx, attack in enumerate(attack_group):
            attack_metrics = {
                "params": attack.get_params(),  # Parameters of the attack (e.g., epsilon, alpha)
                "adv_metrics": self._initialize_metrics_storage(),
            }

            if self.save:
                attack_dir = os.path.join(self.run_dir, f"{attack_idx + 1}. {attack.__repr__()}")

            for idx in range(num_images):
                image, ground_truth = dataset[idx]
                image = image.to(self.device)

                # Create a target tensor if the attack is targeted, else use predictions
                preds = self._get_model_predictions(model, image)
                target_labels = self._get_target_labels(attack, ground_truth, preds, group_idx)

                # Create adversarial image and do predictions
                adv_image = attack(image, target_labels).to(self.device)
                adv_preds = self._get_model_predictions(model, adv_image)

                # Visualize image, ground truth mask, predicted mask and adversarial mask
                utils.visualize_images(
                    image=image,
                    ground_truth=ground_truth,
                    mask=preds,
                    adv_mask=adv_preds,
                    dataset_name=self.dataset_config.name,
                    title=attack.__repr__(),
                    show=self.show,
                    save=self.save,
                    save_dir=attack_dir if self.save else None,
                )

                adv_metrics = self._compute_metrics(targets=ground_truth, preds=adv_preds)
                self._append_metrics(attack_metrics["adv_metrics"], adv_metrics)

            attack_group_name = self.attack_config[group_idx].name
            self.metrics_storage[attack_group_name]["attacks"].append(attack_metrics)

    def _get_target_label(self, group_idx: int) -> int:
        """Retrieves the target label for a given attack group index.

        Args:
            group_idx (int): Index of the attack group.

        Returns:
            int: The target label associated with the specified attack group.
        """
        return self.attack_config[group_idx].target_label

    def _get_target_labels(
        self, attack: attacks.AdversarialAttack, ground_truth: torch.Tensor, preds: torch.Tensor, group_idx: int
    ):
        """Generates target labels for the adversarial attack.

        If the attack is targeted, this method returns a tensor filled with the target label.
        Otherwise, it returns the predicted labels.

        Args:
            attack (attacks.AdversarialAttack): The adversarial attack being applied.
            ground_truth (torch.Tensor): The ground truth labels for the input image.
            preds (torch.Tensor): The predicted labels from the model.
            group_idx (int): Index of the attack group.

        Returns:
            torch.Tensor: The target labels for the adversarial attack.
        """
        if hasattr(attack, "targeted") and attack.targeted:
            return torch.full_like(
                input=ground_truth,
                fill_value=self._get_target_label(group_idx),
                dtype=torch.long,
                device=self.device,
            )
        return preds

    def _save_metrics(self) -> None:
        """Saves the calculated metrics to a JSON file."""
        metrics_file = os.path.join(self.run_dir, "metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(self.metrics_storage, f, indent=4)
