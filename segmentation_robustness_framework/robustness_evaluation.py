import json
import os
from pathlib import Path
from typing import Union

import torch
import torch.nn as nn
import yaml
from torch.utils.data import Dataset

from segmentation_robustness_framework import attacks, datasets, models, utils, validator


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
        output_config (validator.OutputConfig): Configuration for saving generated adversarial images and results.
        save_images (bool): Flag indicating whether generated adversarial images and evaluation results should be saved.
        output_dir (Path): Directory path where the evaluation results and images will be saved.
    """

    def __init__(self, config_path: Union[Path, str], output_dir: Union[Path, str] = None) -> None:
        """Initializes RobustnessEvaluation with the given configuration file.

        The method loads the YAML configuration file that specifies the model, dataset, and
        attack configurations, and prepares the necessary attributes for evaluation.

        Args:
            config_path (Union[Path, str]): Path to the YAML configuration file.
            output_dir (Union[Path, str]): Path to the output directory.
        """
        with open(config_path) as f:
            data = yaml.load(f, yaml.SafeLoader)
        config = validator.Config(**data)

        self.model_config = config.model
        self.device = torch.device(config.model.device)
        self.attack_config = config.attacks
        self.dataset_config = config.dataset

        self.output_dir = Path("./runs/") if output_dir is None else output_dir

    def run(self, show: bool = False, save: bool = False) -> None:
        """Executes the robustness evaluation by applying adversarial attacks and calculating metrics.

        This method loads the model, dataset, and adversarial attacks, evaluates the model on clean and
        adversarial images, and computes performance metrics for each attack. Optionally, visualizes the images.

        Args:
            show (bool): If True, visualizes images and adversarial examples during evaluation.
            save (bool): If true, saves final images.
        """
        # Load model, dataset, attacks
        model = self._load_model()
        model.eval()

        dataset = self._load_dataset()
        attacks_list = [self._get_attacks(model.to(self.device), attack) for attack in self.attack_config]
        num_images = self.dataset_config.max_images if len(dataset) > self.dataset_config.max_images else len(dataset)

        # Create save path if not exists
        self._prepare_output_dir()

        # Initialize structure to store metrics
        self.metrics_storage = {}

        # Step 1: Evaluate clean metrics
        clean_metrics = self._evaluate_clean_images(model, dataset, num_images)
        self.metrics_storage["clean_metrics"] = clean_metrics

        # Step 2: Apply adversarial attacks and compute metrics
        for group_idx, attack_group in enumerate(attacks_list):
            attack_group_name = self.attack_config[group_idx].name
            self.metrics_storage[attack_group_name] = {"attacks": []}

            self._evaluate_attacks(model, dataset, num_images, group_idx, attack_group, show, save)

        # Save metrics to JSON
        self._save_metrics()

    def _load_model(self) -> nn.Module:
        """Loads the segmentation model based on the model configuration.

        Returns:
            nn.Module: The loaded model.

        Raises:
            ValueError: If the model name is not recognized.
        """
        if self.model_config.name == "FCN":
            model = models.FCN(
                encoder_name=self.model_config.encoder,
                encoder_weights=self.model_config.weights,
                num_classes=self.model_config.num_classes,
            )
        elif self.model_config.name == "DeepLabV3":
            model = models.DeepLabV3(
                encoder_name=self.model_config.encoder,
                encoder_weights=self.model_config.weights,
                num_classes=self.model_config.num_classes,
            )
        else:
            raise ValueError(f"Unknown model: {self.model_config.name}")

        return model.to(self.device)

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
        raise ValueError(f"Unknown attack type: {attack_name}")

    def _prepare_output_dir(self) -> None:
        """Prepares the directory for saving evaluation results and adversarial images."""
        os.makedirs(self.output_dir)
        self.run_dir = os.path.join(self.output_dir, f"run_{len(os.listdir(self.output_dir))}")
        os.makedirs(self.run_dir)

    def _initialize_metrics_storage(self) -> dict[str, list[float]]:
        """Initializes storage for performance metrics.

        Returns:
            dict[str, list[float]]: A dictionary to store lists of performance metrics.
        """
        return {
            "mean_iou": [],
            "pixel_accuracy": [],
            "precision_macro": [],
            "precision_micro": [],
            "recall_macro": [],
            "recall_micro": [],
            "f1_macro": [],
            "f1_micro": [],
            "dice_score_macro": [],
            "dice_score_micro": [],
        }

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

    def _compute_metrics(self, targets: torch.Tensor, preds: torch.Tensor, num_classes: int) -> dict[str, float]:
        """Computes performance metrics for segmentation.

        This method calculates various metrics such as mean IoU, pixel accuracy, precision, recall,
        F1 score, and Dice score.

        Args:
            targets (torch.Tensor): The ground truth labels.
            preds (torch.Tensor): The predicted labels from the model.
            num_classes (int): The number of classes in the segmentation task.

        Returns:
            dict[str, float]: A dictionary containing calculated metrics.
        """
        metrics = utils.metrics.SegmentationMetric(targets=targets, preds=preds, num_classes=num_classes)
        precision, recall, f1_score = metrics.precision_recall_f1()
        dice_score = metrics.dice_coefficient()

        return {
            "mean_iou": metrics.mean_iou(),
            "pixel_accuracy": metrics.pixel_accuracy(),
            "precision_macro": precision["macro"],
            "precision_micro": precision["micro"],
            "recall_macro": recall["macro"],
            "recall_micro": recall["micro"],
            "f1_macro": f1_score["macro"],
            "f1_micro": f1_score["micro"],
            "dice_score_macro": dice_score["macro"],
            "dice_score_micro": dice_score["micro"],
        }

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

            clean_metrics = self._compute_metrics(targets=ground_truth, preds=preds, num_classes=dataset.num_classes)
            self._append_metrics(clean_metrics_storage, clean_metrics)

        return clean_metrics_storage

    def _evaluate_attacks(
        self,
        model: nn.Module,
        dataset: Dataset,
        num_images: int,
        group_idx: int,
        attack_group: int,
        show: bool,
        save: bool,
    ) -> None:
        """Evaluates the model on adversarial examples generated by attacks.

        Args:
            model (nn.Module): The segmentation model to be evaluated.
            dataset (Dataset): The dataset used for generating adversarial examples.
            num_images (int): The number of images to evaluate.
            group_idx (int): The index of the attack group being evaluated.
            attack_group (list[attacks.AdversarialAttack]): The list of attacks to apply.
            show (bool): If True, visualizes the adversarial examples and predictions during evaluation.
            save (bool): If true, saves final images.
        """
        for attack_idx, attack in enumerate(attack_group):
            attack_metrics = {
                "params": attack.get_params(),  # Parameters of the attack (e.g., epsilon, alpha)
                "adv_metrics": self._initialize_metrics_storage(),
            }

            if save:
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
                    show=show,
                    save=save,
                    save_dir=attack_dir if save else None,
                )

                adv_metrics = self._compute_metrics(
                    targets=ground_truth, preds=adv_preds, num_classes=dataset.num_classes
                )
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
