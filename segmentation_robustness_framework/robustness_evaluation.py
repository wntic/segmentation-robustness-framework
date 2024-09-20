import json
import os
from pathlib import Path
from typing import List, Union

import torch
import torch.nn as nn
import yaml
from torch.utils.data import Dataset

from segmentation_robustness_framework import attacks, datasets, models, utils, validator


def _get_attacks(model: nn.Module, attack_config: validator.AttackConfig) -> List[attacks.AdversarialAttack]:
    """Generate a list of adversarial attack instances based on the given model and attack configuration.

    Args:
        model (nn.Module): The segmentation model to attack.
        attack_config (validator.AttackConfig): Configuration for the adversarial attack,
            including the type of attack and its parameters.

    Raises:
        ValueError: If the specified attack type in `attack_config.name` is not recognized.

    Returns:
        List[attacks.AdversarialAttack]: A list of adversarial attacks corresponding to the specified attack type.
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


def _get_target_label(attack_config: validator.AttackConfig) -> int:
    """Retrieve the target label for a targeted adversarial attack from the attack configuration.

    Args:
        attack_config (validator.AttackConfig): Configuration for the adversarial attack,
            including the target label.

    Returns:
        int: The target class label for the adversarial attack.
    """
    return attack_config.target_label


class RobustnessEvaluation:
    """Evaluates the robustness of segmentation models against adversarial attacks.

    Loads the segmentation model, dataset, and specified adversarial attacks, then evaluates
    the robustness by generating adversarial examples, computing predictions and metrics.

    Attributes:
        model_config (validator.ModelConfig): Configuration for the model to be loaded.
        device (torch.device): Device on which the model will be run ('cpu' or 'cuda').
        attack_config (validator.AttackConfig): Configuration for the adversarial attacks.
        dataset_config (validator.DatasetConfig): Configuration for the dataset used in evaluation.
        output_config (validator.OutputConfig): Configuration for saving images and results.
        save_images (bool): Whether to save generated adversarial images and results.
        save_dir (Path): Directory to save the evaluation results.
    """

    def __init__(self, config_path: Union[Path, str]) -> None:
        """Initializes the RobustnessEvaluation instance by loading the configuration.

        Args:
            config_path (Union[Path, str]): Path to the configuration YAML file.
        """
        self.logger = utils.log.get_logger()

        self.logger.info("Loading configs...")
        with open(config_path) as f:
            data = yaml.load(f, yaml.SafeLoader)

        from pydantic import ValidationError

        try:
            config = validator.Config(**data)
        except ValidationError:
            self.logger.exception("Config validation error. See details below.")

        self.model_config = config.model
        self.device = torch.device(config.model.device)
        self.attack_config = config.attacks
        self.dataset_config = config.dataset
        self.output_config = config.output
        self.logger.info("Configs are loaded.")

        self.save_images = self.output_config.save_images
        self.save_dir = Path("./runs/") if self.output_config.save_dir is None else self.output_config.save_dir

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
        """Loads the dataset based on the dataset configuration and applies preprocessing.

        Returns:
            Dataset: The loaded dataset.

        Raises:
            ValueError: If the dataset name is not recognized.
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

    def run(self, show: bool = False) -> None:
        """Executes the robustness evaluation.

        Loads the model, dataset, and adversarial attacks, applies the attacks to the dataset,
        and evaluates the model's performance on both clean and adversarial images.

        If configured, it saves the generated adversarial images and results.

        Args:
            show (bool): If True, visualizes images for every evaluation iteration.

        Raises:
            ValueError: If an unknown attack is specified.
        """
        self.logger.info("Starting robustness evaluation.")

        # Load model, dataset, attacks
        self.logger.info("Loading model...")
        model = self._load_model()
        model.eval()
        self.logger.info(f"{self.model_config.name} model loaded and switched to eval mode.")

        self.logger.info("Loading dataset...")
        dataset = self._load_dataset()
        self.logger.info(f"{self.dataset_config.name} dataset loaded.")

        self.logger.info("Parsing attacks...")
        attacks_list = [_get_attacks(model.to(self.device), attack) for attack in self.attack_config]
        self.logger.info("Attacks ready to use.")

        num_images = self.dataset_config.max_images if len(dataset) > self.dataset_config.max_images else len(dataset)

        # Create save path if not exists
        self.save_dir.mkdir(parents=True, exist_ok=True)
        run_dir = os.path.join(self.save_dir, f"run_{len(os.listdir(self.save_dir))}")
        os.makedirs(run_dir)

        # Initialize structure to store metrics
        metrics_storage = {}

        # Step 1: Evaluate clean metrics once
        clean_metrics_storage = {
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

        for idx in range(num_images):
            image, ground_truth = dataset[idx]
            image = image.to(self.device)

            with torch.no_grad():
                output = model(image)
                preds = torch.argmax(output, dim=1)

            # Compute clean metrics
            clean_metrics = utils.metrics.SegmentationMetric(
                targets=ground_truth,
                preds=preds,
                num_classes=dataset.num_classes,
            )

            # Append individual image metrics for clean data
            clean_metrics_storage["mean_iou"].append(clean_metrics.mean_iou())
            clean_metrics_storage["pixel_accuracy"].append(clean_metrics.pixel_accuracy())

            clean_precision, clean_recall, clean_f1_score = clean_metrics.precision_recall_f1()
            clean_metrics_storage["precision_macro"].append(clean_precision["macro"])
            clean_metrics_storage["precision_micro"].append(clean_precision["micro"])
            clean_metrics_storage["recall_macro"].append(clean_recall["macro"])
            clean_metrics_storage["recall_micro"].append(clean_recall["micro"])
            clean_metrics_storage["f1_macro"].append(clean_f1_score["macro"])
            clean_metrics_storage["f1_micro"].append(clean_f1_score["micro"])

            clean_dice_score = clean_metrics.dice_coefficient()
            clean_metrics_storage["dice_score_macro"].append(clean_dice_score["macro"])
            clean_metrics_storage["dice_score_micro"].append(clean_dice_score["micro"])

        metrics_storage["clean_metrics"] = clean_metrics_storage

        # Step 2: Apply adversarial attacks and compute metrics
        for group_idx, attack_group in enumerate(attacks_list):
            attack_group_name = self.attack_config[group_idx].name
            metrics_storage[attack_group_name] = {"attacks": []}

            for attack_idx, attack in enumerate(attack_group):
                attack_metrics = {
                    "params": {},  # Parameters of the attack (e.g., epsilon, alpha)
                    "adv_metrics": {
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
                    },
                }

                # Save attack parameters (e.g., epsilon, alpha)
                attack_metrics["params"] = attack.get_params()

                if self.save_images:
                    attack_dir = os.path.join(run_dir, f"{attack_idx + 1}. {attack.__repr__()}")

                for idx in range(num_images):
                    image, ground_truth = dataset[idx]
                    image = image.to(self.device)

                    # Create a target tensor if the attack is targeted, else use predictions
                    output = model(image)
                    preds = torch.argmax(output, dim=1)
                    if hasattr(attack, "targeted"):
                        if attack.targeted:
                            target_labels = torch.full_like(
                                input=ground_truth,
                                fill_value=_get_target_label(self.attack_config[group_idx]),
                                dtype=torch.long,
                                device=self.device,
                            )
                        else:
                            target_labels = preds
                    else:
                        target_labels = preds

                    # Create adversarial image and do predictions
                    adv_image = attack(image, target_labels)

                    with torch.no_grad():
                        adv_output = model(adv_image)
                        adv_preds = torch.argmax(adv_output, dim=1)

                    # Visualize image, ground truth mask, predicted mask and adversarial mask
                    if show:
                        utils.visualize_results(
                            image=image,
                            ground_truth=ground_truth,
                            mask=preds,
                            adv_mask=adv_preds,
                            dataset_name=self.dataset_config.name,
                            title=attack.__repr__(),
                            save=self.save_images,
                            save_dir=attack_dir if self.save_images else None,
                        )

                    # Compute metrics on adversarial data
                    adv_metrics = utils.metrics.SegmentationMetric(
                        targets=ground_truth,
                        preds=adv_preds,
                        num_classes=dataset.num_classes,
                    )

                    # Append individual image metrics for adversarial data
                    attack_metrics["adv_metrics"]["mean_iou"].append(adv_metrics.mean_iou())
                    attack_metrics["adv_metrics"]["pixel_accuracy"].append(adv_metrics.pixel_accuracy())

                    adv_precision, adv_recall, adv_f1_score = adv_metrics.precision_recall_f1()
                    attack_metrics["adv_metrics"]["precision_macro"].append(adv_precision["macro"])
                    attack_metrics["adv_metrics"]["precision_micro"].append(adv_precision["micro"])
                    attack_metrics["adv_metrics"]["recall_macro"].append(adv_recall["macro"])
                    attack_metrics["adv_metrics"]["recall_micro"].append(adv_recall["micro"])
                    attack_metrics["adv_metrics"]["f1_macro"].append(adv_f1_score["macro"])
                    attack_metrics["adv_metrics"]["f1_micro"].append(adv_f1_score["micro"])

                    adv_dice_score = adv_metrics.dice_coefficient()
                    attack_metrics["adv_metrics"]["dice_score_macro"].append(adv_dice_score["macro"])
                    attack_metrics["adv_metrics"]["dice_score_micro"].append(adv_dice_score["micro"])

                # Store the metrics for the current attack
                metrics_storage[attack_group_name]["attacks"].append(attack_metrics)

        # Save metrics to JSON after evaluation
        metrics_file = os.path.join(run_dir, "metrics.json")
        with open(metrics_file, "w") as f:
            json.dump(metrics_storage, f, indent=4)

        self.logger.info("Robustness evaluation completed.")
