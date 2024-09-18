import os
from pathlib import Path
from typing import List, Union

import torch
import torch.nn as nn
import yaml
from torch.utils.data import Dataset

from segmentation_robustness_framework import attacks, datasets, get_preprocessing_fn, models, utils
from segmentation_robustness_framework.config import validator


def _get_attacks(model: nn.Module, attack_config: validator.AttackConfig) -> List[attacks.AdversarialAttack]:
    """Generate a list of adversarial attack instances based on the given model and attack configuration.

    Args:
        model (nn.Module): The segmentation model to attack.
        attack_config (validator.AttackConfig): Configuration for the adversarial attack,
            which includes the type of attack and its parameters.

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
    else:
        raise ValueError(f"Unknown attack: {attack_name}")


def _get_target_label(attack_config: validator.AttackConfig) -> int:
    """Retrieve the target label for a targeted adversarial attack from the attack configuration.

    Args:
        attack_config (validator.AttackConfig): Configuration for the adversarial attack,
            which includes the target label.

    Returns:
        int: The target class label for the adversarial attack.
    """
    return attack_config.target_label


class RobustnessEvaluation:
    """A class to evaluate the robustness of image segmentation models to adversarial attacks.

    This class loads a segmentation model, dataset, and specified adversarial attacks, then evaluates
    the robustness of the model by generating adversarial examples, computing the resulting predictions and metrics.

    Attributes:
        model_config (validator.ModelConfig): Configuration for the model to be loaded.
        device (torch.device): The device on which the model will be run ('cpu' or 'cuda').
        attack_config (validator.AttackConfig): Configuration for the adversarial attacks.
        dataset_config (validator.DatasetConfig): Configuration for the dataset used in evaluation.
        output_config (validator.OutputConfig): Configuration for saving images and results.
        save_images (bool): Whether to save the generated adversarial images and results.
        save_dir (Path): Directory to save the evaluation results.

    Methods:
        _load_model() -> nn.Module:
            Loads the segmentation model based on the configuration.

        _load_dataset() -> Dataset:
            Loads the dataset based on the configuration and applies preprocessing.

        run() -> None:
            Executes the robustness evaluation by running the model on the dataset,
            applying the adversarial attacks, computing multiclass metrics and saving
            the results if configured.
    """

    def __init__(self, config_path: Union[Path, str]) -> None:
        """Initializes the RobustnessEvaluation instance by loading the configuration.

        Args:
            config_path (Union[Path, str]): Path to the configuration YAML file.
        """
        with open(config_path) as f:
            data = yaml.load(f, yaml.SafeLoader)
        config = validator.Config(**data)

        self.model_config = config.model
        self.device = torch.device(config.model.device)
        self.attack_config = config.attacks
        self.dataset_config = config.dataset
        self.output_config = config.output

        self.save_images = self.output_config.save_images
        self.save_dir = Path("./runs/") if self.output_config.save_dir is None else self.output_config.save_dir

    def _load_model(self) -> nn.Module:
        """Loads a segmentation model based on the model configuration.

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
        return model.to(self.device)

    def _load_dataset(self) -> Dataset:
        """Loads the dataset based on the dataset configuration and applies preprocessing.

        Returns:
            Dataset: The loaded dataset.

        Raises:
            ValueError: If the dataset name is not recognized.
        """
        preprocess, target_preprocess = get_preprocessing_fn(self.dataset_config.image_shape)

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
            raise ValueError(f"Invalid dataset {self.dataset_config.name}")
        return ds

    def run(self) -> None:
        """Executes the robustness evaluation.

        Loads the model, dataset, and adversarial attacks, then applies the attacks to the dataset
        and evaluates the model's performance on both clean and adversarial images.

        If configured, it saves the generated adversarial images and results.

        Raises:
            ValueError: If an unknown attack is specified.
        """
        # Load model, dataset, attacks
        model = self._load_model()
        model.eval()

        dataset = self._load_dataset()
        attacks_list = [_get_attacks(model.to(self.device), attack) for attack in self.attack_config]
        num_images = self.dataset_config.max_images if len(dataset) > self.dataset_config.max_images else len(dataset)

        # Create save path if not exists
        if self.save_images:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            run_dir = os.path.join(self.save_dir, f"run_{len(os.listdir(self.save_dir))}")

        # SRF main loop
        for group_idx, attack_group in enumerate(attacks_list):
            for attack_idx, attack in enumerate(attack_group):
                if self.save_images:
                    attack_dir = os.path.join(run_dir, f"{attack_idx + 1}. {attack.__repr__()}")
                for idx in range(num_images):
                    image, ground_truth = dataset[idx]
                    image = image.to(self.device)

                    print(image.shape)
                    output = model(image)
                    preds = torch.argmax(output, dim=1)

                    # Create a target tensor if the attack is targeted, else use predictions
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
                    adv_output = model(adv_image)
                    adv_preds = torch.argmax(adv_output, dim=1)

                    # Visualize image, ground truth mask, predicted mask and adversarial mask
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
