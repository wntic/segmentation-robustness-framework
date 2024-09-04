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
    return attack_config.target_label


class RobustnessEvaluation:
    def __init__(self, config_path: Union[Path, str]) -> None:
        with open(config_path) as f:
            data = yaml.load(f, yaml.SafeLoader)
        config = validator.Config(**data)

        self.model_config = config.model
        self.device = config.model.device
        self.attack_config = config.attacks
        self.dataset_config = config.dataset
        self.output_config = config.output

        self.save_images = self.output_config.save_images
        self.save_dir = self.output_config.save_dir

    def _load_model(self) -> nn.Module:
        if self.model_config.name == "FCN":
            model = models.FCN(
                encoder_name=self.model_config.encoder,
                encoder_weights=self.model_config.weights,
                num_classes=self.model_config.num_classes,
            )
        elif self.model_config.name == "DeepLabV3":
            model = models.FCN(
                encoder_name=self.model_config.encoder,
                encoder_weights=self.model_config.weights,
                num_classes=self.model_config.num_classes,
            )
        return model.to(self.device)

    def _load_dataset(self) -> Dataset:
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
            )
        else:
            raise ValueError(f"Invalid dataset {self.dataset_config.name}")
        return ds

    def run(self) -> None:
        model = self._load_model()
        dataset = self._load_dataset()
        attacks_list = [_get_attacks(model.to(self.device), attack) for attack in self.attack_config]
        num_images = self.dataset_config.max_images if len(dataset) > self.dataset_config.max_images else len(dataset)

        if self.save_images:
            if not os.path.exists(self.save_dir):
                os.makedirs(self.save_dir)
            run_dir = os.path.join(self.save_dir, f"run_{len(os.listdir(self.save_dir))}")

        for group_idx, attack_group in enumerate(attacks_list):
            for attack_idx, attack in enumerate(attack_group):
                attack_dir = os.path.join(run_dir, f"{attack_idx + 1}. {attack.__repr__()}")
                for idx in range(num_images):
                    image, ground_truth = dataset[idx]
                    image = image.to(self.device)

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

                    adv_image = attack(image, target_labels)
                    adv_output = model(adv_image)
                    adv_preds = torch.argmax(adv_output, dim=1)

                    utils.visualize_results(
                        image=image,
                        ground_truth=ground_truth,
                        mask=preds,
                        adv_mask=adv_preds,
                        dataset_name=self.dataset_config.name,
                        title=attack.__repr__(),
                        save=self.save_images,
                        save_dir=attack_dir,
                    )
