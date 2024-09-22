import json
import os
from pathlib import Path
from typing import Union

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor

from . import _classes as classes
from . import _colors as colors


def denormalize(image: np.ndarray) -> np.ndarray:
    """Denormalizes input image.

    During the pre-processing stage, normalization is applied to the input image.
    This function denormalizes it.

    Args:
        image (np.ndarray): The 3D tensor with shape `[H, W, C]`.

    Returns:
        np.ndarray: Denormalized image.
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    return std * image + mean


def get_class_colors(ds_name: str) -> tuple[list[str], list[tuple[int]]]:
    """Provides the class and associated colors for the specified dataset.

    Args:
        ds_name (str): Dataset name.

    Raises:
        ValueError: If the specified dataset does not exist.

    Returns:
        tuple[list[str], list[tuple[int]]]: tuple of classes and colors.
    """
    if ds_name == "VOC":
        return classes.VOC_classes, colors.VOC_colors
    if ds_name == "ADE20K":
        return classes.ADE20K_classes, colors.ADE20K_colors
    if ds_name == "StanfordBackground":
        return classes.StanfordBackground_classes, colors.StanfordBackground_colors
    if ds_name == "Cityscapes":
        return classes.Cityscapes_classes, colors.Cityscapes_colors
    raise ValueError(f"Invalide dataset {ds_name}")


def create_legend(mask: np.ndarray, classes: list[str], colors: list[tuple[int]]):
    unique_classes = np.unique(mask)

    filtered_classes = [classes[i] for i in unique_classes]
    filtered_colormap = [colors[i] for i in unique_classes]
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color) for color in filtered_colormap]

    return handles, filtered_classes


def visualize_images(
    image: Tensor,
    ground_truth: Tensor,
    mask: Tensor,
    adv_mask: Tensor,
    dataset_name: str,
    denormalize_image: bool = True,
    title: str = None,
    show: bool = False,
    save: bool = False,
    save_dir: str = None,
) -> None:
    if image.ndimension() != 4:
        raise ValueError(f"Expected original image with shape [1, C, H, W], but got {list(image.shape)}")
    if ground_truth.ndimension() != 3:
        raise ValueError(f"Expected ground truth with shape [1, H, W], but got {list(ground_truth.shape)}")
    if mask.ndimension() != 3:
        raise ValueError(f"Expected segmentation mask with shape [1, H, W], but got {list(mask.shape)}")
    if adv_mask.ndimension() != 3:
        raise ValueError(f"Expected adversarial segmentation mask with shape [1, H, W], but got {list(adv_mask.shape)}")

    if save:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    image = image.squeeze().permute(1, 2, 0).cpu().detach().numpy()
    if denormalize_image:
        image = denormalize(image)
    image = np.clip(image, 0, 1)

    np_ground_truth = ground_truth.squeeze().cpu().detach().numpy()
    np_mask = mask.squeeze().cpu().detach().numpy()
    np_adv_mask = adv_mask.squeeze().cpu().detach().numpy()

    classes, colors = get_class_colors(dataset_name)
    colors = np.array(colors) / 255.0
    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(np.arange(len(classes) + 1) - 0.5, len(classes))

    fig = plt.figure(figsize=(16, 4))
    if title is not None:
        fig.suptitle(title)

    fig.add_subplot(1, 4, 1)
    plt.imshow(image)
    plt.axis("off")

    fig.add_subplot(1, 4, 2)
    plt.imshow(np_ground_truth, cmap=cmap, norm=norm)
    plt.axis("off")

    fig.add_subplot(1, 4, 3)
    plt.imshow(np_mask, cmap=cmap, norm=norm)
    handles, filtered_classes = create_legend(np_mask, classes, colors)
    plt.legend(
        handles,
        filtered_classes,
        bbox_to_anchor=(0.5, -0.05),
        loc="upper center",
        borderaxespad=0.0,
        fancybox=True,
        ncols=2,
    )
    plt.axis("off")

    fig.add_subplot(1, 4, 4)
    plt.imshow(np_adv_mask, cmap=cmap, norm=norm)
    handles, filtered_classes = create_legend(np_adv_mask, classes, colors)
    plt.legend(
        handles,
        filtered_classes,
        bbox_to_anchor=(0.5, -0.05),
        loc="upper center",
        borderaxespad=0.0,
        fancybox=True,
        ncols=2,
    )
    plt.axis("off")

    plt.subplots_adjust(wspace=0.05)
    
    if save:
        plt.savefig(f"{save_dir}/{len(os.listdir(save_dir))}.jpg", bbox_inches="tight")
        plt.close()

    if show:
        plt.show()


def visualize_metrics(json_data: Union[Path, str, dict[str, any]], attack_name, attack_param, metric_name) -> None:
    if isinstance(json_data, (str, Path)):
        file_path = Path(json_data)
        if file_path.suffix == ".json":
            try:
                with open(file_path, encoding="utf-8") as file:
                    json_data = json.load(file)
            except FileNotFoundError:
                print(f"File {file_path} does not exist.")
            except json.JSONDecodeError:
                print(f"Error decoding JSON from file {file_path}.")

    if attack_name not in json_data:
        raise ValueError(f"Attack {attack_name} not found in the JSON data.")

    clean_metrics = json_data["clean_metrics"][metric_name]
    mean_clean_metric = np.mean(clean_metrics)

    attack_data = json_data[attack_name]["attacks"]

    param_values = []
    adv_metrics = []

    for attack in attack_data:
        param_value = attack["params"][attack_param]
        param_values.append(param_value)

        mean_adv_metric = np.mean(attack["adv_metrics"][metric_name])
        adv_metrics.append(mean_adv_metric)

    plt.figure(figsize=(10, 6))
    plt.plot(param_values, adv_metrics, marker="o", label=f"{attack_name} {metric_name}", color="b")

    plt.axhline(y=mean_clean_metric, color="r", label=f"Clean {metric_name}")
    plt.text(
        x=plt.xlim()[1],  # Используем правую границу оси X
        y=mean_clean_metric,
        s=f"{mean_clean_metric:.2f}",  # Текст с округленным значением
        color="black",
        va="bottom",  # Вертикальное выравнивание по центру
        ha="right",  # Горизонтальное выравнивание по правой стороне
    )

    plt.title(f"{metric_name} vs {attack_param} for {attack_name} Attack")
    plt.xlabel(attack_param)
    plt.ylabel(metric_name)
    plt.legend()
    plt.grid(True)
    plt.show()
