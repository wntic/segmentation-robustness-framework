import json
import os
from pathlib import Path
from typing import Any, Union

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
        return classes.VOC_classes, colors.VOC_COLORS
    if ds_name == "ADE20K":
        return classes.ADE20K_classes, colors.ADE20K_COLORS
    if ds_name == "StanfordBackground":
        return classes.StanfordBackground_classes, colors.STANFORD_BACKGROUND_COLORS
    if ds_name == "Cityscapes":
        return classes.Cityscapes_classes, colors.CITYSCAPES_COLORS
    raise ValueError(f"Invalide dataset {ds_name}")


def create_legend(mask: np.ndarray, classes: list[str], colors: list[tuple[int]]):
    """
    Creates a legend for a segmentation mask using unique classes and their corresponding colors.

    Args:
        mask (np.ndarray): Segmentation mask array where each pixel is assigned a class index.
        classes (list[str]): List of class names corresponding to the indices in the mask.
        colors (list[tuple[int]]): List of RGB color tuples corresponding to each class index.

    Returns:
        tuple:
            handles (list[matplotlib.patches.Rectangle]): List of colored rectangles representing the class colors for the legend.
            filtered_classes (list[str]): List of class names that are present in the mask.
    """
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
    """
    Visualizes an original image, ground truth, predicted mask, and adversarial mask side by side.

    Args:
        image (Tensor): Original image tensor with shape [1, C, H, W].
        ground_truth (Tensor): Ground truth segmentation mask with shape [1, H, W].
        mask (Tensor): Predicted segmentation mask with shape [1, H, W].
        adv_mask (Tensor): Adversarial segmentation mask with shape [1, H, W].
        dataset_name (str): Name of the dataset to retrieve class names and colors.
        denormalize_image (bool): Whether to denormalize the original image.
        title (str, optional): Title for the figure.
        show (bool): Whether to show the plot after creation.
        save (bool): Whether to save the plot.
        save_dir (str, optional): Directory to save the plot.

    Raises:
        ValueError: If the shape of any tensor is not as expected.
    """
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
    if show:
        plt.show()
    plt.close()


def visualize_metrics(
    json_data: Union[Path, str, dict[str, Any]],
    attack_name: str,
    attack_param: str,
    metric_names: Union[str, list[str]],
) -> None:
    """Visualizes adversarial metrics from the JSON data for the specified attack.

    Args:
        json_data (Union[Path, str, dict[str, any]]): Path to the JSON file, JSON string, or dictionary containing the metrics data.
        attack_name (str): The name of the attack whose metrics are to be visualized.
        attack_param (str): The parameter of the attack (e.g., epsilon) to plot on the x-axis.
        metric_names (Union[str, list[str]]): A single metric name (str) or a list of metric names (list of str) to visualize.

    Raises:
        FileNotFoundError: If the provided file path does not exist.
        json.JSONDecodeError: If there is an error decoding the JSON file.
        ValueError: If the specified attack is not found in the JSON data.

    Example:
        visualize_metrics("metrics.json", "FGSM", "epsilon", "accuracy")
        visualize_metrics("metrics.json", "FGSM", "epsilon", ["accuracy", "iou"])
    """
    if isinstance(json_data, (str, Path)):
        file_path = Path(json_data)
        if file_path.suffix == ".json":
            try:
                with open(file_path, encoding="utf-8") as file:
                    json_data = json.load(file)
            except FileNotFoundError:
                print(f"File {file_path} does not exist.")
                return
            except json.JSONDecodeError:
                raise ValueError(f"Error decoding JSON from file {file_path}.")

    if attack_name not in json_data:
        raise ValueError(f"Attack {attack_name} not found in the JSON data.")

    attack_data = json_data[attack_name]["attacks"]

    param_values = []

    if isinstance(metric_names, str):
        metric_names = [metric_names]

    metrics_data = {metric: [] for metric in metric_names}

    for attack in attack_data:
        param_value = attack["params"][attack_param]
        param_values.append(param_value)

        for metric in metric_names:
            mean_adv_metric = np.mean(attack["adv_metrics"][metric])
            metrics_data[metric].append(mean_adv_metric)

    plt.figure(figsize=(10, 6))

    for metric in metric_names:
        plt.plot(param_values, metrics_data[metric], marker="o", label=f"{metric}")

    plt.title(f"Metrics vs {attack_param} for {attack_name} Attack")
    plt.xlabel(attack_param)
    plt.ylabel("Metric Value")
    plt.legend()
    plt.grid(True)
    plt.show()


def print_clean_metrics(
    json_data: Union[Path, str, dict[str, Any]],
    metric_names: Union[str, list[str]],
) -> None:
    """Prints clean metrics from the JSON data.

    Args:
        json_data (Union[Path, str, dict[str, any]]): Path to the JSON file, JSON string, or dictionary containing the metrics data.
        metric_names (Union[str, list[str]]): A single metric name (str) or a list of metric names to print from the clean metrics.

    Raises:
        FileNotFoundError: If the provided file path does not exist.
        json.JSONDecodeError: If there is an error decoding the JSON file.
        ValueError: If clean metrics are not found in the JSON data.

    Example:
        print_clean_metrics("metrics.json", "accuracy")
        print_clean_metrics("metrics.json", ["precision_macro", "recall_macro"])
    """
    if isinstance(json_data, (str, Path)):
        file_path = Path(json_data)
        if file_path.suffix == ".json":
            try:
                with open(file_path, encoding="utf-8") as file:
                    json_data = json.load(file)
            except FileNotFoundError:
                print(f"File {file_path} does not exist.")
                return
            except json.JSONDecodeError:
                print(f"Error decoding JSON from file {file_path}.")
                return

    if "clean_metrics" not in json_data:
        raise ValueError("Clean metrics not found in the JSON data.")

    clean_metrics = json_data["clean_metrics"]

    if isinstance(metric_names, str):
        metric_names = [metric_names]

    print("Clean Metrics:")
    for metric in metric_names:
        if metric in clean_metrics:
            mean_clean_metric = np.mean(clean_metrics[metric])
            print(f"{metric}: {mean_clean_metric:.3f}")
        else:
            print(f"{metric}: not found in clean metrics")
