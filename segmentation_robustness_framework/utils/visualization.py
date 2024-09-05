import os
from typing import List, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor

from . import _classes as classes
from . import _colors as colors
from .image_utils import denormalize


def get_class_colors(ds_name: str) -> Tuple[List[str], List[Tuple[int]]]:
    """Provides the class and associated colors for the specified dataset.

    Args:
        ds_name (str): Dataset name.

    Raises:
        ValueError: If the specified dataset does not exist.

    Returns:
        Tuple[List[str], List[Tuple[int]]]: Tuple of classes and colors.
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


def create_legend(mask: np.ndarray, classes: List[str], colors: List[Tuple[int]]):
    unique_classes = np.unique(mask)

    filtered_classes = [classes[i] for i in unique_classes]
    filtered_colormap = [colors[i] for i in unique_classes]
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color) for color in filtered_colormap]

    return handles, filtered_classes


def visualize_results(
    image: Tensor,
    ground_truth: Tensor,
    mask: Tensor,
    adv_mask: Tensor,
    dataset_name: str,
    title: str,
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
        plt.savefig(f"{save_dir}/{len(os.listdir(save_dir))}.jpg")
    plt.show()


def show_image(original_image: Tensor, segmentation_mask: Tensor, normalize=True) -> None:
    """Displays original image and its corresponding segmentation mask.

    Args:
        original_image (torch.Tensor): The original image tensor with shape `[1, C, H, W]`.
        segmentation_mask (torch.Tensor): The segmentation mask tensor with shape `[1, H, W]`.

    Returns:
        None
    """
    if original_image.ndimension() != 4:
        raise ValueError(f"Expected original image with shape [1, C, H, W], but got {list(original_image.shape)}")
    if segmentation_mask.ndimension() != 2:
        raise ValueError(f"Expected segmentation mask with shape [1, H, W], but got {list(segmentation_mask.shape)}")

    image = original_image.squeeze().permute(1, 2, 0).cpu().detach().numpy()
    if normalize:
        image = denormalize(image)
    image = np.clip(image, 0, 1)
    mask = segmentation_mask.cpu().detach().numpy()

    fig = plt.figure(figsize=(10, 5))
    fig.add_subplot(1, 2, 1)
    plt.imshow(image)
    fig.add_subplot(1, 2, 2)
    plt.imshow(mask, cmap="tab20")
    plt.show()
