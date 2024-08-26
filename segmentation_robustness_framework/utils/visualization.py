from typing import List, Tuple

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor

from .colormaps import tab20_colors, voc_classes
from .image_utils import denormalize


def create_legend(mask: np.ndarray, classes: list, colors: list) -> Tuple[List[plt.Rectangle], List[str]]:
    unique_classes = np.unique(mask)

    filtered_classes = [classes[i] for i in unique_classes]
    filtered_colormap = [colors[i] for i in unique_classes]

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=color) for color in filtered_colormap]
    return handles, filtered_classes


def visualize_segmentation(
    original_image: Tensor,
    mask: Tensor,
    adv_mask: Tensor,
    classes=voc_classes,
    colors=tab20_colors,
    fname: str = "output",
) -> None:
    """Displays original image, its corresponding segmentation mask and adversarial segmentation mask.

    Args:
        original_image (torch.Tensor): The original image tensor with shape `[1, C, H, W]`.
        mask (torch.Tensor): The segmentation mask tensor with shape `[1, H, W]`.
        adv_mask (torch.Tensor): The perturbed segmentation mask tensor with shape `[1, H, W]`.

    Returns:
        None
    """
    if original_image.ndimension() != 4:
        raise ValueError(f"Expected original image with shape [1, C, H, W], but got {list(original_image.shape)}")
    if mask.ndimension() != 3:
        raise ValueError(f"Expected segmentation mask with shape [1, H, W], but got {list(mask.shape)}")

    image = original_image.squeeze().permute(1, 2, 0).cpu().detach().numpy()
    image = denormalize(image)
    image = np.clip(image, 0, 1)
    np_mask = mask.squeeze().cpu().detach().numpy()
    np_adv_mask = adv_mask.squeeze().cpu().detach().numpy()

    fig = plt.figure(figsize=(15, 5))

    fig.add_subplot(1, 3, 1)
    plt.imshow(image)
    plt.axis("off")

    cmap = mcolors.ListedColormap(colors)
    norm = mcolors.BoundaryNorm(np.arange(len(classes) + 1) - 0.5, len(classes))

    fig.add_subplot(1, 3, 2)
    plt.imshow(np_mask, cmap=cmap, norm=norm)
    handles, filtered_classes = create_legend(np_mask, classes, colors)
    plt.legend(
        handles,
        filtered_classes,
        bbox_to_anchor=(0.5, -0.05),
        loc="upper center",
        borderaxespad=0.0,
        fancybox=True,
        ncols=3,
    )
    plt.axis("off")

    fig.add_subplot(1, 3, 3)
    plt.imshow(np_adv_mask, cmap=cmap, norm=norm)
    handles, filtered_classes = create_legend(np_adv_mask, classes, colors)
    plt.legend(
        handles,
        filtered_classes,
        bbox_to_anchor=(0.5, -0.05),
        loc="upper center",
        borderaxespad=0.0,
        fancybox=True,
        ncols=3,
    )
    plt.axis("off")

    plt.subplots_adjust(wspace=0.05)
    plt.savefig(f"outputs/{fname}.jpg")
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
