import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor, argmax


def visualize_segmentation(output: Tensor):
    pred_labels = argmax(output, dim=1)
    predicted_labels_np = pred_labels.squeeze().cpu().numpy()

    plt.imshow(predicted_labels_np)
    plt.title("Predicted Segmentation Map")
    plt.axis("off")
    plt.show()


def denormalize(image: Tensor) -> Tensor:
    """Denormalizes input image.

    During the pre-processing stage, normalization is applied to the input image.
    This function denormalizes it.

    Args:
        image (torch.Tensor): The image tensor with shape `[1, C, H, W]`.

    Returns:
        torch.Tensor: Denormalized image.
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])

    return std * image + mean


def show_image(original_image: Tensor, segmentation_mask: Tensor):
    """Displays original image and its corresponding segmentation mask.

    Args:
        original_image (torch.Tensor): The original image tensor with shape `[1, C, H, W]`.
        segmentation_mask (torch.Tensor): The segmentation mask tensor with shape `[H, W]`.

    Returns:
        None
    """
    if original_image.ndimension() != 4:
        raise ValueError(f"Expected original image with shape [1, C, H, W], but got {list(original_image.shape)}")
    if segmentation_mask.ndimension() != 2:
        raise ValueError(f"Expected segmentation mask with shape [H, W], but got {list(segmentation_mask.shape)}")

    image = original_image.squeeze().permute(1, 2, 0).cpu().detach().numpy()
    image = denormalize(image)
    image = np.clip(image, 0, 1)
    mask = segmentation_mask.cpu().detach().numpy()

    fig = plt.figure(figsize=(10, 5))
    fig.add_subplot(1, 2, 1)
    plt.imshow(image)
    fig.add_subplot(1, 2, 2)
    plt.imshow(mask, cmap="tab20")
    plt.show()
