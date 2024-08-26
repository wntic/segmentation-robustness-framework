import numpy as np


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
