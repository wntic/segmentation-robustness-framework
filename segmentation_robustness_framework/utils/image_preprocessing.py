import logging
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torchvision.transforms import transforms

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)


DATASET_COLOR_MAPPINGS = {}


def register_dataset_colors(dataset_name: str, colors: list[tuple[int, int, int]]) -> None:
    """Register color mappings for a dataset.

    Args:
        dataset_name (str): Name of the dataset.
        colors (list[tuple[int, int, int]]): List of RGB color tuples for each class.
    """
    DATASET_COLOR_MAPPINGS[dataset_name] = colors


try:
    from segmentation_robustness_framework.utils._colors import (
        ADE20K_COLORS,
        CITYSCAPES_COLORS,
        STANFORD_BACKGROUND_COLORS,
        VOC_COLORS,
    )

    register_dataset_colors("voc", VOC_COLORS)
    register_dataset_colors("ade20k", ADE20K_COLORS)
    register_dataset_colors("cityscapes", CITYSCAPES_COLORS)
    register_dataset_colors("stanford_background", STANFORD_BACKGROUND_COLORS)
except ImportError as e:
    logger.warning(f"Failed to import color mappings from _colors.py: {e}")


def prepare_inputs(sample, maybe_bundle, device="cuda"):
    """Prepare model input dictionary from a sample and model or bundle.

    If the model or bundle has a `processor` attribute (e.g., HuggingFace), use it to
    process the input sample and move all resulting tensors to the specified device.
    Otherwise, assume the sample is a tensor and wrap it in a dictionary under the
    `pixel_values` key.

    Args:
        sample (PIL.Image | np.ndarray | torch.Tensor): Input image sample.
        maybe_bundle (HFSegmentationBundle | nn.Module): Model or bundle, possibly with a processor.
        device (str): Device to move tensors to. Defaults to "cuda".

    Returns:
        dict[str, torch.Tensor]: Dictionary of model input tensors.

    Example:
        ```python
        # For a HuggingFace bundle
        inputs = prepare_inputs(image, hf_bundle, device="cuda")
        # For a plain torch model
        inputs = prepare_inputs(image_tensor, model, device="cpu")
        ```
    """
    if hasattr(maybe_bundle, "processor"):
        proc_inputs = maybe_bundle.processor(sample, return_tensors="pt")
        return {k: v.to(device) for k, v in proc_inputs.items()}
    return {"pixel_values": sample.to(device)}


def get_preprocessing_fn(image_shape: list[int] | tuple[int, int], dataset_name: Optional[str] = None) -> Callable:
    """Get preprocessing functions for images and masks.

    Args:
        image_shape (list[int] | tuple[int, int]): Desired image shape [height, width].
        dataset_name (Optional[str]): Name of the dataset for dataset-specific mask processing.

    Returns:
        Callable: Tuple of (image_preprocess, target_preprocess) functions.
    """
    if not isinstance(image_shape, (list, tuple)):
        raise TypeError(f"Expected a list or tuple [height, width], but got {image_shape}")

    if len(image_shape) == 2:
        w, h = image_shape[0], image_shape[1]
        if not isinstance(h, int) and not isinstance(w, int):
            raise TypeError(f"Expected type [int, int], but got {[type(h).__name__, type(w).__name__]}")
    else:
        raise ValueError(f"Expected image_shape of length 2, but got {len(image_shape)}")

    h = (h // 8 + 1) * 8 if h % 8 != 0 else h  # height must be divisible by stride 8
    w = (w // 8 + 1) * 8 if w % 8 != 0 else w  # width must be divisible by stride 8

    preprocess = transforms.Compose([
        transforms.Resize((h, w), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    def target_preprocess(mask: Image.Image, ignore_index: int = None) -> torch.Tensor:
        """Preprocess segmentation mask to tensor format.

        Args:
            mask (Image.Image): Input segmentation mask.
            ignore_index (int, optional): Index to mark as ignored. Defaults to None.

        Returns:
            torch.Tensor: Preprocessed mask tensor with shape [H, W] and dtype long.
        """
        np_mask = np.array(mask)
        if np_mask.ndim == 3 and np_mask.shape[2] == 3:
            mask_tensor = _convert_rgb_mask_to_index(np_mask, dataset_name)
        else:
            mask_tensor = torch.from_numpy(np_mask).long()

        # Resize to (h, w)
        resize = transforms.Resize((h, w), interpolation=transforms.InterpolationMode.NEAREST)
        resized_mask_tensor = resize(mask_tensor.unsqueeze(0))
        mask_out = resized_mask_tensor.squeeze(0)  # [H, W]

        if ignore_index is not None:
            mask_out[mask_out == ignore_index] = -1

        return mask_out

    return preprocess, target_preprocess


def _convert_rgb_mask_to_index(np_mask: np.ndarray, dataset_name: Optional[str] = None) -> torch.Tensor:
    """Convert RGB mask to class index mask based on dataset color palette.

    Args:
        np_mask (np.ndarray): RGB mask array with shape [H, W, 3].
        dataset_name (Optional[str]): Name of the dataset for color mapping.

    Returns:
        torch.Tensor: Class index mask with shape [H, W] and dtype long.
    """
    if dataset_name and dataset_name in DATASET_COLOR_MAPPINGS:
        colors = DATASET_COLOR_MAPPINGS[dataset_name]
        index_mask = np.zeros(np_mask.shape[:2], dtype=np.int64)

        assigned_pixels = np.zeros(np_mask.shape[:2], dtype=bool)

        for idx, color in enumerate(colors):
            matches = np.all(np_mask == color, axis=-1)
            index_mask[matches] = idx
            assigned_pixels |= matches

        unassigned = ~assigned_pixels
        if np.any(unassigned):
            unassigned_count = np.sum(unassigned)
            total_pixels = np_mask.shape[0] * np_mask.shape[1]
            unassigned_percentage = (unassigned_count / total_pixels) * 100

            if unassigned_percentage > 5:
                logger.warning(
                    f"Found {unassigned_count} pixels ({unassigned_percentage:.1f}%) with colors not in the {dataset_name} color palette"
                )
            else:
                logger.debug(
                    f"Found {unassigned_count} pixels ({unassigned_percentage:.1f}%) with colors not in the {dataset_name} color palette"
                )

            unassigned_mask = np_mask[unassigned]  # shape [N, 3]
            palette = np.array(colors)  # shape [num_classes, 3]
            dists = ((unassigned_mask[:, None, :] - palette[None, :, :]) ** 2).sum(axis=2)  # [N, num_classes]
            nearest = np.argmin(dists, axis=1)  # [N]
            index_mask[unassigned] = nearest

        max_valid_index = len(colors) - 1
        if np.any(index_mask > max_valid_index):
            logger.warning(f"Found indices > {max_valid_index} in {dataset_name} mask, clamping to valid range")
            index_mask = np.clip(index_mask, 0, max_valid_index)

        return torch.from_numpy(index_mask).long()

    return torch.from_numpy(np_mask[..., 0]).long()
