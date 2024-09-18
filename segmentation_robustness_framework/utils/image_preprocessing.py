from typing import Optional, Tuple, List, Callable

from PIL import Image
import torch
import numpy as np
from torchvision.transforms import transforms


def preprocess_image(image_path: str, image_shape: Optional[Tuple[int, int]] = None) -> torch.Tensor:
    """Preprocess an image for input into a segmentation model.

    This function loads an image from the specified file path, resizes it to the target shape if provided,
    or ensures its dimensions are divisible by 8 by default, converts it to a tensor, and normalizes it.

    Args:
        image_path (str): Path to the image file.
        image_shape (Tuple[int, int], optional): The desired image shape as a tuple (height, width).
            If None, the image is resized to the closest dimensions that are divisible by stride 8. Defaults to None.

    Returns:
        torch.Tensor: The preprocessed image as a 4D tensor `[1, C, H, W]` suitable for input into a segmentation model.

    Raises:
        TypeError: If `image_path` is not of type `str`.
    """
    if not isinstance(image_path, str):
        raise TypeError(f"Expected type str, but got {type(image_path).__name__}")

    image = Image.open(image_path).convert("RGB")

    # Check and validate image shape
    if image_shape is None:
        w, h = image.size
        h = (h // 8 + 1) * 8 if h % 8 != 0 else h  # height must be divisible by stride 8
        w = (w // 8 + 1) * 8 if w % 8 != 0 else w  # width must be divisible by stride 8
    elif not isinstance(image_shape, Tuple):
        raise TypeError(f"Expected a tuple (height, width), but got {image_shape}")
    elif len(image_shape) == 2:
        h, w = image_shape[0], image_shape[1]
        if not isinstance(h, int) and not isinstance(w, int):
            raise TypeError(f"Expected type (int, int), but got {(type(h).__name__, type(w).__name__)}")
    else:
        raise ValueError(f"Expected a tuple (height, width), but got {image_shape}")

    preprocess = transforms.Compose([
        transforms.Resize((h, w)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = preprocess(image)
    image = image.unsqueeze(0)
    return image


def get_preprocessing_fn(image_shape: List[int]) -> Callable:
    # Check and validate image shape
    if not isinstance(image_shape, List):
        raise TypeError(f"Expected a list [height, width], but got {image_shape}")

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

    def target_preprocess(mask: Image, ignore_index: int = None) -> torch.Tensor:
        np_mask = np.array(mask)
        mask_tensor = torch.from_numpy(np_mask).long()
        if ignore_index is not None:
            mask_tensor[mask_tensor == ignore_index] = -1
        resize = transforms.Resize((h, w), interpolation=transforms.InterpolationMode.NEAREST)
        resized_mask_tensor = resize(mask_tensor.unsqueeze(0))
        return resized_mask_tensor

    return preprocess, target_preprocess
