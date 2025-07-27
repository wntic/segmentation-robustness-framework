import os
from pathlib import Path
from typing import Callable, Optional, Union

from PIL import Image
from torch.utils.data import Dataset

from segmentation_robustness_framework.datasets.registry import register_dataset
from segmentation_robustness_framework.utils.dataset_utils import download as download_dataset
from segmentation_robustness_framework.utils.dataset_utils import extract as extract_dataset


@register_dataset("ade20k")
class ADE20K(Dataset):
    """ADE20K dataset for semantic segmentation.

    The ADE20K dataset contains 20,210 images with 150 semantic categories.
    Images are paired with pixel-level segmentation masks for training and evaluation.

    **Setup Instructions:**

    The dataset will be automatically downloaded and extracted if not present.
    When `download=True` (default):
        - If `root` is provided, the dataset will be stored at `root/ade20k/ADEChallengeData2016/`.
        - If `root` is `None`, the dataset will be cached in the default cache directory.
    When `download=False`:
        - The dataset must be present at the exact path specified by `root`.
        - If `root` is `None`, the dataset will be looked for in the default cache directory.

    **Supported Splits:**
    - `train`: Training images (~20,000 samples)
    - `val`: Validation images (~2,000 samples)

    Attributes:
        root (str | Path | None): Directory for dataset storage or cache location.
        split (str): Dataset split ('train', 'val').
        transform (callable, optional): Image transformations.
        target_transform (callable, optional): Target transformations.
        download (bool): Whether to download dataset if not present.
        num_classes (int): Number of semantic classes (150).
    """

    URL = "https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip"
    MD5 = "7328b3957e407ddae1d3cbf487f149ef"
    VALID_SPLITS = ["train", "val"]

    def __init__(
        self,
        split: str,
        root: Optional[Union[Path, str]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
    ) -> None:
        """Initialize ADE20K dataset.

        Args:
            split (str): Dataset split. Must be one of 'train' or 'val'.
            root (str | Path | None, optional): Directory for dataset storage.
                If `None`, uses default cache directory. Defaults to None.
            transform (callable, optional): Transform to apply to images.
                Defaults to None.
            target_transform (callable, optional): Transform to apply to masks.
                Defaults to None.
            download (bool, optional): Whether to download dataset if not present.
                Defaults to True.

        Raises:
            FileNotFoundError: If dataset is not found and download fails.
            ValueError: If split is not valid.
        """
        from segmentation_robustness_framework.utils.dataset_utils import get_cache_dir

        if download:
            root_path = Path(root) / "ade20k" if root is not None else get_cache_dir("ade20k")
            dataset_path = root_path / "ADEChallengeData2016"
        else:
            dataset_path = Path(root) if root is not None else get_cache_dir("ade20k")

        if not dataset_path.exists():
            if download:
                downloaded_file = download_dataset(self.URL, root_path, self.MD5)
                extract_dataset(downloaded_file, root_path)
            if not dataset_path.exists():
                raise FileNotFoundError(
                    f"Could not find dataset at '{dataset_path}'. If you set `download=False`, "
                    "make sure the dataset is present. Otherwise ensure write permissions and try again."
                )

        if split not in self.VALID_SPLITS:
            raise ValueError(f"Invalid split '{split}'. Expected one of {self.VALID_SPLITS}.")

        self.split = split
        self.images_dir = (
            dataset_path / "images/training" if self.split == "train" else dataset_path / "images/validation"
        )
        self.masks_dir = (
            dataset_path / "annotations/training" if self.split == "train" else dataset_path / "annotations/validation"
        )
        self.transform = transform
        self.target_transform = target_transform
        self.images = os.listdir(self.images_dir)

        self.num_classes = 150

    def __len__(self):
        """Return the number of images in the dataset.

        Returns:
            int: Number of images in the selected split.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """Get a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image, mask) where image is a PIL Image and mask is the
                segmentation mask.
        """
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.images[idx].replace("jpg", "png"))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.transform is not None:
            image = self.transform(image)  # shape [C, H, W]

        if self.target_transform is not None:
            mask = self.target_transform(mask=mask, ignore_index=None)  # shape [H, W]

        return image, mask
