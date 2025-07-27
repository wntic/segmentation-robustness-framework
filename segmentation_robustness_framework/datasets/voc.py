import os
from pathlib import Path
from typing import Callable, Optional, Union

from PIL import Image
from torch.utils.data import Dataset

from segmentation_robustness_framework.datasets.registry import register_dataset
from segmentation_robustness_framework.utils.dataset_utils import download as download_dataset
from segmentation_robustness_framework.utils.dataset_utils import extract as extract_dataset


@register_dataset("voc")
class VOCSegmentation(Dataset):
    """Pascal VOC 2012 dataset for semantic segmentation.

    The Pascal VOC 2012 dataset contains 21 classes of objects in natural scenes.
    Images are paired with pixel-level segmentation masks for training and evaluation.

    **Setup Instructions:**

    The dataset will be automatically downloaded and extracted if not present.
    When `download=True` (default):
        - If `root` is provided, the dataset will be stored at `root/voc/VOCdevkit/VOC2012/`.
        - If `root` is `None`, the dataset will be cached in the default cache directory.
    When `download=False`:
        - The dataset must be present at the exact path specified by `root`.
        - If `root` is `None`, the dataset will be looked for in the default cache directory.

    **Supported Splits:**
    - `train`: Training images (1,464 samples)
    - `val`: Validation images (1,449 samples)
    - `trainval`: Combined train and validation (2,913 samples)

    Attributes:
        root (str | Path | None): Directory for dataset storage or cache location.
        split (str): Dataset split ('train', 'val', 'trainval').
        transform (callable, optional): Image transformations.
        target_transform (callable, optional): Target transformations.
        download (bool): Whether to download dataset if not present.
        num_classes (int): Number of semantic classes (21).
    """

    URL = "http://host.robots.ox.ac.uk/pascal/VOC/voc2012/VOCtrainval_11-May-2012.tar"
    MD5 = "6cd6e144f989b92b3379bac3b3de84fd"
    VALID_SPLITS = ["train", "val", "trainval"]

    def __init__(
        self,
        split: str,
        root: Optional[Union[Path, str]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
    ) -> None:
        """Initialize Pascal VOC 2012 dataset.

        Args:
            split (str): Dataset split. Must be one of 'train', 'val', or 'trainval'.
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
            root_path = Path(root) / "voc" if root is not None else get_cache_dir("voc")
            dataset_root = root_path / "VOCdevkit" / "VOC2012"
        else:
            dataset_root = Path(root) if root is not None else get_cache_dir("voc")

        if not dataset_root.exists():
            if download:
                downloaded_file = download_dataset(self.URL, root_path, self.MD5)
                extract_dataset(downloaded_file, root_path)
            if not dataset_root.exists():
                raise FileNotFoundError(
                    f"Could not find dataset at '{dataset_root}'. If you set `download=False`, "
                    "make sure the dataset is present. Otherwise ensure write permissions and try again."
                )

        if split not in self.VALID_SPLITS:
            raise ValueError(f"Invalid split '{split}'. Expected one of {self.VALID_SPLITS}.")

        self.images_dir = dataset_root / "JPEGImages"
        self.masks_dir = dataset_root / "SegmentationClass"
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        with open(dataset_root / "ImageSets" / "Segmentation" / f"{split}.txt") as f:
            self.images = f.read().splitlines()

        self.num_classes = 21

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
        img_path = os.path.join(self.images_dir, f"{self.images[idx]}.jpg")
        mask_path = os.path.join(self.masks_dir, f"{self.images[idx]}.png")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.transform is not None:
            image = self.transform(image)  # shape [C, H, W]

        if self.target_transform is not None:
            mask = self.target_transform(mask=mask, ignore_index=255)  # shape [H, W]

        return image, mask
