import os
from pathlib import Path
from typing import Callable, Optional, Union

from PIL import Image
from torch.utils.data import Dataset

from segmentation_robustness_framework.datasets.registry import register_dataset
from segmentation_robustness_framework.utils.dataset_utils import download as download_dataset
from segmentation_robustness_framework.utils.dataset_utils import extract as extract_dataset


@register_dataset("stanford_background")
class StanfordBackground(Dataset):
    """Stanford Background dataset for semantic segmentation.

    The Stanford Background dataset contains 715 images with 9 semantic categories.
    Images are paired with pixel-level segmentation masks for training and evaluation.

    **Setup Instructions:**

    The dataset will be automatically downloaded and extracted if not present.
    When `download=True` (default):
        - If `root` is provided, the dataset will be stored at `root/stanford_background/stanford_background/`.
        - If `root` is `None`, the dataset will be cached in the default cache directory.
    When `download=False`:
        - The dataset must be present at the exact path specified by `root`.
        - If `root` is `None`, the dataset will be looked for in the default cache directory.

    **Dataset Structure:**
    - `images/`: Input RGB images
    - `labels_colored/`: Segmentation masks (color images)

    Attributes:
        root (str | Path | None): Directory for dataset storage or cache location.
        transform (callable, optional): Image transformations.
        target_transform (callable, optional): Target transformations.
        download (bool): Whether to download dataset if not present.
        num_classes (int): Number of semantic classes (9).
    """

    URL = "https://www.kaggle.com/api/v1/datasets/download/balraj98/stanford-background-dataset"
    MD5 = "8932f1c0de6304734b0daa15fcd55f48"

    def __init__(
        self,
        root: Optional[Union[Path, str]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
    ) -> None:
        """Initialize Stanford Background dataset.

        Args:
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
        """
        from segmentation_robustness_framework.utils.dataset_utils import get_cache_dir

        if download:
            root_path = Path(root) / "stanford_background" if root is not None else get_cache_dir("stanford_background")
            dataset_path = root_path / "stanford_background"
        else:
            dataset_path = Path(root) if root is not None else get_cache_dir("stanford_background")

        if not dataset_path.exists():
            if download:
                downloaded_file = download_dataset(self.URL, root_path, self.MD5)
                extract_dataset(downloaded_file, dataset_path)
            if not dataset_path.exists():
                raise FileNotFoundError(
                    f"Could not find dataset at '{dataset_path}'. If you set `download=False`, "
                    "make sure the dataset is present. Otherwise ensure write permissions and try again."
                )

        self.images_dir = dataset_path / "images"
        self.masks_dir = dataset_path / "labels_colored"
        self.transform = transform
        self.target_transform = target_transform
        self.images = os.listdir(self.images_dir)

        self.num_classes = 9

    def __len__(self):
        """Return the number of images in the dataset.

        Returns:
            int: Number of images in the dataset.
        """
        return len(self.images)

    def __getitem__(self, idx):
        """Get a single sample from the dataset.

        Args:
            idx (int): Index of the sample to retrieve.

        Returns:
            tuple: (image, mask) where image is a tensor [C, H, W] and mask is a tensor [H, W] with values in [0, 8].
        """
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.images[idx].replace("jpg", "png"))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("RGB")

        if self.transform is not None:
            image = self.transform(image)  # shape [C, H, W]

        if self.target_transform is not None:
            mask = self.target_transform(mask)  # shape [H, W]

        return image, mask
