import os
from pathlib import Path
from typing import Callable, Optional, Union

from PIL import Image
from torch.utils.data import Dataset

from segmentation_robustness_framework.datasets.registry import register_dataset
from segmentation_robustness_framework.utils.dataset import download as download_dataset
from segmentation_robustness_framework.utils.dataset import extract as extract_dataset


@register_dataset("ade20k")
class ADE20K(Dataset):
    """ADE20K Dataset.
    From: https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip

    Attributes:
        root (str | Path | None, optional): Directory **into which** the archive will
            be downloaded and extracted, or a directory that already contains the
            dataset files. If ``None`` (default), a cache directory is used.
        split (str): Set of images. Must be "train" or "val"
        transform (callable): Images transform.
        target_transform (callable): Masks transform.
        download (bool): If `True`, downloads the dataset from the internet and
            puts it in `root` directory. If `False`, it assumes that `root`
            already contains the dataset files.
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
            split (str): Set of images. Must be "train" or "val"
            root (str | Path | None, optional): Directory **into which** the archive will
                be downloaded and extracted, or a directory that already contains the
                dataset files. If ``None`` (default), a cache directory is used.
            transform (callable, optional): Images transform. Defaults to None.
            target_transform (callable, optional): Masks transform. Defaults to None.
            download (bool, optional): If `True`, downloads the dataset from the internet and
                puts it in `root` directory. If `False`, it assumes that `root`
                already contains the dataset files.
        """
        from segmentation_robustness_framework.utils.dataset import get_cache_dir

        root_path = Path(root) / "ade20k" if root is not None else get_cache_dir("ade20k")
        dataset_path = root_path / "ADEChallengeData2016"

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
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.images[idx].replace("jpg", "png"))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.transform is not None:
            image = self.transform(image).unsqueeze(0)  # shape [1, C, H, W]

        if self.target_transform is not None:
            mask = self.target_transform(mask=mask, ignore_index=None)  # shape [C, H, W]

        return image, mask
