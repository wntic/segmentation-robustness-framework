import os
from pathlib import Path
from typing import Callable, Optional, Union

from PIL import Image
from torch.utils.data import Dataset

from segmentation_robustness_framework.datasets.registry import register_dataset
from segmentation_robustness_framework.utils.dataset import download as download_dataset
from segmentation_robustness_framework.utils.dataset import extract as extract_dataset


@register_dataset("stanford_background")
class StanfordBackground(Dataset):
    """Stanford Background Dataset.
    From: https://www.kaggle.com/datasets/balraj98/stanford-background-dataset

    Attributes:
        root (str): Path to dataset.
        transform (callable): Images transofrm.
        target_transform (callable): Masks transform.
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
            root (str | Path | None, optional): Directory **into which** the archive will
                be downloaded and extracted, or a directory that already contains the
                dataset files. If ``None`` (default), a cache directory is used.
            transform (callable, optional): Images transform.
            target_transform (callable, optional): Masks transform.
            download (bool, optional): If `True`, downloads the dataset from the internet and
                puts it in `root` directory. If `False`, it assumes that `root`
                already contains the dataset files.
        """
        from segmentation_robustness_framework.utils.dataset import get_cache_dir

        root_path = Path(root) / "stanford_background" if root is not None else get_cache_dir("stanford_background")
        dataset_path = root_path / "stanford_background"

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
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.images[idx])
        mask_path = os.path.join(self.masks_dir, self.images[idx].replace("jpg", "png"))

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")

        if self.transform is not None:
            image = self.transform(image).unsqueeze(0)  # shape [1, C, H, W]

        if self.target_transform is not None:
            mask = self.target_transform(mask)  # shape [C, H, W]

        return image, mask
