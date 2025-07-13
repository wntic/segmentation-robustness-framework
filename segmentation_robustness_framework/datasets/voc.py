import os
from pathlib import Path
from typing import Callable, Optional, Union

from PIL import Image
from torch.utils.data import Dataset

from segmentation_robustness_framework.datasets.registry import register_dataset
from segmentation_robustness_framework.utils.dataset import download as download_dataset
from segmentation_robustness_framework.utils.dataset import extract as extract_dataset


@register_dataset("VOC")
class VOCSegmentation(Dataset):
    """Pascal VOC 2012 Dataset.
    From: http://host.robots.ox.ac.uk/pascal/VOC/

    Attributes:
        root (str): Directory **into which** the archive will be downloaded and
            extracted *or* a directory that already contains the
            `VOCdevkit/VOC2012` hierarchy.  For example, if you pass
            `data/`, the dataset will end up at
            `data/VOCdevkit/VOC2012/...`.
        split (str): Set of images. Must be `"train"`, `"val"` or `"trainval"`.
        transform (callable): Images transform.
        target_transform (callable): Masks transform.
        download (bool): If `True`, downloads the dataset from the internet and
            puts it in `root` directory. If `False`, it assumes that `root`
            already contains the `VOCdevkit/VOC2012` hierarchy.
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
            split (str): Set of images. Must be `"train"`, `"val"` or `"trainval"`.
            root (str | Path | None, optional): Directory **into which** the archive will
                be downloaded and extracted, or a directory that already contains the
                ``VOCdevkit/VOC2012`` hierarchy.  If ``None`` (default), a cache
                directory is used (see Notes).
            transform (callable, optional): Images transform. Defaults to None.
            target_transform (callable, optional): Masks transform. Defaults to None.
            download (bool, optional): If `True`, downloads the dataset from the internet and
                puts it in `root` directory. If `False`, it assumes that `root`
                already contains the `VOCdevkit/VOC2012` hierarchy.
        """
        from segmentation_robustness_framework.utils.dataset import get_cache_dir

        root_path = Path(root) / "voc" if root is not None else get_cache_dir("voc")
        dataset_root = root_path / "VOCdevkit" / "VOC2012"

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
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, f"{self.images[idx]}.jpg")
        mask_path = os.path.join(self.masks_dir, f"{self.images[idx]}.png")

        image = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path)

        if self.transform is not None:
            image = self.transform(image).unsqueeze(0)  # shape [1, C, H, W]

        if self.target_transform is not None:
            mask = self.target_transform(mask=mask, ignore_index=255)  # shape [C, H, W]

        return image, mask
