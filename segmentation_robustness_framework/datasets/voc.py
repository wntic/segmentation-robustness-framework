import os
from pathlib import Path
from typing import Callable, Optional, Union

from PIL import Image
from torch.utils.data import Dataset


class VOCSegmentation(Dataset):
    """Pascal VOC 2012 Dataset.
    From: http://host.robots.ox.ac.uk/pascal/VOC/

    Attributes:
        root (str): path/to/VOCdevkit/VOC2012/
        split (str): Set of images. Must be "train", "val" or "trainval".
        transform (callable): Images transofrm.
        target_transform (callable): Masks transform.
    """

    VALID_SPLITS = ["train", "val", "trainval"]

    def __init__(
        self,
        root: Union[Path, str],
        split: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        """Initialize Pascal VOC 2012 dataset.

        Args:
            root (str): path/to/VOCdevkit/VOC2012/
            split (str): Set of images. Must be "train", "val" or "trainval".
            transform (callable, optional): Images transofrm. Defaults to None.
            target_transform (callable, optional): Masks transform. Defaults to None.
        """
        if not os.path.exists(root):
            raise ValueError(f"Root directory '{root}' does not exist.")

        if split not in self.VALID_SPLITS:
            raise ValueError(f"Invalid split '{split}'. Expected one of {self.VALID_SPLITS}.")

        self.images_dir = os.path.join(root, "JPEGImages")
        self.masks_dir = os.path.join(root, "SegmentationClass")
        self.split = split
        self.transform = transform
        self.target_transform = target_transform

        with open(os.path.join(root, "ImageSets/Segmentation", f"{split}.txt")) as f:
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
