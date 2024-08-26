import os
from pathlib import Path
from typing import Callable, Optional, Union

from PIL import Image
from torch.utils.data import Dataset


class ADE20K(Dataset):
    """ADE20K Dataset.
    From: https://data.csail.mit.edu/places/ADEchallenge/ADEChallengeData2016.zip

    Attributes:
        root (str): path/to/ADEChallengeData2016/
        split (str): Set of images. Must be "train" or "val"
        transform (callable): Images transofrm.
        target_transform (callable): Masks transform.
    """

    VALID_SPLITS = ["train", "val"]

    def __init__(
        self,
        root: Union[Path, str],
        split: str,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        """Initialize Pascal VOC 2012 dataset.

        Args:
            root (str): path/to/ADEChallengeData2016/
            split (str): Set of images. Must be "train" or "val"
            transform (callable, optional): Images transofrm. Defaults to None.
            target_transform (callable, optional): Masks transform. Defaults to None.
        """
        if not os.path.exists(root):
            raise ValueError(f"Root directory '{root}' does not exist.")

        if split not in self.VALID_SPLITS:
            raise ValueError(f"Invalid split '{split}'. Expected one of {self.VALID_SPLITS}.")

        self.split = split
        self.images_dir = (
            os.path.join(root, "images/training") if self.split == "train" else os.path.join(root, "images/validation")
        )
        self.masks_dir = (
            os.path.join(root, "annotations/training")
            if self.split == "train"
            else os.path.join(root, "annotations/validation")
        )
        self.transform = transform
        self.target_transform = target_transform
        self.images = os.listdir(self.images_dir)

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
