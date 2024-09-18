import os
from pathlib import Path
from typing import Callable, Optional, Union

from PIL import Image
from torch.utils.data import Dataset


class StanfordBackground(Dataset):
    """Stanford Background Dataset.
    From: https://www.kaggle.com/datasets/balraj98/stanford-background-dataset

    Attributes:
        root (str): Path to dataset.
        transform (callable): Images transofrm.
        target_transform (callable): Masks transform.
    """

    def __init__(
        self,
        root: Union[Path, str],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        """Initialize Stanford Background dataset.

        Args:
            root (str): Path to dataset.
            transform (callable, optional): Images transofrm.
            target_transform (callable, optional): Masks transform.
        """
        if not os.path.exists(root):
            raise ValueError(f"Root directory '{root}' does not exist.")

        self.images_dir = os.path.join(root, "images")
        self.masks_dir = os.path.join(root, "labels_colored")
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
