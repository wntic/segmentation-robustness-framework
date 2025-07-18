import json
import os
from pathlib import Path
from typing import Callable, Optional, Union

from PIL import Image
from torch.utils.data import Dataset

from segmentation_robustness_framework.datasets.registry import register_dataset


@register_dataset("cityscapes")
class Cityscapes(Dataset):
    """Cityscapes dataset for semantic segmentation.

    Cityscapes is a large-scale dataset for semantic understanding of urban street scenes.
    It contains high-quality pixel-level annotations of 5000 images in 50 cities.

    **Setup Instructions:**

    1. Register at https://www.cityscapes-dataset.com/
    2. Download the dataset files:
       - `leftImg8bit_trainvaltest.zip` (11GB) - training/validation/test images
       - `gtFine_trainval.zip` (241MB) - fine annotations for train/val
       - `gtCoarse.zip` (1.3GB) - coarse annotations for train/val/train_extra
       - `leftImg8bit_trainextra.zip` (44GB) - extra training images (optional)
    3. Extract all archives to the same root directory
    4. Ensure the directory structure matches:
       ```
       root/
       ├── leftImg8bit/
       │   ├── train/
       │   ├── val/
       │   └── test/
       ├── gtFine/
       │   ├── train/
       │   └── val/
       ├── gtCoarse/
       │   ├── train/
       │   ├── val/
       │   └── train_extra/
       └── leftImg8bit_trainextra/
           └── train_extra/
       ```

    **Supported Splits:**
    - `train`: Training images with fine annotations
    - `val`: Validation images with fine annotations
    - `test`: Test images (no annotations available)
    - `train_extra`: Extra training images with coarse annotations

    **Supported Modes:**
    - `fine`: High-quality pixel-level annotations
    - `coarse`: Coarse polygon annotations

    **Supported Target Types:**
    - `semantic`: Semantic segmentation masks
    - `instance`: Instance segmentation masks
    - `color`: Color-coded visualization masks
    - `polygon`: Polygon annotations (JSON format)

    Attributes:
        root (str | Path): Path to the Cityscapes dataset root directory.
        split (str): Dataset split ('train', 'val', 'test', 'train_extra').
        mode (str): Annotation mode ('fine' or 'coarse').
        target_type (str | list): Type of target annotations.
        transform (callable, optional): Image transformations.
        target_transform (callable, optional): Target transformations.
        num_classes (int): Number of semantic classes (35).
    """

    VALID_SPLITS = ["train", "val", "test", "train_extra"]
    VALID_MODES = ["fine", "coarse"]
    VALID_TARGET_TYPES = ["semantic", "instance", "color", "polygon"]

    def __init__(
        self,
        root: Union[Path, str],
        split: str = "train",
        mode: str = "fine",
        target_type: str = "semantic",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        """Initialize Cityscapes dataset.

        Args:
            root (str | Path): Path to the Cityscapes dataset root directory.
                Must contain the extracted dataset files with proper directory structure.
            split (str, optional): Dataset split. Must be one of 'train', 'val', 'test',
                or 'train_extra'. Defaults to "train".
            mode (str, optional): Annotation mode. Must be 'fine' or 'coarse'.
                Defaults to "fine".
            target_type (str | list, optional): Type of target annotations. Can be a
                single type or list of types. Must be one or more of 'semantic',
                'instance', 'color', 'polygon'. Defaults to "semantic".
            transform (callable, optional): Transform to apply to images.
                Defaults to None.
            target_transform (callable, optional): Transform to apply to targets.
                Defaults to None.

        Raises:
            ValueError: If root directory does not exist.
            ValueError: If split is not valid.
            ValueError: If mode is not valid.
            ValueError: If target_type is not valid.
            ValueError: If test split is used with coarse mode.
            ValueError: If train_extra split is used with fine mode.
            ValueError: If required dataset files are missing.
        """
        if not os.path.exists(root):
            raise ValueError(f"Root directory '{root}' does not exist.")

        if split not in self.VALID_SPLITS:
            raise ValueError(f"Invalid split '{split}'. Expected one of {self.VALID_SPLITS}.")

        if mode not in self.VALID_MODES:
            raise ValueError(f"Invalid mode '{mode}'. Expected one of {self.VALID_MODES}.")

        if isinstance(target_type, str):
            target_type = [target_type]
        for t in target_type:
            if t not in self.VALID_TARGET_TYPES:
                raise ValueError(f"Invalid target_type '{t}'. Expected one of {self.VALID_TARGET_TYPES}.")

        if split == "test" and mode == "coarse":
            raise ValueError("The 'test' split is not available for 'coarse' mode. Use 'fine' mode instead.")

        if split == "train_extra" and mode == "fine":
            raise ValueError("The 'train_extra' split is not available for 'fine' mode. Use 'coarse' mode instead.")

        self.root = root
        self.split = split
        self.mode = "gtFine" if mode == "fine" else "gtCoarse"
        self.target_type = target_type
        self.transform = transform
        self.target_transform = target_transform

        if split == "train_extra":
            self.images_dir = os.path.join(root, "leftImg8bit_trainextra", split)
            if not os.path.exists(self.images_dir):
                raise ValueError(
                    f"Directory '{self.images_dir}' does not exist. Download 'leftImg8bit_trainextra.zip' and extract to the {root} directory or use another split ('train or 'val)"
                )
        else:
            self.images_dir = os.path.join(root, "leftImg8bit", split)
        self.targets_dir = os.path.join(root, self.mode, split)

        self.images = []
        self.targets = []

        for city in os.listdir(self.images_dir):
            city_image_dir = os.path.join(self.images_dir, city)
            city_target_dir = os.path.join(self.targets_dir, city)

            for file_name in os.listdir(city_image_dir):
                image_path = os.path.join(city_image_dir, file_name)
                self.images.append(image_path)

                target_paths = []
                for t in self.target_type:
                    target_file_name = file_name.replace(
                        "_leftImg8bit.png", f"_{self.mode}_{self._get_target_postfix(t)}"
                    )

                    target_path = os.path.join(city_target_dir, target_file_name)
                    target_paths.append(target_path)

                self.targets.append(target_paths)

        self.num_classes = 35

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
            tuple: (image, target) where image is a PIL Image and target is the
                annotation(s) in the format specified by target_type.
        """
        image = Image.open(self.images[idx]).convert("RGB")

        targets = []
        for target_file, t_type in zip(self.targets[idx], self.target_type):
            if t_type == "polygon":
                target = self._load_json(target_file)
            else:
                target = Image.open(target_file).convert("L")
            targets.append(target)

        target = targets[0] if len(targets) == 1 else targets

        if self.transform:
            image = self.transform(image)

        if self.target_transform:
            if isinstance(target, list):
                target = [self.target_transform(t) if not isinstance(t, dict) else t for t in target]
            else:
                target = self.target_transform(target)

        return image, target

    def _get_target_postfix(self, target: str) -> str:
        """Get the file extension for a given target type.

        Args:
            target (str): Target type ('semantic', 'instance', 'color', 'polygon').

        Returns:
            str: File extension for the target type.
        """
        if target == "semantic":
            return "labelIds.png"
        elif target == "instance":
            return "instanceIds.png"
        elif target == "color":
            return "color.png"
        elif target == "polygon":
            return "polygons.json"

    def _load_json(self, path: str) -> dict:
        """Load JSON file containing polygon annotations.

        Args:
            path (str): Path to the JSON file.

        Returns:
            dict: Loaded JSON data.
        """
        with open(path) as f:
            data = json.load(f)
        return data
