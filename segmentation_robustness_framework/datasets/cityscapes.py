import json
import os
from pathlib import Path
from typing import Callable, Optional, Union

from PIL import Image
from torch.utils.data import Dataset

from segmentation_robustness_framework.datasets.registry import register_dataset


@register_dataset("cityscapes")
class Cityscapes(Dataset):
    """Cityscapes Dataset.

    Attributes:

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
        """
        Args:
            root (str): Корневая директория датасета Cityscapes.
            split (str): 'train', 'val', 'test' или 'train_extra'. Определяет, какое подмножество данных используется.
            mode (str): 'fine' или 'coarse'. Определяет тип аннотаций (точные или грубые).
            target_type (str or list): 'semantic', 'instance', 'polygon', 'color' или их комбинация.
                                       Определяет тип целевых данных (например, семантические маски).
            transform (callable, optional): Опциональные преобразования, применяемые к изображениям и маскам.
        """

        """_summary_

        Args:
            root (Union[Path, str]): Path to root directory with Cityscapes.
            split (str, optional): 'train', 'val', 'test' or 'train_extra'. Determines which subset of data is used.
                Defaults to "train".
            mode (str, optional): 'fine' or 'coarse'. Defaults to "fine".
            target_type (str, optional): _description_. Defaults to "semantic".
            transform (Optional[Callable], optional): _description_. Defaults to None.
            target_transform (Optional[Callable], optional): _description_. Defaults to None.

        Raises:
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
            ValueError: _description_
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
        return len(self.images)

    def __getitem__(self, idx):
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
            image = self.transform(image).unsqueeze(0)

        if self.target_transform:
            if isinstance(target, list):
                target = [self.target_transform(t) if not isinstance(t, dict) else t for t in target]
            else:
                target = self.target_transform(target)

        return image, target

    def _get_target_postfix(self, target: str):
        if target == "semantic":
            return "labelIds.png"
        elif target == "instance":
            return "instanceIds.png"
        elif target == "color":
            return "color.png"
        elif target == "polygon":
            return "polygons.json"

    def _load_json(self, path):
        with open(path) as f:
            data = json.load(f)
        return data
