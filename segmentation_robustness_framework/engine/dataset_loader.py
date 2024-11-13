from segmentation_robustness_framework import datasets
from segmentation_robustness_framework.utils import image_preprocessing
from segmentation_robustness_framework.config import DatasetConfig


class DatasetLoader:
    def __init__(self, dataset_config: DatasetConfig) -> None:
        self.config = dataset_config
        self.dataset_name = self.config.name
        self.root = self.config.root
        self.images_shape = self.config.image_shape

    def load_dataset(self):
        preprocess, target_preprocess = image_preprocessing.get_preprocessing_fn(self.config.image_shape)

        if self.dataset_name == "VOC":
            ds = datasets.VOCSegmentation(
                root=self.root,
                split=self.config.split,
                transform=preprocess,
                target_transform=target_preprocess,
            )
        elif self.dataset_name == "StanfordBackground":
            ds = datasets.StanfordBackground(
                root=self.root,
                transform=preprocess,
                target_transform=target_preprocess,
            )
        elif self.dataset_name == "ADE20K":
            ds = datasets.ADE20K(
                root=self.root,
                split=self.config.split,
                transform=preprocess,
                target_transform=target_preprocess,
            )
        elif self.dataset_name == "Cityscapes":
            ds = datasets.Cityscapes(
                root=self.root,
                split=self.config.split,
                mode=self.config.mode,
                target_type=self.config.target_type,
                transform=preprocess,
                target_transform=target_preprocess,
            )
        else:
            raise ValueError(f"Unknown dataset: {self.dataset_name}")

        return ds
