from torch.utils.data import Dataset

from segmentation_robustness_framework import datasets
from segmentation_robustness_framework.config import DatasetConfig
from segmentation_robustness_framework.utils import image_preprocessing


class DatasetLoader:
    """Loads datasets for image segmentation tasks based on the provided configuration.

    The `DatasetLoader` class initializes and loads a dataset based on its name and configuration,
    applying preprocessing steps for input images and their corresponding segmentation masks.

    Attributes:
        config (DatasetConfig): Configuration specifying the dataset and its parameters.
        dataset_name (str): Name of the dataset to be loaded (e.g., VOC, ADE20K, etc.).
        root (str): Root directory where the dataset is located.
        images_shape (tuple[int, int]): Desired image shape for preprocessing (height, width).
    """
    def __init__(self, dataset_config: DatasetConfig) -> None:
        """Initializes the `DatasetLoader` with the provided dataset configuration.

        Args:
            dataset_config (DatasetConfig): Configuration object specifying the dataset
                name, root directory, split, image shape, and other parameters.
        """

        self.config = dataset_config
        self.dataset_name = self.config.name
        self.root = self.config.root
        self.images_shape = self.config.image_shape

    def load_dataset(self) -> Dataset:
        """Loads and preprocesses the specified dataset.

        Based on the dataset name in the configuration, the corresponding dataset class
        is initialized with appropriate preprocessing transformations applied.

        Returns:
            Dataset: An instance of the dataset class ready for training or evaluation.

        Raises:
            ValueError: If the specified dataset name is not recognized.
        """
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
