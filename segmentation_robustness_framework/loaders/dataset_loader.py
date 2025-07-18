import inspect
from typing import Any

from torch.utils.data import Dataset

from segmentation_robustness_framework.datasets.registry import DATASET_REGISTRY
from segmentation_robustness_framework.utils import image_preprocessing


class DatasetLoader:
    """Load and configure datasets for image segmentation tasks.

    The `DatasetLoader` initializes and loads a dataset by name using the provided configuration,
    and applies preprocessing to input images and segmentation masks.

    Supported attributes:
        - `config` (dict[str, Any]): Configuration specifying the dataset and its parameters.
        - `dataset_name` (str): Name of the dataset to be loaded (e.g., `VOC`, `ADE20K`).
        - `root` (str): Root directory where the dataset is located.
        - `images_shape` (list[int]): Desired image shape for preprocessing [height, width].

    Example:
        ```python
        loader = DatasetLoader({
            "name": "VOCSegmentation",
            "root": "/path/to/voc",
            "image_shape": [256, 256],
            "split": "train",
        })
        dataset = loader.load_dataset()
        ```
    """

    def __init__(self, dataset_config: dict[str, Any]) -> None:
        """Initialize the DatasetLoader with a dataset configuration.

        Args:
            dataset_config (dict[str, Any]):
                - `name` (str): Dataset name.
                - `root` (str): Root directory of the dataset.
                - `image_shape` (list[int]): Desired image shape.
                - Additional dataset-specific parameters.
        """

        self.config = dataset_config
        self.dataset_name = self.config["name"]
        self.root = self.config["root"]
        self.images_shape = self.config["image_shape"]

    def load_dataset(self) -> Dataset:
        """Loads and preprocesses the specified dataset.

        Based on the dataset name in the configuration, the corresponding dataset class
        is initialized with appropriate preprocessing transformations applied.

        Returns:
            Dataset: An instance of the dataset class ready for training or evaluation.

        Raises:
            ValueError: If the specified dataset name is not recognized.
        """
        preprocess, target_preprocess = image_preprocessing.get_preprocessing_fn(self.images_shape, self.dataset_name)

        try:
            ds_cls = DATASET_REGISTRY[self.dataset_name]
        except KeyError:
            raise ValueError(f"Unknown dataset: {self.dataset_name}. Available: {list(DATASET_REGISTRY.keys())}")

        sig = inspect.signature(ds_cls)
        common_kwargs = dict(root=self.root, transform=preprocess, target_transform=target_preprocess)

        extra = {k: v for k, v in self.config.items() if k in sig.parameters}
        common_kwargs.update(extra)

        return ds_cls(**common_kwargs)
