from unittest.mock import Mock, patch

import pytest
import torch
from segmentation_robustness_framework.loaders.dataset_loader import DatasetLoader
from torch.utils.data import Dataset


class MockDataset(Dataset):
    def __init__(
        self, root: str, transform=None, target_transform=None, split: str = "train", download: bool = False, **kwargs
    ):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.split = split
        self.download = download
        self.num_classes = 21
        self.length = 100

        for key, value in kwargs.items():
            setattr(self, key, value)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        image = torch.randn(3, 256, 256)
        mask = torch.randint(0, self.num_classes, (256, 256))
        return image, mask


@pytest.fixture
def mock_dataset_config():
    return {
        "name": "test_dataset",
        "root": "/path/to/dataset",
        "image_shape": [256, 256],
        "split": "train",
        "download": True,
    }


@pytest.fixture
def mock_dataset_config_minimal():
    return {"name": "test_dataset", "root": "/path/to/dataset", "image_shape": [512, 512]}


@pytest.fixture
def mock_dataset_config_with_extra_params():
    return {
        "name": "test_dataset",
        "root": "/path/to/dataset",
        "image_shape": [128, 128],
        "split": "val",
        "download": False,
        "custom_param": "custom_value",
        "num_workers": 4,
    }


def test_dataset_loader_initialization(mock_dataset_config):
    loader = DatasetLoader(mock_dataset_config)

    assert loader.config == mock_dataset_config
    assert loader.dataset_name == "test_dataset"
    assert loader.root == "/path/to/dataset"
    assert loader.images_shape == [256, 256]


def test_dataset_loader_initialization_minimal(mock_dataset_config_minimal):
    loader = DatasetLoader(mock_dataset_config_minimal)

    assert loader.config == mock_dataset_config_minimal
    assert loader.dataset_name == "test_dataset"
    assert loader.root == "/path/to/dataset"
    assert loader.images_shape == [512, 512]


def test_dataset_loader_initialization_with_extra_params(mock_dataset_config_with_extra_params):
    loader = DatasetLoader(mock_dataset_config_with_extra_params)

    assert loader.config == mock_dataset_config_with_extra_params
    assert loader.dataset_name == "test_dataset"
    assert loader.root == "/path/to/dataset"
    assert loader.images_shape == [128, 128]


def test_dataset_loader_missing_required_config_keys():
    incomplete_config = {
        "name": "test_dataset",
    }

    with pytest.raises(KeyError, match="root"):
        DatasetLoader(incomplete_config)

    incomplete_config2 = {
        "name": "test_dataset",
        "root": "/path/to/dataset",
    }

    with pytest.raises(KeyError, match="image_shape"):
        DatasetLoader(incomplete_config2)


@patch("segmentation_robustness_framework.loaders.dataset_loader.image_preprocessing.get_preprocessing_fn")
@patch("segmentation_robustness_framework.loaders.dataset_loader.DATASET_REGISTRY")
def test_load_dataset_success(mock_registry, mock_get_preprocessing_fn, mock_dataset_config):
    mock_preprocess = Mock()
    mock_target_preprocess = Mock()
    mock_get_preprocessing_fn.return_value = (mock_preprocess, mock_target_preprocess)

    mock_registry.__getitem__.return_value = MockDataset

    loader = DatasetLoader(mock_dataset_config)
    dataset = loader.load_dataset()

    mock_get_preprocessing_fn.assert_called_once_with([256, 256], "test_dataset")

    assert isinstance(dataset, MockDataset)
    assert dataset.root == "/path/to/dataset"
    assert dataset.transform == mock_preprocess
    assert dataset.target_transform == mock_target_preprocess
    assert dataset.split == "train"
    assert dataset.download is True


@patch("segmentation_robustness_framework.loaders.dataset_loader.image_preprocessing.get_preprocessing_fn")
@patch("segmentation_robustness_framework.loaders.dataset_loader.DATASET_REGISTRY")
def test_load_dataset_with_extra_parameters(
    mock_registry, mock_get_preprocessing_fn, mock_dataset_config_with_extra_params
):
    mock_preprocess = Mock()
    mock_target_preprocess = Mock()
    mock_get_preprocessing_fn.return_value = (mock_preprocess, mock_target_preprocess)

    mock_registry.__getitem__.return_value = MockDataset

    loader = DatasetLoader(mock_dataset_config_with_extra_params)
    dataset = loader.load_dataset()

    assert isinstance(dataset, MockDataset)
    assert dataset.root == "/path/to/dataset"
    assert dataset.transform == mock_preprocess
    assert dataset.target_transform == mock_target_preprocess
    assert dataset.split == "val"
    assert dataset.download is False
    assert not hasattr(dataset, "custom_param")
    assert not hasattr(dataset, "num_workers")


@patch("segmentation_robustness_framework.loaders.dataset_loader.image_preprocessing.get_preprocessing_fn")
@patch("segmentation_robustness_framework.loaders.dataset_loader.DATASET_REGISTRY")
def test_load_dataset_unknown_dataset(mock_registry, mock_get_preprocessing_fn, mock_dataset_config):
    mock_get_preprocessing_fn.return_value = (Mock(), Mock())

    mock_registry.__getitem__.side_effect = KeyError("test_dataset")
    mock_registry.keys.return_value = ["voc", "ade20k", "cityscapes"]

    loader = DatasetLoader(mock_dataset_config)

    with pytest.raises(ValueError, match="Unknown dataset: test_dataset"):
        loader.load_dataset()


@patch("segmentation_robustness_framework.loaders.dataset_loader.image_preprocessing.get_preprocessing_fn")
@patch("segmentation_robustness_framework.loaders.dataset_loader.DATASET_REGISTRY")
def test_load_dataset_parameter_filtering(
    mock_registry, mock_get_preprocessing_fn, mock_dataset_config_with_extra_params
):
    mock_preprocess = Mock()
    mock_target_preprocess = Mock()
    mock_get_preprocessing_fn.return_value = (mock_preprocess, mock_target_preprocess)

    class SpecificDataset(MockDataset):
        def __init__(
            self, root: str, transform=None, target_transform=None, split: str = "train", download: bool = False
        ):
            super().__init__(root, transform, target_transform, split, download=download)

    mock_registry.__getitem__.return_value = SpecificDataset

    loader = DatasetLoader(mock_dataset_config_with_extra_params)
    dataset = loader.load_dataset()

    assert isinstance(dataset, SpecificDataset)
    assert dataset.root == "/path/to/dataset"
    assert dataset.transform == mock_preprocess
    assert dataset.target_transform == mock_target_preprocess
    assert dataset.split == "val"
    assert dataset.download is False

    assert not hasattr(dataset, "custom_param")
    assert not hasattr(dataset, "num_workers")


@patch("segmentation_robustness_framework.loaders.dataset_loader.image_preprocessing.get_preprocessing_fn")
@patch("segmentation_robustness_framework.loaders.dataset_loader.DATASET_REGISTRY")
def test_load_dataset_inspect_signature_handling(mock_registry, mock_get_preprocessing_fn, mock_dataset_config):
    mock_preprocess = Mock()
    mock_target_preprocess = Mock()
    mock_get_preprocessing_fn.return_value = (mock_preprocess, mock_target_preprocess)

    class SignatureTestDataset(MockDataset):
        def __init__(self, root: str, transform=None, target_transform=None, split: str = "train"):
            super().__init__(root, transform, target_transform, split)

    mock_registry.__getitem__.return_value = SignatureTestDataset

    config_with_extra = mock_dataset_config.copy()
    config_with_extra["invalid_param"] = "should_not_be_passed"

    loader = DatasetLoader(config_with_extra)
    dataset = loader.load_dataset()

    assert isinstance(dataset, SignatureTestDataset)
    assert not hasattr(dataset, "invalid_param")


@patch("segmentation_robustness_framework.loaders.dataset_loader.image_preprocessing.get_preprocessing_fn")
@patch("segmentation_robustness_framework.loaders.dataset_loader.DATASET_REGISTRY")
def test_load_dataset_different_image_shapes(mock_registry, mock_get_preprocessing_fn):
    mock_preprocess = Mock()
    mock_target_preprocess = Mock()
    mock_get_preprocessing_fn.return_value = (mock_preprocess, mock_target_preprocess)

    mock_registry.__getitem__.return_value = MockDataset

    test_shapes = [[64, 64], [128, 128], [256, 256], [512, 512], [1024, 1024]]

    for shape in test_shapes:
        config = {"name": "test_dataset", "root": "/path/to/dataset", "image_shape": shape}

        loader = DatasetLoader(config)
        dataset = loader.load_dataset()

        mock_get_preprocessing_fn.assert_called_with(shape, "test_dataset")
        assert isinstance(dataset, MockDataset)


@patch("segmentation_robustness_framework.loaders.dataset_loader.image_preprocessing.get_preprocessing_fn")
@patch("segmentation_robustness_framework.loaders.dataset_loader.DATASET_REGISTRY")
def test_load_dataset_different_dataset_names(mock_registry, mock_get_preprocessing_fn, mock_dataset_config):
    mock_preprocess = Mock()
    mock_target_preprocess = Mock()
    mock_get_preprocessing_fn.return_value = (mock_preprocess, mock_target_preprocess)

    mock_registry.__getitem__.return_value = MockDataset

    test_names = ["voc", "ade20k", "cityscapes", "stanford_background", "custom_dataset"]

    for name in test_names:
        config = mock_dataset_config.copy()
        config["name"] = name

        loader = DatasetLoader(config)
        dataset = loader.load_dataset()

        mock_get_preprocessing_fn.assert_called_with([256, 256], name)
        assert isinstance(dataset, MockDataset)


def test_dataset_loader_config_immutability(mock_dataset_config):
    original_config = mock_dataset_config.copy()
    loader = DatasetLoader(mock_dataset_config)

    assert mock_dataset_config == original_config

    assert loader.config == mock_dataset_config
    assert loader.dataset_name == "test_dataset"
    assert loader.root == "/path/to/dataset"
    assert loader.images_shape == [256, 256]


@patch("segmentation_robustness_framework.loaders.dataset_loader.image_preprocessing.get_preprocessing_fn")
@patch("segmentation_robustness_framework.loaders.dataset_loader.DATASET_REGISTRY")
def test_load_dataset_preprocessing_functions_passed_correctly(
    mock_registry, mock_get_preprocessing_fn, mock_dataset_config
):
    mock_preprocess = Mock()
    mock_target_preprocess = Mock()
    mock_get_preprocessing_fn.return_value = (mock_preprocess, mock_target_preprocess)

    mock_registry.__getitem__.return_value = MockDataset

    loader = DatasetLoader(mock_dataset_config)
    dataset = loader.load_dataset()

    assert dataset.transform == mock_preprocess
    assert dataset.target_transform == mock_target_preprocess

    assert callable(dataset.transform)
    assert callable(dataset.target_transform)


@patch("segmentation_robustness_framework.loaders.dataset_loader.image_preprocessing.get_preprocessing_fn")
@patch("segmentation_robustness_framework.loaders.dataset_loader.DATASET_REGISTRY")
def test_load_dataset_with_complex_parameters(mock_registry, mock_get_preprocessing_fn):
    mock_preprocess = Mock()
    mock_target_preprocess = Mock()
    mock_get_preprocessing_fn.return_value = (mock_preprocess, mock_target_preprocess)

    class ComplexDataset(MockDataset):
        def __init__(
            self,
            root: str,
            transform=None,
            target_transform=None,
            split: str = "train",
            download: bool = False,
            num_workers: int = 0,
            pin_memory: bool = True,
            drop_last: bool = False,
            shuffle: bool = True,
        ):
            super().__init__(root, transform, target_transform, split, download=download)
            self.num_workers = num_workers
            self.pin_memory = pin_memory
            self.drop_last = drop_last
            self.shuffle = shuffle

    mock_registry.__getitem__.return_value = ComplexDataset

    config = {
        "name": "complex_dataset",
        "root": "/path/to/dataset",
        "image_shape": [256, 256],
        "split": "val",
        "download": True,
        "num_workers": 4,
        "pin_memory": False,
        "drop_last": True,
        "shuffle": False,
    }

    loader = DatasetLoader(config)
    dataset = loader.load_dataset()

    assert isinstance(dataset, ComplexDataset)
    assert dataset.split == "val"
    assert dataset.download is True
    assert dataset.num_workers == 4
    assert dataset.pin_memory is False
    assert dataset.drop_last is True
    assert dataset.shuffle is False


@patch("segmentation_robustness_framework.loaders.dataset_loader.image_preprocessing.get_preprocessing_fn")
@patch("segmentation_robustness_framework.loaders.dataset_loader.DATASET_REGISTRY")
def test_load_dataset_error_handling_preprocessing(mock_registry, mock_get_preprocessing_fn, mock_dataset_config):
    mock_get_preprocessing_fn.side_effect = Exception("Preprocessing error")
    mock_registry.__getitem__.return_value = MockDataset

    loader = DatasetLoader(mock_dataset_config)

    with pytest.raises(Exception, match="Preprocessing error"):
        loader.load_dataset()


@patch("segmentation_robustness_framework.loaders.dataset_loader.image_preprocessing.get_preprocessing_fn")
@patch("segmentation_robustness_framework.loaders.dataset_loader.DATASET_REGISTRY")
def test_load_dataset_error_handling_dataset_construction(
    mock_registry, mock_get_preprocessing_fn, mock_dataset_config
):
    mock_preprocess = Mock()
    mock_target_preprocess = Mock()
    mock_get_preprocessing_fn.return_value = (mock_preprocess, mock_target_preprocess)

    class ErrorDataset(MockDataset):
        def __init__(self, root: str, transform=None, target_transform=None, **kwargs):
            raise ValueError("Dataset construction failed")

    mock_registry.__getitem__.return_value = ErrorDataset

    loader = DatasetLoader(mock_dataset_config)

    with pytest.raises(ValueError, match="Dataset construction failed"):
        loader.load_dataset()


def test_dataset_loader_repr(mock_dataset_config):
    loader = DatasetLoader(mock_dataset_config)
    repr_str = repr(loader)

    assert "DatasetLoader" in repr_str
    assert "object at" in repr_str or "DatasetLoader" in repr_str


@patch("segmentation_robustness_framework.loaders.dataset_loader.image_preprocessing.get_preprocessing_fn")
@patch("segmentation_robustness_framework.loaders.dataset_loader.DATASET_REGISTRY")
def test_load_dataset_returns_dataset_instance(mock_registry, mock_get_preprocessing_fn, mock_dataset_config):
    mock_preprocess = Mock()
    mock_target_preprocess = Mock()
    mock_get_preprocessing_fn.return_value = (mock_preprocess, mock_target_preprocess)

    mock_registry.__getitem__.return_value = MockDataset

    loader = DatasetLoader(mock_dataset_config)
    dataset = loader.load_dataset()

    assert isinstance(dataset, Dataset)
    assert hasattr(dataset, "__len__")
    assert hasattr(dataset, "__getitem__")

    assert len(dataset) == 100
    image, mask = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert image.shape == (3, 256, 256)
    assert mask.shape == (256, 256)


@patch("segmentation_robustness_framework.loaders.dataset_loader.image_preprocessing.get_preprocessing_fn")
@patch("segmentation_robustness_framework.loaders.dataset_loader.DATASET_REGISTRY")
def test_load_dataset_with_none_preprocessing(mock_registry, mock_get_preprocessing_fn, mock_dataset_config):
    mock_get_preprocessing_fn.return_value = (None, None)
    mock_registry.__getitem__.return_value = MockDataset

    loader = DatasetLoader(mock_dataset_config)
    dataset = loader.load_dataset()

    assert isinstance(dataset, MockDataset)
    assert dataset.transform is None
    assert dataset.target_transform is None


@patch("segmentation_robustness_framework.loaders.dataset_loader.image_preprocessing.get_preprocessing_fn")
@patch("segmentation_robustness_framework.loaders.dataset_loader.DATASET_REGISTRY")
def test_load_dataset_parameter_override(mock_registry, mock_get_preprocessing_fn, mock_dataset_config):
    mock_preprocess = Mock()
    mock_target_preprocess = Mock()
    mock_get_preprocessing_fn.return_value = (mock_preprocess, mock_target_preprocess)

    class OverrideTestDataset(MockDataset):
        def __init__(self, root: str, transform=None, target_transform=None, split: str = "default"):
            super().__init__(root, transform, target_transform, split)

    mock_registry.__getitem__.return_value = OverrideTestDataset

    config = mock_dataset_config.copy()
    config["split"] = "custom_split"

    loader = DatasetLoader(config)
    dataset = loader.load_dataset()

    assert dataset.split == "custom_split"


@patch("segmentation_robustness_framework.loaders.dataset_loader.image_preprocessing.get_preprocessing_fn")
@patch("segmentation_robustness_framework.loaders.dataset_loader.DATASET_REGISTRY")
def test_load_dataset_inspect_signature_edge_cases(mock_registry, mock_get_preprocessing_fn, mock_dataset_config):
    mock_preprocess = Mock()
    mock_target_preprocess = Mock()
    mock_get_preprocessing_fn.return_value = (mock_preprocess, mock_target_preprocess)

    class NoParamsDataset(MockDataset):
        def __init__(self):
            super().__init__("/default/root")

    mock_registry.__getitem__.return_value = NoParamsDataset

    loader = DatasetLoader(mock_dataset_config)

    with pytest.raises(TypeError, match="unexpected keyword argument"):
        dataset = loader.load_dataset()

    class FlexibleDataset(MockDataset):
        def __init__(self, *args, **kwargs):
            super().__init__("/default/root")
            self.args = args
            self.kwargs = kwargs

    mock_registry.__getitem__.return_value = FlexibleDataset

    loader = DatasetLoader(mock_dataset_config)
    dataset = loader.load_dataset()

    assert isinstance(dataset, FlexibleDataset)
    assert len(dataset.args) > 0 or len(dataset.kwargs) > 0
