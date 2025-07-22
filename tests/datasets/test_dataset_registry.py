import pytest
from segmentation_robustness_framework.datasets.registry import register_dataset
from torch.utils.data import Dataset


def test_register_dataset_raises_type_error():
    with pytest.raises(TypeError):

        @register_dataset("test_dataset")
        class TestDataset:
            pass


def test_register_dataset_raises_value_error():
    @register_dataset("test_dataset")
    class TestDataset1(Dataset):
        pass

    with pytest.raises(ValueError):

        @register_dataset("test_dataset")
        class TestDataset2(Dataset):
            pass
