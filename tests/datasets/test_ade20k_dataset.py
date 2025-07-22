import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from PIL import Image
from segmentation_robustness_framework.datasets import ADE20K


@pytest.fixture
def mock_image():
    return Mock(spec=Image.Image)


@pytest.fixture
def mock_mask():
    return Mock(spec=Image.Image)


@pytest.fixture
def temp_dataset_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        ade20k_dir = Path(temp_dir) / "ade20k" / "ADEChallengeData2016"
        ade20k_dir.mkdir(parents=True, exist_ok=True)

        (ade20k_dir / "images" / "training").mkdir(parents=True, exist_ok=True)
        (ade20k_dir / "annotations" / "training").mkdir(parents=True, exist_ok=True)

        (ade20k_dir / "images" / "validation").mkdir(parents=True, exist_ok=True)
        (ade20k_dir / "annotations" / "validation").mkdir(parents=True, exist_ok=True)

        sample_train_images = ["ADE_train_00000001", "ADE_train_00000002", "ADE_train_00000003"]
        for img_name in sample_train_images:
            img_path = ade20k_dir / "images" / "training" / f"{img_name}.jpg"
            img_path.touch()

            mask_path = ade20k_dir / "annotations" / "training" / f"{img_name}.png"
            mask_path.touch()

        sample_val_images = ["ADE_val_00000001", "ADE_val_00000002"]
        for img_name in sample_val_images:
            img_path = ade20k_dir / "images" / "validation" / f"{img_name}.jpg"
            img_path.touch()

            mask_path = ade20k_dir / "annotations" / "validation" / f"{img_name}.png"
            mask_path.touch()

        yield temp_dir


def test_init_with_valid_split(temp_dataset_dir):
    dataset = ADE20K(split="train", root=temp_dataset_dir)

    assert dataset.split == "train"
    assert dataset.num_classes == 150
    assert len(dataset.images) == 3


def test_init_with_invalid_split(temp_dataset_dir):
    with pytest.raises(ValueError, match="Invalid split 'invalid'. Expected one of"):
        ADE20K(split="invalid", root=temp_dataset_dir)


def test_init_with_none_root_uses_cache():
    with patch("segmentation_robustness_framework.utils.dataset_utils.get_cache_dir") as mock_cache:
        mock_cache.return_value = Path("/tmp/cache/ade20k")

        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = False

            with patch("segmentation_robustness_framework.datasets.ade20k.download_dataset") as mock_download:
                with patch("segmentation_robustness_framework.datasets.ade20k.extract_dataset") as mock_extract:
                    mock_download.return_value = "/tmp/downloaded_file.zip"

                    with pytest.raises(FileNotFoundError):
                        ADE20K(split="train", root=None)

                    mock_download.assert_called_once()


@patch("PIL.Image.open")
def test_getitem_with_transforms(mock_open, temp_dataset_dir, mock_image, mock_mask):
    mock_open.side_effect = [mock_image, mock_mask]
    mock_image.convert.return_value = mock_image
    mock_mask.convert.return_value = mock_mask

    mock_transform = Mock()
    mock_target_transform = Mock()
    mock_transform.return_value = "transformed_image"
    mock_target_transform.return_value = "transformed_mask"

    dataset = ADE20K(
        split="train", root=temp_dataset_dir, transform=mock_transform, target_transform=mock_target_transform
    )

    image, mask = dataset[0]

    assert image == "transformed_image"
    assert mask == "transformed_mask"
    mock_transform.assert_called_once_with(mock_image)
    mock_target_transform.assert_called_once_with(mask=mock_mask, ignore_index=None)


def test_dataset_len(temp_dataset_dir):
    dataset = ADE20K(split="train", root=temp_dataset_dir)
    assert len(dataset) == 3
