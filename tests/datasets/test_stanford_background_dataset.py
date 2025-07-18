import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from PIL import Image
from segmentation_robustness_framework.datasets import StanfordBackground


@pytest.fixture
def mock_image():
    return Mock(spec=Image.Image)


@pytest.fixture
def mock_mask():
    return Mock(spec=Image.Image)


@pytest.fixture
def temp_dataset_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        stanford_background_dir = Path(temp_dir) / "stanford_background" / "stanford_background"
        stanford_background_dir.mkdir(parents=True, exist_ok=True)

        (stanford_background_dir / "images").mkdir(parents=True, exist_ok=True)
        (stanford_background_dir / "labels_colored").mkdir(parents=True, exist_ok=True)

        sample_images = ["0000072", "0000059", "0000047"]
        for img_name in sample_images:
            img_path = stanford_background_dir / "images" / f"{img_name}.jpg"
            img_path.touch()

            mask_path = stanford_background_dir / "labels_colored" / f"{img_name}.png"
            mask_path.touch()
        yield temp_dir


def test_init_with_none_root_uses_cache():
    with patch("segmentation_robustness_framework.utils.dataset_utils.get_cache_dir") as mock_cache:
        mock_cache.return_value = Path("/tmp/cache/stanford_background")

        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = False

            with patch(
                "segmentation_robustness_framework.datasets.stanford_background.download_dataset"
            ) as mock_download:
                with patch(
                    "segmentation_robustness_framework.datasets.stanford_background.extract_dataset"
                ) as mock_extract:
                    mock_download.return_value = "/tmp/downloaded_file.tar"

                    with pytest.raises(FileNotFoundError):
                        StanfordBackground(root=None)

                    mock_download.assert_called_once()


@patch("PIL.Image.open")
def test_getitem_with_transforms(mock_open, temp_dataset_dir, mock_image, mock_mask):
    mock_open.side_effect = [mock_image, mock_mask]
    mock_image.convert.return_value = mock_image
    mock_mask.convert.return_value = mock_mask

    mock_transform = Mock()
    mock_target_transform = Mock()
    mock_tensor = Mock()
    mock_tensor.unsqueeze.return_value = mock_image
    mock_transform.return_value = mock_tensor
    mock_target_transform.return_value = mock_mask

    dataset = StanfordBackground(
        root=temp_dataset_dir, transform=mock_transform, target_transform=mock_target_transform
    )

    image, mask = dataset[0]

    assert image == mock_image
    assert mask == mock_mask
    mock_transform.assert_called_once_with(mock_image)
    mock_target_transform.assert_called_once_with(mock_mask)


def test_dataset_len(temp_dataset_dir):
    dataset = StanfordBackground(root=temp_dataset_dir)
    assert len(dataset) == 3
