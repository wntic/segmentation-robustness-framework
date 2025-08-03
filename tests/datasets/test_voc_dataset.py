import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from PIL import Image
from segmentation_robustness_framework.datasets import VOCSegmentation


@pytest.fixture
def mock_image():
    return Mock(spec=Image.Image)


@pytest.fixture
def mock_mask():
    return Mock(spec=Image.Image)


@pytest.fixture
def temp_dataset_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        voc_dir = Path(temp_dir) / "voc" / "VOCdevkit" / "VOC2012"
        voc_dir.mkdir(parents=True, exist_ok=True)

        (voc_dir / "JPEGImages").mkdir()
        (voc_dir / "SegmentationClass").mkdir()
        (voc_dir / "ImageSets" / "Segmentation").mkdir(parents=True)

        sample_images = ["2007_000032", "2007_000039", "2007_000063"]
        for img_name in sample_images:
            img_path = voc_dir / "JPEGImages" / f"{img_name}.jpg"
            img_path.touch()

            mask_path = voc_dir / "SegmentationClass" / f"{img_name}.png"
            mask_path.touch()

        split_file = voc_dir / "ImageSets" / "Segmentation" / "train.txt"
        split_file.write_text("\n".join(sample_images))

        yield temp_dir


def test_init_with_valid_split(temp_dataset_dir):
    dataset = VOCSegmentation(split="train", root=temp_dataset_dir)

    assert dataset.split == "train"
    assert dataset.num_classes == 21
    assert len(dataset.images) == 3
    assert dataset.images == ["2007_000032", "2007_000039", "2007_000063"]


def test_init_with_invalid_split(temp_dataset_dir):
    with pytest.raises(ValueError, match="Invalid split 'invalid'. Expected one of"):
        VOCSegmentation(split="invalid", root=temp_dataset_dir)


def test_init_with_none_root_uses_cache():
    with patch("segmentation_robustness_framework.utils.dataset_utils.get_cache_dir") as mock_cache:
        mock_cache.return_value = Path("/tmp/cache/voc")

        with patch("pathlib.Path.exists") as mock_exists:
            mock_exists.return_value = False

            with patch("segmentation_robustness_framework.datasets.voc.download_dataset") as mock_download:
                with patch("segmentation_robustness_framework.datasets.voc.extract_dataset") as mock_extract:
                    mock_download.return_value = "/tmp/downloaded_file.tar"

                    with pytest.raises(FileNotFoundError):
                        VOCSegmentation(split="train", root=None)

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

    dataset = VOCSegmentation(
        split="train", root=temp_dataset_dir, transform=mock_transform, target_transform=mock_target_transform
    )

    image, mask = dataset[0]

    assert image == "transformed_image"
    assert mask == "transformed_mask"
    mock_transform.assert_called_once_with(mock_image)
    mock_target_transform.assert_called_once_with(mask=mock_mask, ignore_index=255)


def test_dataset_len(temp_dataset_dir):
    dataset = VOCSegmentation(split="train", root=temp_dataset_dir)
    assert len(dataset) == 3


def test_path_handling_with_download_true():
    with tempfile.TemporaryDirectory() as temp_dir:
        root_path = Path(temp_dir)

        voc_dir = root_path / "voc" / "VOCdevkit" / "VOC2012"
        voc_dir.mkdir(parents=True, exist_ok=True)

        (voc_dir / "JPEGImages").mkdir()
        (voc_dir / "SegmentationClass").mkdir()
        (voc_dir / "ImageSets" / "Segmentation").mkdir(parents=True)

        sample_images = ["2007_000032"]
        for img_name in sample_images:
            img_path = voc_dir / "JPEGImages" / f"{img_name}.jpg"
            img_path.touch()

            mask_path = voc_dir / "SegmentationClass" / f"{img_name}.png"
            mask_path.touch()

        split_file = voc_dir / "ImageSets" / "Segmentation" / "train.txt"
        split_file.write_text("\n".join(sample_images))

        with patch("segmentation_robustness_framework.datasets.voc.download_dataset") as mock_download:
            with patch("segmentation_robustness_framework.datasets.voc.extract_dataset") as mock_extract:
                mock_download.return_value = "/tmp/downloaded_file.tar"

                dataset = VOCSegmentation(split="train", root=root_path, download=True)

                assert dataset.images_dir == voc_dir / "JPEGImages"
                assert dataset.masks_dir == voc_dir / "SegmentationClass"
                assert len(dataset.images) == 1


def test_path_handling_with_download_false():
    with tempfile.TemporaryDirectory() as temp_dir:
        voc2012_dir = Path(temp_dir) / "VOCdevkit" / "VOC2012"
        voc2012_dir.mkdir(parents=True, exist_ok=True)

        (voc2012_dir / "JPEGImages").mkdir()
        (voc2012_dir / "SegmentationClass").mkdir()
        (voc2012_dir / "ImageSets" / "Segmentation").mkdir(parents=True)

        sample_images = ["2007_000032"]
        for img_name in sample_images:
            img_path = voc2012_dir / "JPEGImages" / f"{img_name}.jpg"
            img_path.touch()

            mask_path = voc2012_dir / "SegmentationClass" / f"{img_name}.png"
            mask_path.touch()

        split_file = voc2012_dir / "ImageSets" / "Segmentation" / "train.txt"
        split_file.write_text("\n".join(sample_images))

        dataset = VOCSegmentation(split="train", root=voc2012_dir, download=False)

        assert dataset.images_dir == voc2012_dir / "JPEGImages"
        assert dataset.masks_dir == voc2012_dir / "SegmentationClass"
        assert len(dataset.images) == 1


def test_path_handling_with_download_false_and_none_root():
    with tempfile.TemporaryDirectory() as temp_dir:
        cache_dir = Path(temp_dir) / "cache" / "voc"
        cache_dir.mkdir(parents=True, exist_ok=True)

        (cache_dir / "JPEGImages").mkdir()
        (cache_dir / "SegmentationClass").mkdir()
        (cache_dir / "ImageSets" / "Segmentation").mkdir(parents=True)

        sample_images = ["2007_000032"]
        for img_name in sample_images:
            img_path = cache_dir / "JPEGImages" / f"{img_name}.jpg"
            img_path.touch()

            mask_path = cache_dir / "SegmentationClass" / f"{img_name}.png"
            mask_path.touch()

        split_file = cache_dir / "ImageSets" / "Segmentation" / "train.txt"
        split_file.write_text("\n".join(sample_images))

        with patch("segmentation_robustness_framework.utils.dataset_utils.get_cache_dir") as mock_cache:
            mock_cache.return_value = cache_dir

            dataset = VOCSegmentation(split="train", root=None, download=False)

            assert dataset.images_dir == cache_dir / "JPEGImages"
            assert dataset.masks_dir == cache_dir / "SegmentationClass"
            assert len(dataset.images) == 1

            mock_cache.assert_called_once_with("voc")
