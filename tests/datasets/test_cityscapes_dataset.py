import json
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from PIL import Image
from segmentation_robustness_framework.datasets import Cityscapes


@pytest.fixture
def mock_image():
    return Mock(spec=Image.Image)


@pytest.fixture
def mock_mask():
    return Mock(spec=Image.Image)


@pytest.fixture
def temp_dataset_dir():
    with tempfile.TemporaryDirectory() as temp_dir:
        root_path = Path(temp_dir)

        (root_path / "leftImg8bit" / "train").mkdir(parents=True, exist_ok=True)
        (root_path / "leftImg8bit" / "val").mkdir(parents=True, exist_ok=True)
        (root_path / "leftImg8bit" / "test").mkdir(parents=True, exist_ok=True)

        (root_path / "gtFine" / "train").mkdir(parents=True, exist_ok=True)
        (root_path / "gtFine" / "val").mkdir(parents=True, exist_ok=True)
        (root_path / "gtCoarse" / "train").mkdir(parents=True, exist_ok=True)
        (root_path / "gtCoarse" / "val").mkdir(parents=True, exist_ok=True)
        (root_path / "gtCoarse" / "train_extra").mkdir(parents=True, exist_ok=True)

        cities = ["aachen", "bochum", "bremen"]

        for city in cities:
            city_img_dir = root_path / "leftImg8bit" / "train" / city
            city_img_dir.mkdir(parents=True, exist_ok=True)

            city_gt_dir = root_path / "gtFine" / "train" / city
            city_gt_dir.mkdir(parents=True, exist_ok=True)

            sample_files = [
                f"{city}_000000_000000_leftImg8bit.png",
                f"{city}_000000_000001_leftImg8bit.png",
                f"{city}_000000_000002_leftImg8bit.png",
            ]

            for file_name in sample_files:
                img_path = city_img_dir / file_name
                img_path.touch()

                base_name = file_name.replace("_leftImg8bit.png", "")

                semantic_path = city_gt_dir / f"{base_name}_gtFine_labelIds.png"
                semantic_path.touch()

                instance_path = city_gt_dir / f"{base_name}_gtFine_instanceIds.png"
                instance_path.touch()

                color_path = city_gt_dir / f"{base_name}_gtFine_color.png"
                color_path.touch()

                polygon_path = city_gt_dir / f"{base_name}_gtFine_polygons.json"
                polygon_data = {"objects": [{"polygon": [[0, 0], [100, 0], [100, 100], [0, 100]]}]}
                polygon_path.write_text(json.dumps(polygon_data))

        yield temp_dir


@pytest.fixture
def temp_dataset_dir_with_train_extra():
    with tempfile.TemporaryDirectory() as temp_dir:
        root_path = Path(temp_dir)

        (root_path / "leftImg8bit" / "train").mkdir(parents=True, exist_ok=True)
        (root_path / "leftImg8bit" / "val").mkdir(parents=True, exist_ok=True)
        (root_path / "leftImg8bit" / "test").mkdir(parents=True, exist_ok=True)

        (root_path / "gtFine" / "train").mkdir(parents=True, exist_ok=True)
        (root_path / "gtFine" / "val").mkdir(parents=True, exist_ok=True)
        (root_path / "gtCoarse" / "train").mkdir(parents=True, exist_ok=True)
        (root_path / "gtCoarse" / "val").mkdir(parents=True, exist_ok=True)
        (root_path / "gtCoarse" / "train_extra").mkdir(parents=True, exist_ok=True)

        (root_path / "leftImg8bit_trainextra" / "train_extra").mkdir(parents=True, exist_ok=True)

        cities = ["aachen", "bochum"]

        for city in cities:
            city_img_dir = root_path / "leftImg8bit_trainextra" / "train_extra" / city
            city_img_dir.mkdir(parents=True, exist_ok=True)

            city_gt_dir = root_path / "gtCoarse" / "train_extra" / city
            city_gt_dir.mkdir(parents=True, exist_ok=True)

            sample_files = [f"{city}_000000_000000_leftImg8bit.png", f"{city}_000000_000001_leftImg8bit.png"]

            for file_name in sample_files:
                img_path = city_img_dir / file_name
                img_path.touch()

                base_name = file_name.replace("_leftImg8bit.png", "")

                semantic_path = city_gt_dir / f"{base_name}_gtCoarse_labelIds.png"
                semantic_path.touch()

                instance_path = city_gt_dir / f"{base_name}_gtCoarse_instanceIds.png"
                instance_path.touch()

                color_path = city_gt_dir / f"{base_name}_gtCoarse_color.png"
                color_path.touch()

                polygon_path = city_gt_dir / f"{base_name}_gtCoarse_polygons.json"
                polygon_data = {"objects": [{"polygon": [[0, 0], [100, 0], [100, 100], [0, 100]]}]}
                polygon_path.write_text(json.dumps(polygon_data))

        yield temp_dir


def test_init_with_valid_split(temp_dataset_dir):
    dataset = Cityscapes(root=temp_dataset_dir, split="train")

    assert dataset.split == "train"
    assert dataset.mode == "gtFine"
    assert dataset.num_classes == 35
    assert len(dataset.images) == 9


def test_init_with_invalid_split(temp_dataset_dir):
    with pytest.raises(ValueError, match="Invalid split 'invalid'. Expected one of"):
        Cityscapes(root=temp_dataset_dir, split="invalid")


def test_init_with_nonexistent_root_directory():
    nonexistent_path = "/path/that/does/not/exist"
    with pytest.raises(ValueError, match=f"Root directory '{nonexistent_path}' does not exist."):
        Cityscapes(root=nonexistent_path)


def test_init_with_invalid_mode(temp_dataset_dir):
    with pytest.raises(ValueError, match="Invalid mode 'invalid'. Expected one of"):
        Cityscapes(root=temp_dataset_dir, mode="invalid")


def test_init_with_invalid_target_type(temp_dataset_dir):
    with pytest.raises(ValueError, match="Invalid target_type 'invalid'. Expected one of"):
        Cityscapes(root=temp_dataset_dir, target_type="invalid")


def test_init_with_test_split_and_coarse_mode(temp_dataset_dir):
    with pytest.raises(ValueError, match="The 'test' split is not available for 'coarse' mode"):
        Cityscapes(root=temp_dataset_dir, split="test", mode="coarse")


def test_init_with_train_extra_split_and_fine_mode(temp_dataset_dir):
    with pytest.raises(ValueError, match="The 'train_extra' split is not available for 'fine' mode"):
        Cityscapes(root=temp_dataset_dir, split="train_extra", mode="fine")


def test_train_extra_split_with_missing_leftImg8bit_trainextra_directory(temp_dataset_dir):
    with pytest.raises(ValueError, match="Directory.*leftImg8bit_trainextra.*train_extra.*does not exist"):
        Cityscapes(root=temp_dataset_dir, split="train_extra", mode="coarse")


@patch("PIL.Image.open")
def test_getitem_with_multiple_target_types(mock_open, temp_dataset_dir, mock_image, mock_mask):
    mock_open.side_effect = [mock_image, mock_mask, mock_mask, mock_mask]
    mock_image.convert.return_value = mock_image
    mock_mask.convert.return_value = mock_mask

    dataset = Cityscapes(root=temp_dataset_dir, split="train", target_type=["semantic", "instance", "color"])
    image, targets = dataset[0]

    assert image == mock_image
    assert isinstance(targets, list)
    assert len(targets) == 3
    assert all(target == mock_mask for target in targets)


@patch("PIL.Image.open")
def test_getitem_with_polygon_target(mock_open, temp_dataset_dir, mock_image):
    mock_open.side_effect = [mock_image]
    mock_image.convert.return_value = mock_image

    dataset = Cityscapes(root=temp_dataset_dir, split="train", target_type="polygon")
    image, target = dataset[0]

    assert image == mock_image
    assert isinstance(target, dict)
    assert "objects" in target


@patch("PIL.Image.open")
def test_getitem_with_mixed_target_types_and_transform(mock_open, temp_dataset_dir, mock_image, mock_mask):
    mock_open.side_effect = [mock_image, mock_mask, mock_mask]
    mock_image.convert.return_value = mock_image
    mock_mask.convert.return_value = mock_mask

    mock_transform = Mock()
    mock_target_transform = Mock()
    mock_transform.return_value = "transformed_image"
    mock_target_transform.return_value = "transformed_mask"

    dataset = Cityscapes(
        root=temp_dataset_dir,
        split="train",
        target_type=["semantic", "polygon"],
        transform=mock_transform,
        target_transform=mock_target_transform,
    )

    with patch.object(dataset, "_load_json", return_value={"objects": [{"polygon": [[0, 0], [100, 100]]}]}):
        image, targets = dataset[0]

    assert image == "transformed_image"
    assert isinstance(targets, list)
    assert len(targets) == 2

    assert mock_target_transform.call_count == 1
    assert targets[0] == "transformed_mask"
    assert isinstance(targets[1], dict)
    assert "objects" in targets[1]


@patch("PIL.Image.open")
def test_getitem_with_single_target_and_transform(mock_open, temp_dataset_dir, mock_image, mock_mask):
    mock_open.side_effect = [mock_image, mock_mask]
    mock_image.convert.return_value = mock_image
    mock_mask.convert.return_value = mock_mask

    mock_transform = Mock()
    mock_target_transform = Mock()
    mock_transform.return_value = "transformed_image"
    mock_target_transform.return_value = "transformed_mask"

    dataset = Cityscapes(
        root=temp_dataset_dir,
        split="train",
        target_type="semantic",
        transform=mock_transform,
        target_transform=mock_target_transform,
    )

    image, target = dataset[0]

    assert image == "transformed_image"
    assert target == "transformed_mask"
    mock_transform.assert_called_once_with(mock_image)
    mock_target_transform.assert_called_once_with(mock_mask)


def test_dataset_len(temp_dataset_dir):
    dataset = Cityscapes(root=temp_dataset_dir, split="train")
    assert len(dataset) == 9
