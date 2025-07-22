from unittest.mock import Mock, patch

import numpy as np
import pytest
import torch
from PIL import Image
from segmentation_robustness_framework.utils.image_preprocessing import (
    DATASET_COLOR_MAPPINGS,
    _convert_rgb_mask_to_index,
    get_preprocessing_fn,
    prepare_inputs,
    register_dataset_colors,
)


def test_register_dataset_colors():
    DATASET_COLOR_MAPPINGS.clear()

    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    register_dataset_colors("test_dataset", colors)

    assert "test_dataset" in DATASET_COLOR_MAPPINGS
    assert DATASET_COLOR_MAPPINGS["test_dataset"] == colors


def test_register_dataset_colors_overwrite():
    DATASET_COLOR_MAPPINGS.clear()

    initial_colors = [(255, 0, 0), (0, 255, 0)]
    register_dataset_colors("test_dataset", initial_colors)

    new_colors = [(0, 0, 255), (255, 255, 0)]
    register_dataset_colors("test_dataset", new_colors)

    assert DATASET_COLOR_MAPPINGS["test_dataset"] == new_colors
    assert len(DATASET_COLOR_MAPPINGS["test_dataset"]) == 2


def test_prepare_inputs_with_processor():
    mock_bundle = Mock()
    mock_processor = Mock()
    mock_bundle.processor = mock_processor

    mock_processor.return_value = {
        "pixel_values": torch.randn(1, 3, 224, 224),
        "attention_mask": torch.ones(1, 224, 224),
    }

    sample_image = Image.new("RGB", (224, 224), color="red")

    result = prepare_inputs(sample_image, mock_bundle, device="cpu")

    mock_processor.assert_called_once_with(sample_image, return_tensors="pt")
    assert "pixel_values" in result
    assert "attention_mask" in result
    assert result["pixel_values"].device.type == "cpu"
    assert result["attention_mask"].device.type == "cpu"


def test_prepare_inputs_without_processor():
    mock_model = Mock()
    delattr(mock_model, "processor")

    sample_tensor = torch.randn(1, 3, 224, 224)

    result = prepare_inputs(sample_tensor, mock_model, device="cpu")

    assert "pixel_values" in result
    assert torch.equal(result["pixel_values"], sample_tensor)
    assert result["pixel_values"].device.type == "cpu"


def test_prepare_inputs_pil_image():
    mock_bundle = Mock()
    mock_processor = Mock()
    mock_bundle.processor = mock_processor
    mock_processor.return_value = {"pixel_values": torch.randn(1, 3, 224, 224)}

    sample_image = Image.new("RGB", (224, 224), color="blue")

    result = prepare_inputs(sample_image, mock_bundle, device="cpu")

    mock_processor.assert_called_once_with(sample_image, return_tensors="pt")


def test_prepare_inputs_numpy_array():
    mock_bundle = Mock()
    mock_processor = Mock()
    mock_bundle.processor = mock_processor
    mock_processor.return_value = {"pixel_values": torch.randn(1, 3, 224, 224)}

    sample_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)

    result = prepare_inputs(sample_array, mock_bundle, device="cpu")

    mock_processor.assert_called_once_with(sample_array, return_tensors="pt")


def test_prepare_inputs_torch_tensor():
    mock_bundle = Mock()
    mock_processor = Mock()
    mock_bundle.processor = mock_processor
    mock_processor.return_value = {"pixel_values": torch.randn(1, 3, 224, 224)}

    sample_tensor = torch.randn(3, 224, 224)

    result = prepare_inputs(sample_tensor, mock_bundle, device="cpu")

    mock_processor.assert_called_once_with(sample_tensor, return_tensors="pt")


def test_get_preprocessing_fn_valid_input():
    image_shape = [224, 224]
    preprocess, target_preprocess = get_preprocessing_fn(image_shape)

    assert callable(preprocess)
    assert callable(target_preprocess)


def test_get_preprocessing_fn_height_divisible_by_8():
    image_shape = [230, 224]
    preprocess, target_preprocess = get_preprocessing_fn(image_shape)

    assert callable(preprocess)
    assert callable(target_preprocess)


def test_get_preprocessing_fn_width_divisible_by_8():
    image_shape = [224, 230]
    preprocess, target_preprocess = get_preprocessing_fn(image_shape)

    assert callable(preprocess)
    assert callable(target_preprocess)


def test_get_preprocessing_fn_both_dimensions_adjusted():
    image_shape = [230, 230]
    preprocess, target_preprocess = get_preprocessing_fn(image_shape)

    assert callable(preprocess)
    assert callable(target_preprocess)


def test_get_preprocessing_fn_invalid_type():
    with pytest.raises(TypeError, match="Expected a list"):
        get_preprocessing_fn("invalid")


def test_get_preprocessing_fn_invalid_length():
    with pytest.raises(ValueError, match="Expected image_shape of length 2"):
        get_preprocessing_fn([224])


def test_get_preprocessing_fn_invalid_dimension_types():
    with pytest.raises(TypeError, match="Expected type"):
        get_preprocessing_fn(["224", "224"])


def test_preprocessing_functions_work():
    image_shape = [224, 224]
    preprocess, target_preprocess = get_preprocessing_fn(image_shape)

    test_image = Image.new("RGB", (100, 100), color="red")
    processed_image = preprocess(test_image)

    assert isinstance(processed_image, torch.Tensor)
    assert processed_image.shape == (3, 224, 224)

    test_mask = Image.new("L", (100, 100), color=1)
    processed_mask = target_preprocess(test_mask)

    assert isinstance(processed_mask, torch.Tensor)
    assert processed_mask.shape == (224, 224)
    assert processed_mask.dtype == torch.long


def test_target_preprocess_with_ignore_index():
    image_shape = [224, 224]
    _, target_preprocess = get_preprocessing_fn(image_shape)

    test_mask = Image.new("L", (100, 100), color=1)
    mask_array = np.array(test_mask)
    mask_array[0, 0] = 255
    test_mask = Image.fromarray(mask_array)

    processed_mask = target_preprocess(test_mask, ignore_index=255)

    assert processed_mask[0, 0] == -1


def test_convert_rgb_mask_to_index_with_dataset():
    DATASET_COLOR_MAPPINGS.clear()
    test_colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
    register_dataset_colors("test_dataset", test_colors)

    mask = np.zeros((10, 10, 3), dtype=np.uint8)
    mask[0:5, :] = [255, 0, 0]
    mask[5:, :] = [0, 255, 0]

    result = _convert_rgb_mask_to_index(mask, "test_dataset")

    assert isinstance(result, torch.Tensor)
    assert result.shape == (10, 10)
    assert result.dtype == torch.long
    assert torch.all(result[0:5, :] == 0)
    assert torch.all(result[5:, :] == 1)


def test_convert_rgb_mask_to_index_without_dataset():
    mask = np.zeros((10, 10, 3), dtype=np.uint8)
    mask[:, :, 0] = 255

    result = _convert_rgb_mask_to_index(mask)

    assert isinstance(result, torch.Tensor)
    assert result.shape == (10, 10)
    assert result.dtype == torch.long
    assert torch.all(result == 255)


def test_convert_rgb_mask_to_index_unknown_colors():
    DATASET_COLOR_MAPPINGS.clear()
    test_colors = [(255, 0, 0), (0, 255, 0)]
    register_dataset_colors("test_dataset", test_colors)

    mask = np.zeros((10, 10, 3), dtype=np.uint8)
    mask[0:5, :] = [255, 0, 0]
    mask[5:, :] = [128, 128, 128]

    with patch("segmentation_robustness_framework.utils.image_preprocessing.logger") as mock_logger:
        result = _convert_rgb_mask_to_index(mask, "test_dataset")

    assert isinstance(result, torch.Tensor)
    assert result.shape == (10, 10)


def test_convert_rgb_mask_to_index_high_unassigned_percentage():
    DATASET_COLOR_MAPPINGS.clear()
    test_colors = [(255, 0, 0)]
    register_dataset_colors("test_dataset", test_colors)

    mask = np.zeros((10, 10, 3), dtype=np.uint8)
    mask[0, 0] = [255, 0, 0]
    mask[1:, :] = [128, 128, 128]

    with patch("segmentation_robustness_framework.utils.image_preprocessing.logger") as mock_logger:
        result = _convert_rgb_mask_to_index(mask, "test_dataset")

    mock_logger.warning.assert_called()


def test_convert_rgb_mask_to_index_indices_out_of_range():
    DATASET_COLOR_MAPPINGS.clear()
    test_colors = [(255, 0, 0), (0, 255, 0)]
    register_dataset_colors("test_dataset", test_colors)

    mask = np.zeros((10, 10, 3), dtype=np.uint8)
    mask[0:5, :] = [255, 0, 0]
    mask[5:, :] = [128, 128, 128]

    with patch("segmentation_robustness_framework.utils.image_preprocessing.logger") as mock_logger:
        with patch("numpy.argmin", return_value=np.full(50, 5)):
            result = _convert_rgb_mask_to_index(mask, "test_dataset")

    mock_logger.warning.assert_called()


def test_preprocessing_functions_with_different_shapes():
    test_shapes = [[256, 256], [512, 512], [100, 100]]

    for shape in test_shapes:
        preprocess, target_preprocess = get_preprocessing_fn(shape)

        test_image = Image.new("RGB", (50, 50), color="blue")
        processed_image = preprocess(test_image)

        expected_h = (shape[0] // 8 + 1) * 8 if shape[0] % 8 != 0 else shape[0]
        expected_w = (shape[1] // 8 + 1) * 8 if shape[1] % 8 != 0 else shape[1]

        assert processed_image.shape == (3, expected_h, expected_w)


def test_prepare_inputs_device_conversion():
    if torch.cuda.is_available():
        mock_bundle = Mock()
        mock_processor = Mock()
        mock_bundle.processor = mock_processor
        mock_processor.return_value = {
            "pixel_values": torch.randn(1, 3, 224, 224),
            "attention_mask": torch.ones(1, 224, 224),
        }

        sample_image = Image.new("RGB", (224, 224), color="green")

        result = prepare_inputs(sample_image, mock_bundle, device="cuda")

        assert result["pixel_values"].device.type == "cuda"
        assert result["attention_mask"].device.type == "cuda"


def test_prepare_inputs_cpu_device():
    mock_bundle = Mock()
    mock_processor = Mock()
    mock_bundle.processor = mock_processor
    mock_processor.return_value = {
        "pixel_values": torch.randn(1, 3, 224, 224),
        "attention_mask": torch.ones(1, 224, 224),
    }

    sample_image = Image.new("RGB", (224, 224), color="yellow")

    result = prepare_inputs(sample_image, mock_bundle, device="cpu")

    assert result["pixel_values"].device.type == "cpu"
    assert result["attention_mask"].device.type == "cpu"


def test_target_preprocess_grayscale_mask():
    image_shape = [224, 224]
    _, target_preprocess = get_preprocessing_fn(image_shape)

    test_mask = Image.new("L", (100, 100), color=128)

    result = target_preprocess(test_mask)

    assert isinstance(result, torch.Tensor)
    assert result.shape == (224, 224)
    assert result.dtype == torch.long
    assert torch.all(result == 128)


def test_target_preprocess_rgb_mask_no_dataset():
    image_shape = [224, 224]
    _, target_preprocess = get_preprocessing_fn(image_shape)

    test_mask = Image.new("RGB", (100, 100), color=(255, 0, 0))

    result = target_preprocess(test_mask)

    assert isinstance(result, torch.Tensor)
    assert result.shape == (224, 224)
    assert result.dtype == torch.long
    assert torch.all(result == 255)
