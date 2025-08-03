import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest
import torch
from segmentation_robustness_framework.utils import visualization


def test_denormalize_basic():
    normalized_image = np.array([[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]])

    result = visualization.denormalize(normalized_image)

    # Expected: std * image + mean
    # std = [0.229, 0.224, 0.225], mean = [0.485, 0.456, 0.406]
    expected = np.array([
        [
            [0.229 * 0.1 + 0.485, 0.224 * 0.2 + 0.456, 0.225 * 0.3 + 0.406],
            [0.229 * 0.4 + 0.485, 0.224 * 0.5 + 0.456, 0.225 * 0.6 + 0.406],
        ]
    ])

    np.testing.assert_array_almost_equal(result, expected)


def test_denormalize_shape_preservation():
    shapes = [(100, 100, 3), (256, 512, 3), (64, 64, 3)]

    for shape in shapes:
        normalized_image = np.random.rand(*shape)
        result = visualization.denormalize(normalized_image)

        assert result.shape == shape


def test_denormalize_zero_input():
    zero_image = np.zeros((10, 10, 3))
    result = visualization.denormalize(zero_image)

    expected = np.array([0.485, 0.456, 0.406])
    np.testing.assert_array_almost_equal(result[0, 0], expected)


def test_get_class_colors_voc():
    classes, colors = visualization.get_class_colors("VOC")

    assert isinstance(classes, list)
    assert isinstance(colors, list)
    assert len(classes) > 0
    assert len(colors) > 0
    assert len(classes) == len(colors)


def test_get_class_colors_ade20k():
    classes, colors = visualization.get_class_colors("ADE20K")

    assert isinstance(classes, list)
    assert isinstance(colors, list)
    assert len(classes) > 0
    assert len(colors) > 0
    assert len(classes) == len(colors)


def test_get_class_colors_stanford_background():
    classes, colors = visualization.get_class_colors("StanfordBackground")

    assert isinstance(classes, list)
    assert isinstance(colors, list)
    assert len(classes) > 0
    assert len(colors) > 0
    assert len(classes) == len(colors)


def test_get_class_colors_cityscapes():
    classes, colors = visualization.get_class_colors("Cityscapes")

    assert isinstance(classes, list)
    assert isinstance(colors, list)
    assert len(classes) > 0
    assert len(colors) > 0
    assert len(classes) == len(colors)


def test_get_class_colors_invalid_dataset():
    with pytest.raises(ValueError, match="Invalide dataset"):
        visualization.get_class_colors("InvalidDataset")


def test_create_legend_basic():
    mask = np.array([[0, 1, 2], [0, 1, 2]])
    classes = ["background", "person", "car"]
    colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0)]

    colors = [(c[0] / 255, c[1] / 255, c[2] / 255) for c in colors]

    handles, filtered_classes = visualization.create_legend(mask, classes, colors)

    assert len(handles) == 3
    assert len(filtered_classes) == 3
    assert filtered_classes == ["background", "person", "car"]


def test_create_legend_single_class():
    mask = np.array([[0, 0], [0, 0]])
    classes = ["background", "person", "car"]
    colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0)]

    colors = [(c[0] / 255, c[1] / 255, c[2] / 255) for c in colors]

    handles, filtered_classes = visualization.create_legend(mask, classes, colors)

    assert len(handles) == 1
    assert len(filtered_classes) == 1
    assert filtered_classes == ["background"]


def test_create_legend_no_classes():
    mask = np.array([])
    classes = ["background", "person", "car"]
    colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0)]

    colors = [(c[0] / 255, c[1] / 255, c[2] / 255) for c in colors]

    handles, filtered_classes = visualization.create_legend(mask, classes, colors)

    assert len(handles) == 0
    assert len(filtered_classes) == 0


def test_create_legend_handles_type():
    mask = np.array([[0, 1], [0, 1]])
    classes = ["background", "person"]
    colors = [(0, 0, 0), (255, 0, 0)]

    colors = [(c[0] / 255, c[1] / 255, c[2] / 255) for c in colors]

    handles, _ = visualization.create_legend(mask, classes, colors)

    for handle in handles:
        assert isinstance(handle, plt.Rectangle)


@pytest.fixture
def sample_tensors():
    image = torch.randn(1, 3, 256, 256)
    ground_truth = torch.randint(0, 21, (1, 256, 256))
    mask = torch.randint(0, 21, (1, 256, 256))
    adv_mask = torch.randint(0, 21, (1, 256, 256))
    return image, ground_truth, mask, adv_mask


def test_visualize_images_basic(sample_tensors):
    image, ground_truth, mask, adv_mask = sample_tensors

    visualization.visualize_images(
        image, ground_truth, mask, adv_mask, "VOC", denormalize_image=False, show=False, save=False
    )


def test_visualize_images_with_denormalization(sample_tensors):
    image, ground_truth, mask, adv_mask = sample_tensors

    visualization.visualize_images(
        image, ground_truth, mask, adv_mask, "VOC", denormalize_image=True, show=False, save=False
    )


def test_visualize_images_with_title(sample_tensors):
    image, ground_truth, mask, adv_mask = sample_tensors

    visualization.visualize_images(
        image, ground_truth, mask, adv_mask, "VOC", title="Test Visualization", show=False, save=False
    )


def test_visualize_images_invalid_image_shape(sample_tensors):
    _, ground_truth, mask, adv_mask = sample_tensors
    invalid_image = torch.randn(3, 256, 256)

    with pytest.raises(ValueError, match="Expected original image with shape"):
        visualization.visualize_images(invalid_image, ground_truth, mask, adv_mask, "VOC", show=False, save=False)


def test_visualize_images_invalid_ground_truth_shape(sample_tensors):
    image, _, mask, adv_mask = sample_tensors
    invalid_ground_truth = torch.randint(0, 21, (256, 256))

    with pytest.raises(ValueError, match="Expected ground truth with shape"):
        visualization.visualize_images(image, invalid_ground_truth, mask, adv_mask, "VOC", show=False, save=False)


def test_visualize_images_invalid_mask_shape(sample_tensors):
    image, ground_truth, _, adv_mask = sample_tensors
    invalid_mask = torch.randint(0, 21, (256, 256))

    with pytest.raises(ValueError, match="Expected segmentation mask with shape"):
        visualization.visualize_images(image, ground_truth, invalid_mask, adv_mask, "VOC", show=False, save=False)


def test_visualize_images_invalid_adv_mask_shape(sample_tensors):
    image, ground_truth, mask, _ = sample_tensors
    invalid_adv_mask = torch.randint(0, 21, (256, 256))

    with pytest.raises(ValueError, match="Expected adversarial segmentation mask with shape"):
        visualization.visualize_images(image, ground_truth, mask, invalid_adv_mask, "VOC", show=False, save=False)


def test_visualize_images_save(sample_tensors):
    image, ground_truth, mask, adv_mask = sample_tensors

    with tempfile.TemporaryDirectory() as temp_dir:
        visualization.visualize_images(
            image, ground_truth, mask, adv_mask, "VOC", save=True, save_dir=temp_dir, show=False
        )

        # Check that file was created
        files = list(Path(temp_dir).glob("*.jpg"))
        assert len(files) == 1


def test_visualize_images_different_datasets(sample_tensors):
    image, ground_truth, mask, adv_mask = sample_tensors
    datasets = ["VOC", "ADE20K", "StanfordBackground", "Cityscapes"]

    for dataset in datasets:
        small_mask = torch.randint(0, 5, (1, 64, 64))
        small_adv_mask = torch.randint(0, 5, (1, 64, 64))

        visualization.visualize_images(image, ground_truth, small_mask, small_adv_mask, dataset, show=False, save=False)


@pytest.fixture
def sample_json_data():
    return {
        "FGSM": {
            "attacks": [
                {"params": {"epsilon": 0.01}, "adv_metrics": {"accuracy": [0.8, 0.7, 0.6], "iou": [0.6, 0.5, 0.4]}},
                {"params": {"epsilon": 0.02}, "adv_metrics": {"accuracy": [0.7, 0.6, 0.5], "iou": [0.5, 0.4, 0.3]}},
            ]
        }
    }


@patch("matplotlib.pyplot.show")
def test_visualize_metrics_basic(mock_show, sample_json_data):
    visualization.visualize_metrics(sample_json_data, "FGSM", "epsilon", "accuracy")
    mock_show.assert_called_once()


@patch("matplotlib.pyplot.show")
def test_visualize_metrics_multiple_metrics(mock_show, sample_json_data):
    visualization.visualize_metrics(sample_json_data, "FGSM", "epsilon", ["accuracy", "iou"])
    mock_show.assert_called_once()


def test_visualize_metrics_from_file(sample_json_data):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_json_data, f)
        json_path = f.name

    try:
        with patch("matplotlib.pyplot.show") as mock_show:
            visualization.visualize_metrics(json_path, "FGSM", "epsilon", "accuracy")
            mock_show.assert_called_once()
    finally:
        Path(json_path).unlink()


def test_visualize_metrics_file_not_found():
    with patch("builtins.print") as mock_print:
        visualization.visualize_metrics("nonexistent.json", "FGSM", "epsilon", "accuracy")
        mock_print.assert_called_with("File nonexistent.json does not exist.")


def test_visualize_metrics_invalid_json():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("invalid json content")
        json_path = f.name

    try:
        with pytest.raises(ValueError, match="Error decoding JSON from file"):
            visualization.visualize_metrics(json_path, "FGSM", "epsilon", "accuracy")
    finally:
        Path(json_path).unlink()


def test_visualize_metrics_attack_not_found(sample_json_data):
    with pytest.raises(ValueError, match="Attack NonExistentAttack not found"):
        visualization.visualize_metrics(sample_json_data, "NonExistentAttack", "epsilon", "accuracy")


def test_visualize_metrics_string_input(sample_json_data):
    with patch("matplotlib.pyplot.show") as mock_show:
        visualization.visualize_metrics(sample_json_data, "FGSM", "epsilon", "accuracy")
        mock_show.assert_called_once()


def test_visualize_metrics_path_input(sample_json_data):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_json_data, f)
        json_path = Path(f.name)

    try:
        with patch("matplotlib.pyplot.show") as mock_show:
            visualization.visualize_metrics(json_path, "FGSM", "epsilon", "accuracy")
            mock_show.assert_called_once()
    finally:
        json_path.unlink()


@pytest.fixture
def sample_clean_json_data():
    return {
        "clean_metrics": {"accuracy": [0.9, 0.85, 0.88], "iou": [0.7, 0.65, 0.68], "precision_macro": [0.8, 0.75, 0.78]}
    }


def test_print_clean_metrics_basic(sample_clean_json_data, capsys):
    visualization.print_clean_metrics(sample_clean_json_data, "accuracy")

    captured = capsys.readouterr()
    assert "Clean Metrics:" in captured.out
    assert "accuracy:" in captured.out


def test_print_clean_metrics_multiple_metrics(sample_clean_json_data, capsys):
    visualization.print_clean_metrics(sample_clean_json_data, ["accuracy", "iou"])

    captured = capsys.readouterr()
    assert "Clean Metrics:" in captured.out
    assert "accuracy:" in captured.out
    assert "iou:" in captured.out


def test_print_clean_metrics_from_file(sample_clean_json_data, capsys):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_clean_json_data, f)
        json_path = f.name

    try:
        visualization.print_clean_metrics(json_path, "accuracy")
        captured = capsys.readouterr()
        assert "Clean Metrics:" in captured.out
        assert "accuracy:" in captured.out
    finally:
        Path(json_path).unlink()


def test_print_clean_metrics_file_not_found(capsys):
    visualization.print_clean_metrics("nonexistent.json", "accuracy")

    captured = capsys.readouterr()
    assert "File nonexistent.json does not exist." in captured.out


def test_print_clean_metrics_invalid_json():
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        f.write("invalid json content")
        json_path = f.name

    try:
        with patch("builtins.print") as mock_print:
            visualization.print_clean_metrics(json_path, "accuracy")
            mock_print.assert_called()
            call_args = mock_print.call_args_list
            error_calls = [call for call in call_args if "Error decoding JSON from file" in str(call)]
            assert len(error_calls) > 0
    finally:
        Path(json_path).unlink()


def test_print_clean_metrics_no_clean_metrics(capsys):
    json_data = {"some_other_data": [1, 2, 3]}

    with pytest.raises(ValueError, match="Clean metrics not found"):
        visualization.print_clean_metrics(json_data, "accuracy")


def test_print_clean_metrics_metric_not_found(sample_clean_json_data, capsys):
    visualization.print_clean_metrics(sample_clean_json_data, "nonexistent_metric")

    captured = capsys.readouterr()
    assert "nonexistent_metric: not found in clean metrics" in captured.out


def test_print_clean_metrics_string_input(sample_clean_json_data, capsys):
    visualization.print_clean_metrics(sample_clean_json_data, "accuracy")

    captured = capsys.readouterr()
    assert "Clean Metrics:" in captured.out
    assert "accuracy:" in captured.out


def test_print_clean_metrics_path_input(sample_clean_json_data, capsys):
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        json.dump(sample_clean_json_data, f)
        json_path = Path(f.name)

    try:
        visualization.print_clean_metrics(json_path, "accuracy")
        captured = capsys.readouterr()
        assert "Clean Metrics:" in captured.out
        assert "accuracy:" in captured.out
    finally:
        json_path.unlink()


def test_full_visualization_workflow():
    image = torch.randn(1, 3, 256, 256)
    ground_truth = torch.randint(0, 21, (1, 256, 256))
    mask = torch.randint(0, 21, (1, 256, 256))
    adv_mask = torch.randint(0, 21, (1, 256, 256))

    visualization.visualize_images(
        image, ground_truth, mask, adv_mask, "VOC", denormalize_image=True, show=False, save=False
    )


def test_metrics_workflow():
    metrics_data = {
        "FGSM": {
            "attacks": [
                {"params": {"epsilon": 0.01}, "adv_metrics": {"accuracy": [0.8, 0.7, 0.6]}},
                {"params": {"epsilon": 0.02}, "adv_metrics": {"accuracy": [0.7, 0.6, 0.5]}},
            ]
        },
        "clean_metrics": {"accuracy": [0.9, 0.85, 0.88]},
    }

    with patch("matplotlib.pyplot.show"):
        visualization.visualize_metrics(metrics_data, "FGSM", "epsilon", "accuracy")

    with patch("builtins.print"):
        visualization.print_clean_metrics(metrics_data, "accuracy")


def test_denormalize_integration():
    normalized_image = np.random.rand(100, 100, 3)

    denormalized_image = visualization.denormalize(normalized_image)

    assert np.any(denormalized_image > 0)
    assert np.any(denormalized_image < 1)
    assert denormalized_image.shape == (100, 100, 3)


def test_class_colors_integration():
    datasets = ["VOC", "ADE20K", "StanfordBackground", "Cityscapes"]

    for dataset in datasets:
        classes, colors = visualization.get_class_colors(dataset)

        assert len(classes) > 0
        assert len(colors) > 0
        assert len(classes) == len(colors)

        for color in colors:
            assert len(color) == 3
            assert all(0 <= c <= 255 for c in color)


def test_legend_integration():
    mask = np.random.randint(0, 5, (100, 100))
    classes = ["background", "person", "car", "bicycle", "motorcycle"]
    colors = [(0, 0, 0), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0)]

    colors = [(c[0] / 255, c[1] / 255, c[2] / 255) for c in colors]

    handles, filtered_classes = visualization.create_legend(mask, classes, colors)

    assert len(handles) > 0
    assert len(filtered_classes) > 0
    assert len(handles) == len(filtered_classes)

    for handle in handles:
        assert isinstance(handle, plt.Rectangle)
