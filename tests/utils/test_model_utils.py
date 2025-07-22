from unittest.mock import Mock, patch

import pytest
import torch
from segmentation_robustness_framework.utils.model_utils import (
    get_huggingface_output_size,
    get_model_output_size,
)


class MockHuggingFaceModel(torch.nn.Module):
    def __init__(self, output_shape=(1, 10, 224, 224)):
        super().__init__()
        self.output_shape = output_shape

    def logits(self, x):
        return torch.randn(self.output_shape)

    def forward(self, x):
        return torch.randn(self.output_shape)


class MockHuggingFaceOutput:
    def __init__(self, logits):
        self.logits = logits


def test_get_huggingface_output_size_with_logits_method():
    model = MockHuggingFaceModel(output_shape=(1, 10, 224, 224))
    input_shape = (3, 224, 224)

    output_size = get_huggingface_output_size(model, input_shape, "cpu")

    assert output_size == (224, 224)


def test_get_huggingface_output_size_with_forward_method():
    class ModelWithoutLogits(torch.nn.Module):
        def __init__(self, output_shape=(1, 10, 256, 256)):
            super().__init__()
            self.output_shape = output_shape

        def forward(self, x):
            return torch.randn(self.output_shape)

    model = ModelWithoutLogits(output_shape=(1, 10, 256, 256))
    input_shape = (3, 224, 224)

    output_size = get_huggingface_output_size(model, input_shape, "cpu")

    assert output_size == (256, 256)


def test_get_huggingface_output_size_different_input_shapes():
    test_cases = [
        ((3, 224, 224), (1, 10, 224, 224)),
        ((3, 512, 512), (1, 10, 512, 512)),
        ((1, 128, 128), (1, 5, 128, 128)),
    ]

    for input_shape, output_shape in test_cases:
        model = MockHuggingFaceModel(output_shape=output_shape)
        output_size = get_huggingface_output_size(model, input_shape, "cpu")

        expected_h, expected_w = output_shape[2], output_shape[3]
        assert output_size == (expected_h, expected_w)


def test_get_huggingface_output_size_3d_output():
    model = MockHuggingFaceModel(output_shape=(1, 10, 224))
    input_shape = (3, 224, 224)

    with pytest.raises(RuntimeError, match="Expected 4D output tensor, got 3D"):
        get_huggingface_output_size(model, input_shape, "cpu")


def test_get_huggingface_output_size_5d_output():
    model = MockHuggingFaceModel(output_shape=(1, 10, 224, 224, 1))
    input_shape = (3, 224, 224)

    with pytest.raises(RuntimeError, match="Expected 4D output tensor, got 5D"):
        get_huggingface_output_size(model, input_shape, "cpu")


def test_get_huggingface_output_size_model_error():
    model = MockHuggingFaceModel()
    model.logits = Mock(side_effect=RuntimeError("Model error"))
    input_shape = (3, 224, 224)

    with pytest.raises(RuntimeError, match="Could not determine model output size"):
        get_huggingface_output_size(model, input_shape, "cpu")


def test_get_huggingface_output_size_device_conversion():
    if torch.cuda.is_available():
        model = MockHuggingFaceModel(output_shape=(1, 10, 224, 224))
        model.test_param = torch.nn.Parameter(torch.randn(1))
        input_shape = (3, 224, 224)

        output_size = get_huggingface_output_size(model, input_shape, "cuda")

        assert output_size == (224, 224)
        assert model.test_param.device.type == "cuda"


def test_get_model_output_size_with_logits_method():
    model = MockHuggingFaceModel(output_shape=(1, 10, 224, 224))
    input_shape = (3, 224, 224)

    output_size = get_model_output_size(model, input_shape, "cpu")

    assert output_size == (224, 224)


def test_get_model_output_size_with_forward_method():
    class ModelWithoutLogits(torch.nn.Module):
        def __init__(self, output_shape=(1, 10, 256, 256)):
            super().__init__()
            self.output_shape = output_shape

        def forward(self, x):
            return torch.randn(self.output_shape)

    model = ModelWithoutLogits(output_shape=(1, 10, 256, 256))
    input_shape = (3, 224, 224)

    output_size = get_model_output_size(model, input_shape, "cpu")

    assert output_size == (256, 256)


def test_get_model_output_size_with_huggingface_output_object():
    model = MockHuggingFaceModel(output_shape=(1, 10, 224, 224))
    input_shape = (3, 224, 224)

    output_size = get_model_output_size(model, input_shape, "cpu")

    assert output_size == (224, 224)


def test_get_model_output_size_with_none_logits():
    model = Mock()
    output_obj = MockHuggingFaceOutput(None)
    model.logits = Mock(return_value=output_obj)
    input_shape = (3, 224, 224)

    with pytest.raises(RuntimeError, match="Could not determine model output size"):
        get_model_output_size(model, input_shape, "cpu")


def test_get_model_output_size_output_without_dim():
    model = Mock()
    model.logits = Mock(return_value="not_a_tensor")
    input_shape = (3, 224, 224)

    with pytest.raises(RuntimeError, match="Could not determine model output size"):
        get_model_output_size(model, input_shape, "cpu")


def test_get_model_output_size_3d_output():
    model = MockHuggingFaceModel(output_shape=(1, 10, 224))
    input_shape = (3, 224, 224)

    with pytest.raises(RuntimeError, match="Expected 4D output tensor, got 3D"):
        get_model_output_size(model, input_shape, "cpu")


def test_get_model_output_size_5d_output():
    model = MockHuggingFaceModel(output_shape=(1, 10, 224, 224, 1))
    input_shape = (3, 224, 224)

    with pytest.raises(RuntimeError, match="Expected 4D output tensor, got 5D"):
        get_model_output_size(model, input_shape, "cpu")


def test_get_model_output_size_model_error():
    model = MockHuggingFaceModel()
    model.logits = Mock(side_effect=RuntimeError("Model error"))
    input_shape = (3, 224, 224)

    with pytest.raises(RuntimeError, match="Could not determine model output size"):
        get_model_output_size(model, input_shape, "cpu")


def test_get_model_output_size_device_conversion():
    if torch.cuda.is_available():
        model = MockHuggingFaceModel(output_shape=(1, 10, 224, 224))
        model.test_param = torch.nn.Parameter(torch.randn(1))
        input_shape = (3, 224, 224)

        output_size = get_model_output_size(model, input_shape, "cuda")

        assert output_size == (224, 224)
        assert model.test_param.device.type == "cuda"


def test_get_model_output_size_different_input_shapes():
    test_cases = [
        ((3, 224, 224), (1, 10, 224, 224)),
        ((3, 512, 512), (1, 10, 512, 512)),
        ((1, 128, 128), (1, 5, 128, 128)),
    ]

    for input_shape, output_shape in test_cases:
        model = MockHuggingFaceModel(output_shape=output_shape)
        output_size = get_model_output_size(model, input_shape, "cpu")

        expected_h, expected_w = output_shape[2], output_shape[3]
        assert output_size == (expected_h, expected_w)


def test_get_model_output_size_with_realistic_hf_output():
    class RealisticHFOutput:
        def __init__(self, logits):
            self.logits = logits
            self.hidden_states = None
            self.attentions = None

    model = MockHuggingFaceModel(output_shape=(1, 10, 224, 224))
    input_shape = (3, 224, 224)

    output_size = get_model_output_size(model, input_shape, "cpu")

    assert output_size == (224, 224)


def test_get_model_output_size_with_dict_output():
    model = Mock()
    model.logits = Mock(return_value={"logits": torch.randn(1, 10, 224, 224)})
    input_shape = (3, 224, 224)

    with pytest.raises(RuntimeError, match="Could not determine model output size"):
        get_model_output_size(model, input_shape, "cpu")


def test_get_model_output_size_with_list_output():
    model = Mock()
    model.logits = Mock(return_value=[torch.randn(1, 10, 224, 224)])
    input_shape = (3, 224, 224)

    with pytest.raises(RuntimeError, match="Could not determine model output size"):
        get_model_output_size(model, input_shape, "cpu")


def test_get_model_output_size_with_string_output():
    model = Mock()
    model.logits = Mock(return_value="invalid_output")
    input_shape = (3, 224, 224)

    with pytest.raises(RuntimeError, match="Could not determine model output size"):
        get_model_output_size(model, input_shape, "cpu")


def test_get_model_output_size_with_none_output():
    model = Mock()
    model.logits = Mock(return_value=None)
    input_shape = (3, 224, 224)

    with pytest.raises(RuntimeError, match="Could not determine model output size"):
        get_model_output_size(model, input_shape, "cpu")


def test_get_model_output_size_with_zero_dimensions():
    model = MockHuggingFaceModel(output_shape=(1, 10, 0, 0))
    input_shape = (3, 224, 224)

    output_size = get_model_output_size(model, input_shape, "cpu")

    assert output_size == (0, 0)


def test_get_model_output_size_with_single_pixel():
    model = MockHuggingFaceModel(output_shape=(1, 10, 1, 1))
    input_shape = (3, 224, 224)

    output_size = get_model_output_size(model, input_shape, "cpu")

    assert output_size == (1, 1)


def test_get_model_output_size_logging():
    model = MockHuggingFaceModel(output_shape=(1, 10, 224, 224))
    input_shape = (3, 224, 224)

    with patch("segmentation_robustness_framework.utils.model_utils.logger") as mock_logger:
        output_size = get_model_output_size(model, input_shape, "cpu")

    mock_logger.info.assert_called_with("Detected model output size: 224x224 for input 224x224")


def test_get_huggingface_output_size_logging():
    model = MockHuggingFaceModel(output_shape=(1, 10, 224, 224))
    input_shape = (3, 224, 224)

    with patch("segmentation_robustness_framework.utils.model_utils.logger") as mock_logger:
        output_size = get_huggingface_output_size(model, input_shape, "cpu")

    mock_logger.info.assert_called_with("Detected model output size: 224x224 for input 224x224")


def test_get_model_output_size_error_logging():
    model = MockHuggingFaceModel()
    model.logits = Mock(side_effect=RuntimeError("Model error"))
    input_shape = (3, 224, 224)

    with patch("segmentation_robustness_framework.utils.model_utils.logger") as mock_logger:
        with pytest.raises(RuntimeError):
            get_model_output_size(model, input_shape, "cpu")

    mock_logger.error.assert_called_with("Failed to detect model output size: Model error")


def test_get_huggingface_output_size_error_logging():
    model = MockHuggingFaceModel()
    model.logits = Mock(side_effect=RuntimeError("Model error"))
    input_shape = (3, 224, 224)

    with patch("segmentation_robustness_framework.utils.model_utils.logger") as mock_logger:
        with pytest.raises(RuntimeError):
            get_huggingface_output_size(model, input_shape, "cpu")

    mock_logger.error.assert_called_with("Failed to detect model output size: Model error")


def test_get_model_output_size_sets_model_to_eval():
    model = MockHuggingFaceModel(output_shape=(1, 10, 224, 224))
    model.train()
    input_shape = (3, 224, 224)

    get_model_output_size(model, input_shape, "cpu")

    assert not model.training


def test_get_huggingface_output_size_sets_model_to_eval():
    model = MockHuggingFaceModel(output_shape=(1, 10, 224, 224))
    model.train()
    input_shape = (3, 224, 224)

    get_huggingface_output_size(model, input_shape, "cpu")

    assert not model.training


def test_get_model_output_size_uses_torch_no_grad():
    model = MockHuggingFaceModel(output_shape=(1, 10, 224, 224))
    input_shape = (3, 224, 224)

    with patch("torch.no_grad") as mock_no_grad:
        get_model_output_size(model, input_shape, "cpu")

    mock_no_grad.assert_called_once()


def test_get_huggingface_output_size_uses_torch_no_grad():
    model = MockHuggingFaceModel(output_shape=(1, 10, 224, 224))
    input_shape = (3, 224, 224)

    with patch("torch.no_grad") as mock_no_grad:
        get_huggingface_output_size(model, input_shape, "cpu")

    mock_no_grad.assert_called_once()
