from unittest.mock import Mock

import pytest
import torch
from segmentation_robustness_framework.adapters.base_protocol import SegmentationModelProtocol
from segmentation_robustness_framework.adapters.custom_adapter import CustomAdapter


class MockModel(torch.nn.Module):
    def __init__(self, output_channels: int = 3):
        super().__init__()
        self.output_channels = output_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        height, width = x.shape[2], x.shape[3]
        return torch.randn(batch_size, self.output_channels, height, width)


@pytest.fixture
def mock_model():
    return MockModel(output_channels=3)


@pytest.fixture
def custom_adapter(mock_model):
    return CustomAdapter(mock_model, num_classes=3)


def test_custom_adapter_initialization(mock_model):
    adapter = CustomAdapter(mock_model, num_classes=5)

    assert adapter.model is mock_model
    assert adapter.num_classes == 5
    assert isinstance(adapter, torch.nn.Module)
    assert isinstance(adapter, SegmentationModelProtocol)


def test_custom_adapter_initialization_default_num_classes(mock_model):
    adapter = CustomAdapter(mock_model)

    assert adapter.model is mock_model
    assert adapter.num_classes == 1


def test_custom_adapter_logits(custom_adapter, mock_model):
    batch_size, channels, height, width = 2, 3, 64, 64
    x = torch.randn(batch_size, channels, height, width)

    expected_logits = torch.randn(batch_size, custom_adapter.num_classes, height, width)
    mock_model.forward = Mock(return_value=expected_logits)

    logits = custom_adapter.logits(x)

    assert logits.shape == (batch_size, custom_adapter.num_classes, height, width)
    assert torch.equal(logits, expected_logits)
    mock_model.forward.assert_called_once_with(x)


def test_custom_adapter_predictions(custom_adapter, mock_model):
    batch_size, channels, height, width = 2, 3, 64, 64
    x = torch.randn(batch_size, channels, height, width)

    logits = torch.zeros(batch_size, custom_adapter.num_classes, height, width)
    logits[:, 1, :, :] = 1.0
    mock_model.forward = Mock(return_value=logits)

    predictions = custom_adapter.predictions(x)

    assert predictions.shape == (batch_size, height, width)
    assert torch.all(predictions == 1)
    mock_model.forward.assert_called_once_with(x)


def test_custom_adapter_forward(custom_adapter, mock_model):
    batch_size, channels, height, width = 2, 3, 64, 64
    x = torch.randn(batch_size, channels, height, width)

    expected_logits = torch.randn(batch_size, custom_adapter.num_classes, height, width)
    mock_model.forward = Mock(return_value=expected_logits)

    output = custom_adapter.forward(x)

    assert torch.equal(output, expected_logits)
    mock_model.forward.assert_called_once_with(x)


def test_custom_adapter_protocol_compliance(custom_adapter):
    assert isinstance(custom_adapter, SegmentationModelProtocol)
    assert hasattr(custom_adapter, "num_classes")
    assert hasattr(custom_adapter, "logits")
    assert hasattr(custom_adapter, "predictions")
    assert callable(custom_adapter.logits)
    assert callable(custom_adapter.predictions)


def test_custom_adapter_with_different_input_shapes(custom_adapter, mock_model):
    test_shapes = [
        (1, 3, 32, 32),
        (4, 3, 128, 128),
        (8, 3, 256, 256),
    ]

    for batch_size, channels, height, width in test_shapes:
        x = torch.randn(batch_size, channels, height, width)
        expected_logits = torch.randn(batch_size, custom_adapter.num_classes, height, width)
        mock_model.forward = Mock(return_value=expected_logits)

        logits = custom_adapter.logits(x)
        predictions = custom_adapter.predictions(x)

        assert logits.shape == (batch_size, custom_adapter.num_classes, height, width)
        assert predictions.shape == (batch_size, height, width)
        assert torch.equal(logits, expected_logits)


def test_custom_adapter_with_different_num_classes():
    mock_model = MockModel(output_channels=5)
    adapter = CustomAdapter(mock_model, num_classes=5)

    batch_size, channels, height, width = 2, 3, 64, 64
    x = torch.randn(batch_size, channels, height, width)

    expected_logits = torch.randn(batch_size, 5, height, width)
    mock_model.forward = Mock(return_value=expected_logits)

    logits = adapter.logits(x)
    predictions = adapter.predictions(x)

    assert logits.shape == (batch_size, 5, height, width)
    assert predictions.shape == (batch_size, height, width)
    assert torch.equal(logits, expected_logits)


def test_custom_adapter_model_attributes_preserved(custom_adapter, mock_model):
    mock_model.some_attribute = "test_value"
    mock_model.some_method = lambda: "test_method"

    assert hasattr(custom_adapter.model, "some_attribute")
    assert custom_adapter.model.some_attribute == "test_value"
    assert hasattr(custom_adapter.model, "some_method")
    assert custom_adapter.model.some_method() == "test_method"


def test_custom_adapter_training_mode_propagation(custom_adapter, mock_model):
    custom_adapter.train()
    assert custom_adapter.training
    assert mock_model.training

    custom_adapter.eval()
    assert not custom_adapter.training
    assert not mock_model.training


def test_custom_adapter_parameters_inheritance(custom_adapter, mock_model):
    mock_model.param1 = torch.nn.Parameter(torch.randn(10))
    mock_model.param2 = torch.nn.Parameter(torch.randn(5, 5))

    adapter_params = list(custom_adapter.parameters())
    model_params = list(mock_model.parameters())

    assert len(adapter_params) == len(model_params)
    for adapter_param, model_param in zip(adapter_params, model_params):
        assert torch.equal(adapter_param, model_param)


def test_custom_adapter_state_dict_inheritance(custom_adapter, mock_model):
    mock_model.register_buffer("buffer1", torch.randn(10))
    mock_model.param1 = torch.nn.Parameter(torch.randn(5))

    adapter_state = custom_adapter.state_dict()
    model_state = mock_model.state_dict()

    expected_adapter_keys = {f"model.{key}" for key in model_state.keys()}
    assert set(adapter_state.keys()) == expected_adapter_keys
    for key in model_state.keys():
        assert torch.equal(adapter_state[f"model.{key}"], model_state[key])


def test_custom_adapter_gradient_flow(custom_adapter, mock_model):
    batch_size, channels, height, width = 2, 3, 64, 64
    x = torch.randn(batch_size, channels, height, width, requires_grad=True)

    simple_model = torch.nn.Conv2d(channels, custom_adapter.num_classes, kernel_size=3, padding=1)
    adapter = CustomAdapter(simple_model, num_classes=custom_adapter.num_classes)

    logits = adapter.logits(x)
    loss = logits.sum()

    loss.backward()

    assert x.grad is not None
    assert simple_model.weight.grad is not None
    assert simple_model.bias.grad is not None


def test_custom_adapter_device_movement(custom_adapter, mock_model):
    if torch.cuda.is_available():
        device = torch.device("cuda")

        mock_model.param1 = torch.nn.Parameter(torch.randn(10))

        custom_adapter.to(device)
        assert next(custom_adapter.parameters()).device.type == "cuda"
        assert next(mock_model.parameters()).device.type == "cuda"

        custom_adapter.to("cpu")
        assert next(custom_adapter.parameters()).device.type == "cpu"
        assert next(mock_model.parameters()).device.type == "cpu"


def test_custom_adapter_dtype_conversion(custom_adapter, mock_model):
    mock_model.param1 = torch.nn.Parameter(torch.randn(10))

    custom_adapter.half()
    assert next(custom_adapter.parameters()).dtype == torch.float16
    assert next(mock_model.parameters()).dtype == torch.float16

    custom_adapter.float()
    assert next(custom_adapter.parameters()).dtype == torch.float32
    assert next(mock_model.parameters()).dtype == torch.float32


def test_custom_adapter_repr(custom_adapter):
    repr_str = repr(custom_adapter)
    assert "CustomAdapter" in repr_str
    assert "MockModel" in repr_str
