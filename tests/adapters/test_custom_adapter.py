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
