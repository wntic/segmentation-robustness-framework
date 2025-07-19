from unittest.mock import Mock

import pytest
import torch
from segmentation_robustness_framework.adapters.base_protocol import SegmentationModelProtocol
from segmentation_robustness_framework.adapters.huggingface_adapter import HuggingFaceAdapter


class MockHuggingFaceModel(torch.nn.Module):
    """Mock HuggingFace model for testing HuggingFaceAdapter."""

    def __init__(self, num_labels: int = 3):
        super().__init__()
        self.config = Mock()
        self.config.num_labels = num_labels

    def forward(self, pixel_values: torch.Tensor):
        """Return mock output with logits attribute."""
        batch_size = pixel_values.shape[0]
        height, width = pixel_values.shape[2], pixel_values.shape[3]
        logits = torch.randn(batch_size, self.config.num_labels, height, width)

        output = Mock()
        output.logits = logits
        return output


class MockHuggingFaceModelWithoutConfig(torch.nn.Module):
    """Mock HuggingFace model without config for testing fallback behavior."""

    def __init__(self):
        super().__init__()
        # Create a config object without num_labels attribute
        self.config = type("Config", (), {})()

    def forward(self, pixel_values: torch.Tensor):
        """Return mock output with logits attribute."""
        batch_size = pixel_values.shape[0]
        height, width = pixel_values.shape[2], pixel_values.shape[3]
        logits = torch.randn(batch_size, 1, height, width)  # Default to 1 class

        output = Mock()
        output.logits = logits
        return output


@pytest.fixture
def mock_hf_model():
    return MockHuggingFaceModel(num_labels=3)


@pytest.fixture
def mock_hf_model_without_config():
    return MockHuggingFaceModelWithoutConfig()


@pytest.fixture
def hf_adapter(mock_hf_model):
    return HuggingFaceAdapter(mock_hf_model)


def test_huggingface_adapter_initialization(mock_hf_model):
    adapter = HuggingFaceAdapter(mock_hf_model)

    assert adapter.model is mock_hf_model
    assert adapter.num_classes == 3
    assert isinstance(adapter, torch.nn.Module)
    assert isinstance(adapter, SegmentationModelProtocol)


def test_huggingface_adapter_initialization_without_config(mock_hf_model_without_config):
    adapter = HuggingFaceAdapter(mock_hf_model_without_config)

    assert adapter.model is mock_hf_model_without_config
    assert adapter.num_classes == 1  # Default fallback
    assert isinstance(adapter, torch.nn.Module)
    assert isinstance(adapter, SegmentationModelProtocol)


def test_huggingface_adapter_logits(hf_adapter, mock_hf_model):
    batch_size, channels, height, width = 2, 3, 64, 64
    x = torch.randn(batch_size, channels, height, width)

    # Mock the model's forward method to return predictable output
    expected_logits = torch.randn(batch_size, hf_adapter.num_classes, height, width)
    mock_output = Mock()
    mock_output.logits = expected_logits
    mock_hf_model.forward = Mock(return_value=mock_output)

    logits = hf_adapter.logits(x)

    assert logits.shape == (batch_size, hf_adapter.num_classes, height, width)
    assert torch.equal(logits, expected_logits)
    mock_hf_model.forward.assert_called_once_with(pixel_values=x)


def test_huggingface_adapter_predictions(hf_adapter, mock_hf_model):
    batch_size, channels, height, width = 2, 3, 64, 64
    x = torch.randn(batch_size, channels, height, width)

    # Create logits where argmax will give predictable results
    logits = torch.zeros(batch_size, hf_adapter.num_classes, height, width)
    logits[:, 1, :, :] = 1.0
    mock_output = Mock()
    mock_output.logits = logits
    mock_hf_model.forward = Mock(return_value=mock_output)

    predictions = hf_adapter.predictions(x)

    assert predictions.shape == (batch_size, height, width)
    assert torch.all(predictions == 1)
    mock_hf_model.forward.assert_called_once_with(pixel_values=x)


def test_huggingface_adapter_forward(hf_adapter, mock_hf_model):
    batch_size, channels, height, width = 2, 3, 64, 64
    x = torch.randn(batch_size, channels, height, width)

    expected_logits = torch.randn(batch_size, hf_adapter.num_classes, height, width)
    mock_output = Mock()
    mock_output.logits = expected_logits
    mock_hf_model.forward = Mock(return_value=mock_output)

    output = hf_adapter.forward(x)

    assert torch.equal(output, expected_logits)
    mock_hf_model.forward.assert_called_once_with(pixel_values=x)


def test_huggingface_adapter_protocol_compliance(hf_adapter):
    assert isinstance(hf_adapter, SegmentationModelProtocol)
    assert hasattr(hf_adapter, "num_classes")
    assert hasattr(hf_adapter, "logits")
    assert hasattr(hf_adapter, "predictions")
    assert callable(hf_adapter.logits)
    assert callable(hf_adapter.predictions)


def test_huggingface_adapter_with_different_input_shapes(hf_adapter, mock_hf_model):
    test_shapes = [
        (1, 3, 32, 32),
        (4, 3, 128, 128),
        (8, 3, 256, 256),
    ]

    for batch_size, channels, height, width in test_shapes:
        x = torch.randn(batch_size, channels, height, width)
        expected_logits = torch.randn(batch_size, hf_adapter.num_classes, height, width)
        mock_output = Mock()
        mock_output.logits = expected_logits
        mock_hf_model.forward = Mock(return_value=mock_output)

        logits = hf_adapter.logits(x)
        predictions = hf_adapter.predictions(x)

        assert logits.shape == (batch_size, hf_adapter.num_classes, height, width)
        assert predictions.shape == (batch_size, height, width)
        assert torch.equal(logits, expected_logits)


def test_huggingface_adapter_with_different_num_classes():
    mock_model = MockHuggingFaceModel(num_labels=5)
    adapter = HuggingFaceAdapter(mock_model)

    batch_size, channels, height, width = 2, 3, 64, 64
    x = torch.randn(batch_size, channels, height, width)

    expected_logits = torch.randn(batch_size, 5, height, width)
    mock_output = Mock()
    mock_output.logits = expected_logits
    mock_model.forward = Mock(return_value=mock_output)

    logits = adapter.logits(x)
    predictions = adapter.predictions(x)

    assert logits.shape == (batch_size, 5, height, width)
    assert predictions.shape == (batch_size, height, width)
    assert torch.equal(logits, expected_logits)


def test_huggingface_adapter_model_attributes_preserved(hf_adapter, mock_hf_model):
    mock_hf_model.some_attribute = "test_value"
    mock_hf_model.some_method = lambda: "test_method"

    assert hasattr(hf_adapter.model, "some_attribute")
    assert hf_adapter.model.some_attribute == "test_value"
    assert hasattr(hf_adapter.model, "some_method")
    assert hf_adapter.model.some_method() == "test_method"


def test_huggingface_adapter_training_mode_propagation(hf_adapter, mock_hf_model):
    hf_adapter.train()
    assert hf_adapter.training
    assert mock_hf_model.training

    hf_adapter.eval()
    assert not hf_adapter.training
    assert not mock_hf_model.training


def test_huggingface_adapter_parameters_inheritance(hf_adapter, mock_hf_model):
    mock_hf_model.param1 = torch.nn.Parameter(torch.randn(10))
    mock_hf_model.param2 = torch.nn.Parameter(torch.randn(5, 5))

    adapter_params = list(hf_adapter.parameters())
    model_params = list(mock_hf_model.parameters())

    assert len(adapter_params) == len(model_params)
    for adapter_param, model_param in zip(adapter_params, model_params):
        assert torch.equal(adapter_param, model_param)


def test_huggingface_adapter_state_dict_inheritance(hf_adapter, mock_hf_model):
    mock_hf_model.register_buffer("buffer1", torch.randn(10))
    mock_hf_model.param1 = torch.nn.Parameter(torch.randn(5))

    adapter_state = hf_adapter.state_dict()
    model_state = mock_hf_model.state_dict()

    expected_adapter_keys = {f"model.{key}" for key in model_state.keys()}
    assert set(adapter_state.keys()) == expected_adapter_keys
    for key in model_state.keys():
        assert torch.equal(adapter_state[f"model.{key}"], model_state[key])


def test_huggingface_adapter_gradient_flow(hf_adapter, mock_hf_model):
    batch_size, channels, height, width = 2, 3, 64, 64
    x = torch.randn(batch_size, channels, height, width, requires_grad=True)

    # Create a simple model with parameters and add config
    simple_model = torch.nn.Conv2d(channels, hf_adapter.num_classes, kernel_size=3, padding=1)
    simple_model.config = Mock()
    simple_model.config.num_labels = hf_adapter.num_classes
    adapter = HuggingFaceAdapter(simple_model)

    # Mock the forward method to return logits directly without recursion
    def mock_forward(pixel_values):
        output = Mock()
        # Call the original Conv2d forward method directly
        output.logits = torch.nn.functional.conv2d(
            pixel_values,
            simple_model.weight,
            simple_model.bias,
            simple_model.stride,
            simple_model.padding,
            simple_model.dilation,
            simple_model.groups,
        )
        return output

    adapter.model.forward = mock_forward

    logits = adapter.logits(x)
    loss = logits.sum()

    loss.backward()

    assert x.grad is not None
    assert simple_model.weight.grad is not None
    assert simple_model.bias.grad is not None


def test_huggingface_adapter_device_movement(hf_adapter, mock_hf_model):
    if torch.cuda.is_available():
        device = torch.device("cuda")

        mock_hf_model.param1 = torch.nn.Parameter(torch.randn(10))

        hf_adapter.to(device)
        assert next(hf_adapter.parameters()).device.type == "cuda"
        assert next(mock_hf_model.parameters()).device.type == "cuda"

        hf_adapter.to("cpu")
        assert next(hf_adapter.parameters()).device.type == "cpu"
        assert next(mock_hf_model.parameters()).device.type == "cpu"


def test_huggingface_adapter_dtype_conversion(hf_adapter, mock_hf_model):
    mock_hf_model.param1 = torch.nn.Parameter(torch.randn(10))

    hf_adapter.half()
    assert next(hf_adapter.parameters()).dtype == torch.float16
    assert next(mock_hf_model.parameters()).dtype == torch.float16

    hf_adapter.float()
    assert next(hf_adapter.parameters()).dtype == torch.float32
    assert next(mock_hf_model.parameters()).dtype == torch.float32


def test_huggingface_adapter_repr(hf_adapter):
    repr_str = repr(hf_adapter)
    assert "HuggingFaceAdapter" in repr_str
    assert "MockHuggingFaceModel" in repr_str


def test_huggingface_adapter_pixel_values_key(hf_adapter, mock_hf_model):
    """Test that the adapter correctly uses 'pixel_values' as the input key."""
    batch_size, channels, height, width = 2, 3, 64, 64
    x = torch.randn(batch_size, channels, height, width)

    # Mock the model to capture the call arguments
    mock_hf_model.forward = Mock()
    mock_output = Mock()
    mock_output.logits = torch.randn(batch_size, hf_adapter.num_classes, height, width)
    mock_hf_model.forward.return_value = mock_output

    hf_adapter.logits(x)

    # Verify that the model was called with pixel_values key
    mock_hf_model.forward.assert_called_once()
    call_args = mock_hf_model.forward.call_args
    assert "pixel_values" in call_args.kwargs
    assert torch.equal(call_args.kwargs["pixel_values"], x)


def test_huggingface_adapter_logits_attribute_access(hf_adapter, mock_hf_model):
    """Test that the adapter correctly accesses the logits attribute from the model output."""
    batch_size, channels, height, width = 2, 3, 64, 64
    x = torch.randn(batch_size, channels, height, width)

    # Create a mock output without logits attribute
    mock_output = Mock()
    # Explicitly remove logits attribute to test error handling
    if hasattr(mock_output, "logits"):
        delattr(mock_output, "logits")

    mock_hf_model.forward = Mock(return_value=mock_output)

    # This should raise an AttributeError since the output doesn't have logits
    with pytest.raises(AttributeError):
        hf_adapter.logits(x)


def test_huggingface_adapter_config_num_labels_variations():
    """Test adapter initialization with different config.num_labels values."""
    test_cases = [
        (5, 5),
        (1, 1),
        (10, 10),
        (None, 1),  # No num_labels attribute
    ]

    for num_labels, expected in test_cases:
        mock_model = Mock()
        mock_model.config = Mock()
        if num_labels is not None:
            mock_model.config.num_labels = num_labels
        else:
            delattr(mock_model.config, "num_labels")

        adapter = HuggingFaceAdapter(mock_model)
        assert adapter.num_classes == expected


def test_huggingface_adapter_with_realistic_hf_output_structure():
    """Test adapter with a more realistic HuggingFace output structure."""
    batch_size, channels, height, width = 2, 3, 64, 64
    num_classes = 3

    # Create a realistic HF model output
    class RealisticHFOutput:
        def __init__(self, logits):
            self.logits = logits
            self.hidden_states = None  # Additional HF attributes
            self.attentions = None

    mock_model = MockHuggingFaceModel(num_labels=num_classes)
    adapter = HuggingFaceAdapter(mock_model)

    x = torch.randn(batch_size, channels, height, width)
    expected_logits = torch.randn(batch_size, num_classes, height, width)

    # Mock to return realistic output
    realistic_output = RealisticHFOutput(expected_logits)
    mock_model.forward = Mock(return_value=realistic_output)

    logits = adapter.logits(x)
    predictions = adapter.predictions(x)

    assert torch.equal(logits, expected_logits)
    assert predictions.shape == (batch_size, height, width)
