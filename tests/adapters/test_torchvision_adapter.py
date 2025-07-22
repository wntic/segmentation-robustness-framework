from unittest.mock import Mock

import pytest
import torch
from segmentation_robustness_framework.adapters.base_protocol import SegmentationModelProtocol
from segmentation_robustness_framework.adapters.torchvision_adapter import TorchvisionAdapter


class MockTorchvisionModel(torch.nn.Module):
    def __init__(self, num_classes: int = 21):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> dict:
        batch_size = x.shape[0]
        height, width = x.shape[2], x.shape[3]
        logits = torch.randn(batch_size, self.num_classes, height, width)
        return {"out": logits}


class MockTorchvisionModelWithClassifier(torch.nn.Module):
    def __init__(self, out_channels: int = 5):
        super().__init__()
        self.classifier = Mock()
        self.classifier.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> dict:
        batch_size = x.shape[0]
        height, width = x.shape[2], x.shape[3]
        logits = torch.randn(batch_size, self.classifier.out_channels, height, width)
        return {"out": logits}


class MockTorchvisionModelWithoutClassifier(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> dict:
        batch_size = x.shape[0]
        height, width = x.shape[2], x.shape[3]
        logits = torch.randn(batch_size, 21, height, width)
        return {"out": logits}


@pytest.fixture
def mock_torchvision_model():
    return MockTorchvisionModel(num_classes=21)


@pytest.fixture
def mock_torchvision_model_with_classifier():
    return MockTorchvisionModelWithClassifier(out_channels=5)


@pytest.fixture
def mock_torchvision_model_without_classifier():
    return MockTorchvisionModelWithoutClassifier()


@pytest.fixture
def torchvision_adapter(mock_torchvision_model):
    return TorchvisionAdapter(mock_torchvision_model)


def test_torchvision_adapter_initialization(mock_torchvision_model):
    adapter = TorchvisionAdapter(mock_torchvision_model)

    assert adapter.model is mock_torchvision_model
    assert adapter.num_classes == 21
    assert isinstance(adapter, torch.nn.Module)
    assert isinstance(adapter, SegmentationModelProtocol)


def test_torchvision_adapter_initialization_with_classifier(mock_torchvision_model_with_classifier):
    adapter = TorchvisionAdapter(mock_torchvision_model_with_classifier)

    assert adapter.model is mock_torchvision_model_with_classifier
    assert adapter.num_classes == 5
    assert isinstance(adapter, torch.nn.Module)
    assert isinstance(adapter, SegmentationModelProtocol)


def test_torchvision_adapter_initialization_without_classifier(mock_torchvision_model_without_classifier):
    adapter = TorchvisionAdapter(mock_torchvision_model_without_classifier)

    assert adapter.model is mock_torchvision_model_without_classifier
    assert adapter.num_classes == 21  # Default fallback
    assert isinstance(adapter, torch.nn.Module)
    assert isinstance(adapter, SegmentationModelProtocol)


def test_torchvision_adapter_logits(torchvision_adapter, mock_torchvision_model):
    batch_size, channels, height, width = 2, 3, 64, 64
    x = torch.randn(batch_size, channels, height, width)

    expected_logits = torch.randn(batch_size, torchvision_adapter.num_classes, height, width)
    mock_torchvision_model.forward = Mock(return_value={"out": expected_logits})

    logits = torchvision_adapter.logits(x)

    assert logits.shape == (batch_size, torchvision_adapter.num_classes, height, width)
    assert torch.equal(logits, expected_logits)
    mock_torchvision_model.forward.assert_called_once_with(x)


def test_torchvision_adapter_predictions(torchvision_adapter, mock_torchvision_model):
    batch_size, channels, height, width = 2, 3, 64, 64
    x = torch.randn(batch_size, channels, height, width)

    logits = torch.zeros(batch_size, torchvision_adapter.num_classes, height, width)
    logits[:, 1, :, :] = 1.0
    mock_torchvision_model.forward = Mock(return_value={"out": logits})

    predictions = torchvision_adapter.predictions(x)

    assert predictions.shape == (batch_size, height, width)
    assert torch.all(predictions == 1)
    mock_torchvision_model.forward.assert_called_once_with(x)


def test_torchvision_adapter_forward(torchvision_adapter, mock_torchvision_model):
    batch_size, channels, height, width = 2, 3, 64, 64
    x = torch.randn(batch_size, channels, height, width)

    expected_logits = torch.randn(batch_size, torchvision_adapter.num_classes, height, width)
    mock_torchvision_model.forward = Mock(return_value={"out": expected_logits})

    output = torchvision_adapter.forward(x)

    assert torch.equal(output, expected_logits)
    mock_torchvision_model.forward.assert_called_once_with(x)


def test_torchvision_adapter_protocol_compliance(torchvision_adapter):
    assert isinstance(torchvision_adapter, SegmentationModelProtocol)
    assert hasattr(torchvision_adapter, "num_classes")
    assert hasattr(torchvision_adapter, "logits")
    assert hasattr(torchvision_adapter, "predictions")
    assert callable(torchvision_adapter.logits)
    assert callable(torchvision_adapter.predictions)


def test_torchvision_adapter_with_different_input_shapes(torchvision_adapter, mock_torchvision_model):
    test_shapes = [
        (1, 3, 32, 32),
        (4, 3, 128, 128),
        (8, 3, 256, 256),
    ]

    for batch_size, channels, height, width in test_shapes:
        x = torch.randn(batch_size, channels, height, width)
        expected_logits = torch.randn(batch_size, torchvision_adapter.num_classes, height, width)
        mock_torchvision_model.forward = Mock(return_value={"out": expected_logits})

        logits = torchvision_adapter.logits(x)
        predictions = torchvision_adapter.predictions(x)

        assert logits.shape == (batch_size, torchvision_adapter.num_classes, height, width)
        assert predictions.shape == (batch_size, height, width)
        assert torch.equal(logits, expected_logits)


def test_torchvision_adapter_with_different_num_classes():
    mock_model = MockTorchvisionModel(num_classes=7)
    adapter = TorchvisionAdapter(mock_model)

    batch_size, channels, height, width = 2, 3, 64, 64
    x = torch.randn(batch_size, channels, height, width)

    expected_logits = torch.randn(batch_size, 7, height, width)
    mock_model.forward = Mock(return_value={"out": expected_logits})

    logits = adapter.logits(x)
    predictions = adapter.predictions(x)

    assert logits.shape == (batch_size, 7, height, width)
    assert predictions.shape == (batch_size, height, width)
    assert torch.equal(logits, expected_logits)


def test_torchvision_adapter_model_attributes_preserved(torchvision_adapter, mock_torchvision_model):
    mock_torchvision_model.some_attribute = "test_value"
    mock_torchvision_model.some_method = lambda: "test_method"

    assert hasattr(torchvision_adapter.model, "some_attribute")
    assert torchvision_adapter.model.some_attribute == "test_value"
    assert hasattr(torchvision_adapter.model, "some_method")
    assert torchvision_adapter.model.some_method() == "test_method"


def test_torchvision_adapter_training_mode_propagation(torchvision_adapter, mock_torchvision_model):
    torchvision_adapter.train()
    assert torchvision_adapter.training
    assert mock_torchvision_model.training

    torchvision_adapter.eval()
    assert not torchvision_adapter.training
    assert not mock_torchvision_model.training


def test_torchvision_adapter_parameters_inheritance(torchvision_adapter, mock_torchvision_model):
    mock_torchvision_model.param1 = torch.nn.Parameter(torch.randn(10))
    mock_torchvision_model.param2 = torch.nn.Parameter(torch.randn(5, 5))

    adapter_params = list(torchvision_adapter.parameters())
    model_params = list(mock_torchvision_model.parameters())

    assert len(adapter_params) == len(model_params)
    for adapter_param, model_param in zip(adapter_params, model_params):
        assert torch.equal(adapter_param, model_param)


def test_torchvision_adapter_state_dict_inheritance(torchvision_adapter, mock_torchvision_model):
    mock_torchvision_model.register_buffer("buffer1", torch.randn(10))
    mock_torchvision_model.param1 = torch.nn.Parameter(torch.randn(5))

    adapter_state = torchvision_adapter.state_dict()
    model_state = mock_torchvision_model.state_dict()

    expected_adapter_keys = {f"model.{key}" for key in model_state.keys()}
    assert set(adapter_state.keys()) == expected_adapter_keys
    for key in model_state.keys():
        assert torch.equal(adapter_state[f"model.{key}"], model_state[key])


def test_torchvision_adapter_gradient_flow(torchvision_adapter, mock_torchvision_model):
    batch_size, channels, height, width = 2, 3, 64, 64
    x = torch.randn(batch_size, channels, height, width, requires_grad=True)

    simple_model = torch.nn.Conv2d(channels, torchvision_adapter.num_classes, kernel_size=3, padding=1)
    adapter = TorchvisionAdapter(simple_model)

    def mock_forward(input_tensor):
        output = torch.nn.functional.conv2d(
            input_tensor,
            simple_model.weight,
            simple_model.bias,
            simple_model.stride,
            simple_model.padding,
            simple_model.dilation,
            simple_model.groups,
        )
        return {"out": output}

    simple_model.forward = mock_forward

    logits = adapter.logits(x)
    loss = logits.sum()

    loss.backward()

    assert x.grad is not None
    assert simple_model.weight.grad is not None
    assert simple_model.bias.grad is not None


def test_torchvision_adapter_device_movement(torchvision_adapter, mock_torchvision_model):
    if torch.cuda.is_available():
        device = torch.device("cuda")

        mock_torchvision_model.param1 = torch.nn.Parameter(torch.randn(10))

        torchvision_adapter.to(device)
        assert next(torchvision_adapter.parameters()).device.type == "cuda"
        assert next(mock_torchvision_model.parameters()).device.type == "cuda"

        torchvision_adapter.to("cpu")
        assert next(torchvision_adapter.parameters()).device.type == "cpu"
        assert next(mock_torchvision_model.parameters()).device.type == "cpu"


def test_torchvision_adapter_dtype_conversion(torchvision_adapter, mock_torchvision_model):
    mock_torchvision_model.param1 = torch.nn.Parameter(torch.randn(10))

    torchvision_adapter.half()
    assert next(torchvision_adapter.parameters()).dtype == torch.float16
    assert next(mock_torchvision_model.parameters()).dtype == torch.float16

    torchvision_adapter.float()
    assert next(torchvision_adapter.parameters()).dtype == torch.float32
    assert next(mock_torchvision_model.parameters()).dtype == torch.float32


def test_torchvision_adapter_repr(torchvision_adapter):
    repr_str = repr(torchvision_adapter)
    assert "TorchvisionAdapter" in repr_str
    assert "MockTorchvisionModel" in repr_str


def test_torchvision_adapter_classifier_detection_priority():
    mock_model = Mock()
    mock_model.classifier = Mock()
    mock_model.classifier.out_channels = 5
    mock_model.num_classes = 3

    adapter = TorchvisionAdapter(mock_model)
    assert adapter.num_classes == 5


def test_torchvision_adapter_classifier_without_out_channels():
    mock_model = Mock()
    mock_model.classifier = type("Classifier", (), {})()
    mock_model.num_classes = 3

    adapter = TorchvisionAdapter(mock_model)
    assert adapter.num_classes == 3


def test_torchvision_adapter_no_classifier_no_num_classes():
    mock_model = type("Model", (), {})()

    adapter = TorchvisionAdapter(mock_model)
    assert adapter.num_classes == 21


def test_torchvision_adapter_classifier_variations():
    test_cases = [
        (True, True, True, 5),
        (True, True, False, 5),
        (True, False, True, 3),
        (True, False, False, 21),
        (False, False, True, 3),
        (False, False, False, 21),
    ]

    for has_classifier, has_out_channels, has_num_classes, expected in test_cases:
        mock_model = type("Model", (), {})()

        if has_classifier:
            if has_out_channels:
                mock_model.classifier = type("Classifier", (), {"out_channels": 5})()
            else:
                mock_model.classifier = type("Classifier", (), {})()

        if has_num_classes:
            mock_model.num_classes = 3

        adapter = TorchvisionAdapter(mock_model)
        assert adapter.num_classes == expected


def test_torchvision_adapter_dict_output_handling():
    batch_size, channels, height, width = 2, 3, 64, 64
    x = torch.randn(batch_size, channels, height, width)

    mock_model = MockTorchvisionModel(num_classes=4)
    adapter = TorchvisionAdapter(mock_model)

    expected_logits = torch.randn(batch_size, 4, height, width)
    mock_model.forward = Mock(return_value={"out": expected_logits})

    logits = adapter.logits(x)

    assert torch.equal(logits, expected_logits)
    mock_model.forward.assert_called_once_with(x)


def test_torchvision_adapter_dict_output_with_additional_keys():
    batch_size, channels, height, width = 2, 3, 64, 64
    x = torch.randn(batch_size, channels, height, width)

    mock_model = MockTorchvisionModel(num_classes=4)
    adapter = TorchvisionAdapter(mock_model)

    expected_logits = torch.randn(batch_size, 4, height, width)
    mock_output = {
        "out": expected_logits,
        "aux": torch.randn(batch_size, 4, height, width),
        "features": torch.randn(batch_size, 256, height, width),
    }
    mock_model.forward = Mock(return_value=mock_output)

    logits = adapter.logits(x)

    assert torch.equal(logits, expected_logits)
    mock_model.forward.assert_called_once_with(x)


def test_torchvision_adapter_missing_out_key_error():
    batch_size, channels, height, width = 2, 3, 64, 64
    x = torch.randn(batch_size, channels, height, width)

    mock_model = MockTorchvisionModel(num_classes=4)
    adapter = TorchvisionAdapter(mock_model)

    mock_output = {"aux": torch.randn(batch_size, 4, height, width)}
    mock_model.forward = Mock(return_value=mock_output)

    with pytest.raises(KeyError):
        adapter.logits(x)


def test_torchvision_adapter_non_dict_output_error():
    batch_size, channels, height, width = 2, 3, 64, 64
    x = torch.randn(batch_size, channels, height, width)

    mock_model = MockTorchvisionModel(num_classes=4)
    adapter = TorchvisionAdapter(mock_model)

    expected_logits = torch.randn(batch_size, 4, height, width)
    mock_model.forward = Mock(return_value=expected_logits)

    with pytest.raises(TypeError):
        adapter.logits(x)


def test_torchvision_adapter_with_realistic_torchvision_structure():
    batch_size, channels, height, width = 2, 3, 64, 64
    num_classes = 6

    class RealisticTorchvisionModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Conv2d(channels, 64, kernel_size=3, padding=1)
            self.classifier = torch.nn.Conv2d(64, num_classes, kernel_size=1)

        def forward(self, x):
            features = self.encoder(x)
            logits = self.classifier(features)
            return {"out": logits}

    realistic_model = RealisticTorchvisionModel()
    adapter = TorchvisionAdapter(realistic_model)

    x = torch.randn(batch_size, channels, height, width)

    logits = adapter.logits(x)
    predictions = adapter.predictions(x)

    assert logits.shape == (batch_size, num_classes, height, width)
    assert predictions.shape == (batch_size, height, width)
    assert adapter.num_classes == num_classes


def test_torchvision_adapter_default_num_classes():
    mock_model = type("Model", (), {})()

    adapter = TorchvisionAdapter(mock_model)
    assert adapter.num_classes == 21


def test_torchvision_adapter_custom_default_num_classes():
    mock_model = type("Model", (), {})()
    mock_model.num_classes = 10

    adapter = TorchvisionAdapter(mock_model)
    assert adapter.num_classes == 10


def test_torchvision_adapter_forward_returns_logits():
    batch_size, channels, height, width = 2, 3, 64, 64
    x = torch.randn(batch_size, channels, height, width)

    mock_model = MockTorchvisionModel(num_classes=4)
    adapter = TorchvisionAdapter(mock_model)

    expected_logits = torch.randn(batch_size, 4, height, width)
    mock_model.forward = Mock(return_value={"out": expected_logits})

    forward_output = adapter.forward(x)
    logits_output = adapter.logits(x)

    assert torch.equal(forward_output, logits_output)
    assert torch.equal(forward_output, expected_logits)
