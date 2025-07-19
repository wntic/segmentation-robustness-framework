from unittest.mock import Mock

import pytest
import torch
from segmentation_robustness_framework.adapters.base_protocol import SegmentationModelProtocol
from segmentation_robustness_framework.adapters.smp_adapter import SMPAdapter


class MockSMPModel(torch.nn.Module):
    def __init__(self, num_classes: int = 3):
        super().__init__()
        self.num_classes = num_classes

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        height, width = x.shape[2], x.shape[3]
        return torch.randn(batch_size, self.num_classes, height, width)


class MockSMPModelWithClassifier(torch.nn.Module):
    def __init__(self, out_channels: int = 5):
        super().__init__()
        self.classifier = Mock()
        self.classifier.out_channels = out_channels

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        height, width = x.shape[2], x.shape[3]
        return torch.randn(batch_size, self.classifier.out_channels, height, width)


class MockSMPModelWithoutClassifier(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size = x.shape[0]
        height, width = x.shape[2], x.shape[3]
        return torch.randn(batch_size, 1, height, width)


@pytest.fixture
def mock_smp_model():
    return MockSMPModel(num_classes=3)


@pytest.fixture
def mock_smp_model_with_classifier():
    return MockSMPModelWithClassifier(out_channels=5)


@pytest.fixture
def mock_smp_model_without_classifier():
    return MockSMPModelWithoutClassifier()


@pytest.fixture
def smp_adapter(mock_smp_model):
    return SMPAdapter(mock_smp_model)


def test_smp_adapter_initialization(mock_smp_model):
    adapter = SMPAdapter(mock_smp_model)

    assert adapter.model is mock_smp_model
    assert adapter.num_classes == 3
    assert isinstance(adapter, torch.nn.Module)
    assert isinstance(adapter, SegmentationModelProtocol)


def test_smp_adapter_initialization_with_classifier(mock_smp_model_with_classifier):
    adapter = SMPAdapter(mock_smp_model_with_classifier)

    assert adapter.model is mock_smp_model_with_classifier
    assert adapter.num_classes == 5
    assert isinstance(adapter, torch.nn.Module)
    assert isinstance(adapter, SegmentationModelProtocol)


def test_smp_adapter_initialization_without_classifier(mock_smp_model_without_classifier):
    adapter = SMPAdapter(mock_smp_model_without_classifier)

    assert adapter.model is mock_smp_model_without_classifier
    assert adapter.num_classes == 1
    assert isinstance(adapter, torch.nn.Module)
    assert isinstance(adapter, SegmentationModelProtocol)


def test_smp_adapter_logits(smp_adapter, mock_smp_model):
    batch_size, channels, height, width = 2, 3, 64, 64
    x = torch.randn(batch_size, channels, height, width)

    expected_logits = torch.randn(batch_size, smp_adapter.num_classes, height, width)
    mock_smp_model.forward = Mock(return_value=expected_logits)

    logits = smp_adapter.logits(x)

    assert logits.shape == (batch_size, smp_adapter.num_classes, height, width)
    assert torch.equal(logits, expected_logits)
    mock_smp_model.forward.assert_called_once_with(x)


def test_smp_adapter_predictions(smp_adapter, mock_smp_model):
    batch_size, channels, height, width = 2, 3, 64, 64
    x = torch.randn(batch_size, channels, height, width)

    logits = torch.zeros(batch_size, smp_adapter.num_classes, height, width)
    logits[:, 1, :, :] = 1.0
    mock_smp_model.forward = Mock(return_value=logits)

    predictions = smp_adapter.predictions(x)

    assert predictions.shape == (batch_size, height, width)
    assert torch.all(predictions == 1)
    mock_smp_model.forward.assert_called_once_with(x)


def test_smp_adapter_forward(smp_adapter, mock_smp_model):
    batch_size, channels, height, width = 2, 3, 64, 64
    x = torch.randn(batch_size, channels, height, width)

    expected_logits = torch.randn(batch_size, smp_adapter.num_classes, height, width)
    mock_smp_model.forward = Mock(return_value=expected_logits)

    output = smp_adapter.forward(x)

    assert torch.equal(output, expected_logits)
    mock_smp_model.forward.assert_called_once_with(x)


def test_smp_adapter_protocol_compliance(smp_adapter):
    assert isinstance(smp_adapter, SegmentationModelProtocol)
    assert hasattr(smp_adapter, "num_classes")
    assert hasattr(smp_adapter, "logits")
    assert hasattr(smp_adapter, "predictions")
    assert callable(smp_adapter.logits)
    assert callable(smp_adapter.predictions)


def test_smp_adapter_with_different_input_shapes(smp_adapter, mock_smp_model):
    test_shapes = [
        (1, 3, 32, 32),
        (4, 3, 128, 128),
        (8, 3, 256, 256),
    ]

    for batch_size, channels, height, width in test_shapes:
        x = torch.randn(batch_size, channels, height, width)
        expected_logits = torch.randn(batch_size, smp_adapter.num_classes, height, width)
        mock_smp_model.forward = Mock(return_value=expected_logits)

        logits = smp_adapter.logits(x)
        predictions = smp_adapter.predictions(x)

        assert logits.shape == (batch_size, smp_adapter.num_classes, height, width)
        assert predictions.shape == (batch_size, height, width)
        assert torch.equal(logits, expected_logits)


def test_smp_adapter_with_different_num_classes():
    mock_model = MockSMPModel(num_classes=7)
    adapter = SMPAdapter(mock_model)

    batch_size, channels, height, width = 2, 3, 64, 64
    x = torch.randn(batch_size, channels, height, width)

    expected_logits = torch.randn(batch_size, 7, height, width)
    mock_model.forward = Mock(return_value=expected_logits)

    logits = adapter.logits(x)
    predictions = adapter.predictions(x)

    assert logits.shape == (batch_size, 7, height, width)
    assert predictions.shape == (batch_size, height, width)
    assert torch.equal(logits, expected_logits)


def test_smp_adapter_model_attributes_preserved(smp_adapter, mock_smp_model):
    mock_smp_model.some_attribute = "test_value"
    mock_smp_model.some_method = lambda: "test_method"

    assert hasattr(smp_adapter.model, "some_attribute")
    assert smp_adapter.model.some_attribute == "test_value"
    assert hasattr(smp_adapter.model, "some_method")
    assert smp_adapter.model.some_method() == "test_method"


def test_smp_adapter_training_mode_propagation(smp_adapter, mock_smp_model):
    smp_adapter.train()
    assert smp_adapter.training
    assert mock_smp_model.training

    smp_adapter.eval()
    assert not smp_adapter.training
    assert not mock_smp_model.training


def test_smp_adapter_parameters_inheritance(smp_adapter, mock_smp_model):
    mock_smp_model.param1 = torch.nn.Parameter(torch.randn(10))
    mock_smp_model.param2 = torch.nn.Parameter(torch.randn(5, 5))

    adapter_params = list(smp_adapter.parameters())
    model_params = list(mock_smp_model.parameters())

    assert len(adapter_params) == len(model_params)
    for adapter_param, model_param in zip(adapter_params, model_params):
        assert torch.equal(adapter_param, model_param)


def test_smp_adapter_state_dict_inheritance(smp_adapter, mock_smp_model):
    mock_smp_model.register_buffer("buffer1", torch.randn(10))
    mock_smp_model.param1 = torch.nn.Parameter(torch.randn(5))

    adapter_state = smp_adapter.state_dict()
    model_state = mock_smp_model.state_dict()

    expected_adapter_keys = {f"model.{key}" for key in model_state.keys()}
    assert set(adapter_state.keys()) == expected_adapter_keys
    for key in model_state.keys():
        assert torch.equal(adapter_state[f"model.{key}"], model_state[key])


def test_smp_adapter_gradient_flow(smp_adapter, mock_smp_model):
    batch_size, channels, height, width = 2, 3, 64, 64
    x = torch.randn(batch_size, channels, height, width, requires_grad=True)

    simple_model = torch.nn.Conv2d(channels, smp_adapter.num_classes, kernel_size=3, padding=1)
    adapter = SMPAdapter(simple_model)

    logits = adapter.logits(x)
    loss = logits.sum()

    loss.backward()

    assert x.grad is not None
    assert simple_model.weight.grad is not None
    assert simple_model.bias.grad is not None


def test_smp_adapter_device_movement(smp_adapter, mock_smp_model):
    if torch.cuda.is_available():
        device = torch.device("cuda")

        mock_smp_model.param1 = torch.nn.Parameter(torch.randn(10))

        smp_adapter.to(device)
        assert next(smp_adapter.parameters()).device.type == "cuda"
        assert next(mock_smp_model.parameters()).device.type == "cuda"

        smp_adapter.to("cpu")
        assert next(smp_adapter.parameters()).device.type == "cpu"
        assert next(mock_smp_model.parameters()).device.type == "cpu"


def test_smp_adapter_dtype_conversion(smp_adapter, mock_smp_model):
    mock_smp_model.param1 = torch.nn.Parameter(torch.randn(10))

    smp_adapter.half()
    assert next(smp_adapter.parameters()).dtype == torch.float16
    assert next(mock_smp_model.parameters()).dtype == torch.float16

    smp_adapter.float()
    assert next(smp_adapter.parameters()).dtype == torch.float32
    assert next(mock_smp_model.parameters()).dtype == torch.float32


def test_smp_adapter_repr(smp_adapter):
    repr_str = repr(smp_adapter)
    assert "SMPAdapter" in repr_str
    assert "MockSMPModel" in repr_str


def test_smp_adapter_classifier_detection_priority():
    mock_model = Mock()
    mock_model.classifier = Mock()
    mock_model.classifier.out_channels = 5
    mock_model.num_classes = 3

    adapter = SMPAdapter(mock_model)
    assert adapter.num_classes == 5


def test_smp_adapter_classifier_without_out_channels():
    mock_model = Mock()
    mock_model.classifier = type("Classifier", (), {})()
    mock_model.num_classes = 3

    adapter = SMPAdapter(mock_model)
    assert adapter.num_classes == 3


def test_smp_adapter_no_classifier_no_num_classes():
    mock_model = type("Model", (), {})()

    adapter = SMPAdapter(mock_model)
    assert adapter.num_classes == 1


def test_smp_adapter_classifier_variations():
    test_cases = [
        (True, True, True, 5),
        (True, True, False, 5),
        (True, False, True, 3),
        (True, False, False, 1),
        (False, False, True, 3),
        (False, False, False, 1),
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

        adapter = SMPAdapter(mock_model)
        assert adapter.num_classes == expected


def test_smp_adapter_direct_logits_return():
    batch_size, channels, height, width = 2, 3, 64, 64
    x = torch.randn(batch_size, channels, height, width)

    mock_model = MockSMPModel(num_classes=4)
    adapter = SMPAdapter(mock_model)

    expected_logits = torch.randn(batch_size, 4, height, width)
    mock_model.forward = Mock(return_value=expected_logits)

    logits = adapter.logits(x)

    assert torch.equal(logits, expected_logits)
    mock_model.forward.assert_called_once_with(x)


def test_smp_adapter_with_realistic_smp_structure():
    batch_size, channels, height, width = 2, 3, 64, 64
    num_classes = 6

    class RealisticSMPModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder = torch.nn.Conv2d(channels, 64, kernel_size=3, padding=1)
            self.classifier = torch.nn.Conv2d(64, num_classes, kernel_size=1)

        def forward(self, x):
            features = self.encoder(x)
            return self.classifier(features)

    realistic_model = RealisticSMPModel()
    adapter = SMPAdapter(realistic_model)

    x = torch.randn(batch_size, channels, height, width)

    logits = adapter.logits(x)
    predictions = adapter.predictions(x)

    assert logits.shape == (batch_size, num_classes, height, width)
    assert predictions.shape == (batch_size, height, width)
    assert adapter.num_classes == num_classes
