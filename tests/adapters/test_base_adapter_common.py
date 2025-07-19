from unittest.mock import Mock

import torch


def test_adapter_with_different_input_shapes(adapter, mock_model):
    test_shapes = [
        (1, 3, 32, 32),
        (4, 3, 128, 128),
        (8, 3, 256, 256),
    ]

    for batch_size, channels, height, width in test_shapes:
        x = torch.randn(batch_size, channels, height, width)
        expected_logits = torch.randn(batch_size, adapter.num_classes, height, width)
        mock_model.forward = Mock(return_value=expected_logits)

        logits = adapter.logits(x)
        predictions = adapter.predictions(x)

        assert logits.shape == (batch_size, adapter.num_classes, height, width)
        assert predictions.shape == (batch_size, height, width)
        assert torch.equal(logits, expected_logits)


def test_adapter_protocol_compliance(adapter):
    from segmentation_robustness_framework.adapters.base_protocol import SegmentationModelProtocol

    assert isinstance(adapter, SegmentationModelProtocol)
    assert hasattr(adapter, "num_classes")
    assert hasattr(adapter, "logits")
    assert hasattr(adapter, "predictions")
    assert callable(adapter.logits)
    assert callable(adapter.predictions)


def test_adapter_model_attributes_preserved(adapter, mock_model):
    mock_model.some_attribute = "test_value"
    mock_model.some_method = lambda: "test_method"

    assert hasattr(adapter.model, "some_attribute")
    assert adapter.model.some_attribute == "test_value"
    assert hasattr(adapter.model, "some_method")
    assert adapter.model.some_method() == "test_method"


def test_adapter_training_mode_propagation(adapter, mock_model):
    adapter.train()
    assert adapter.training
    assert mock_model.training

    adapter.eval()
    assert not adapter.training
    assert not mock_model.training


def test_adapter_parameters_inheritance(adapter, mock_model):
    mock_model.param1 = torch.nn.Parameter(torch.randn(10))
    mock_model.param2 = torch.nn.Parameter(torch.randn(5, 5))

    adapter_params = list(adapter.parameters())
    model_params = list(mock_model.parameters())

    assert len(adapter_params) == len(model_params)
    for adapter_param, model_param in zip(adapter_params, model_params):
        assert torch.equal(adapter_param, model_param)


def test_adapter_state_dict_inheritance(adapter, mock_model):
    mock_model.register_buffer("buffer1", torch.randn(10))
    mock_model.param1 = torch.nn.Parameter(torch.randn(5))

    adapter_state = adapter.state_dict()
    model_state = mock_model.state_dict()

    expected_adapter_keys = {f"model.{key}" for key in model_state.keys()}
    assert set(adapter_state.keys()) == expected_adapter_keys
    for key in model_state.keys():
        assert torch.equal(adapter_state[f"model.{key}"], model_state[key])


def test_adapter_gradient_flow(adapter, mock_model):
    batch_size, channels, height, width = 2, 3, 64, 64
    x = torch.randn(batch_size, channels, height, width, requires_grad=True)

    simple_model = torch.nn.Conv2d(channels, adapter.num_classes, kernel_size=3, padding=1)
    from segmentation_robustness_framework.adapters.custom_adapter import CustomAdapter

    test_adapter = CustomAdapter(simple_model, num_classes=adapter.num_classes)

    logits = test_adapter.logits(x)
    loss = logits.sum()

    loss.backward()

    assert x.grad is not None
    assert simple_model.weight.grad is not None
    assert simple_model.bias.grad is not None


def test_adapter_device_movement(adapter, mock_model):
    if torch.cuda.is_available():
        device = torch.device("cuda")

        mock_model.param1 = torch.nn.Parameter(torch.randn(10))

        adapter.to(device)
        assert next(adapter.parameters()).device.type == "cuda"
        assert next(mock_model.parameters()).device.type == "cuda"

        adapter.to("cpu")
        assert next(adapter.parameters()).device.type == "cpu"
        assert next(mock_model.parameters()).device.type == "cpu"


def test_adapter_dtype_conversion(adapter, mock_model):
    mock_model.param1 = torch.nn.Parameter(torch.randn(10))

    adapter.half()
    assert next(adapter.parameters()).dtype == torch.float16
    assert next(mock_model.parameters()).dtype == torch.float16

    adapter.float()
    assert next(adapter.parameters()).dtype == torch.float32
    assert next(mock_model.parameters()).dtype == torch.float32


def test_adapter_repr(adapter):
    repr_str = repr(adapter)
    assert "Adapter" in repr_str
    assert isinstance(repr_str, str)
