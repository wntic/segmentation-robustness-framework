import logging
from types import SimpleNamespace
from unittest import mock

import pytest
from segmentation_robustness_framework.loaders.models.universal_loader import UniversalModelLoader


class _DummyModel:
    pass


def _make_universal(monkeypatch, custom_loaders):
    ul = UniversalModelLoader()
    patched = ul.loaders.copy()
    patched.update(custom_loaders)
    monkeypatch.setattr(ul, "loaders", patched, raising=True)
    return ul


def test_universal_loader_unsupported_model_type(monkeypatch):
    ul = _make_universal(monkeypatch, {})
    with pytest.raises(ValueError):
        ul.load_model("unknown", {})


def test_universal_loader_missing_dependency(monkeypatch):
    ul = _make_universal(monkeypatch, {"huggingface": None})
    with pytest.raises(ImportError):
        ul.load_model("huggingface", {})


def test_universal_loader_delegates_load_model(monkeypatch):
    mock_loader = mock.MagicMock()
    mock_model = _DummyModel()
    mock_loader.load_model.return_value = mock_model

    mock_adapter_class = mock.MagicMock()
    mock_adapter_class.__name__ = "MockAdapter"
    mock_adapter_instance = mock.MagicMock()
    mock_adapter_instance.model = mock_model
    mock_adapter_class.return_value = mock_adapter_instance

    ul = _make_universal(monkeypatch, {"torchvision": mock_loader})

    with mock.patch(
        "segmentation_robustness_framework.loaders.models.universal_loader.get_adapter", return_value=mock_adapter_class
    ):
        returned = ul.load_model("torchvision", {"some": "cfg"})

    mock_loader.load_model.assert_called_once_with({"some": "cfg"})
    mock_loader.load_weights.assert_not_called()
    assert hasattr(returned, "model")
    assert returned.model is mock_model


def test_universal_loader_loads_weights(monkeypatch):
    mock_loader = mock.MagicMock()
    mock_model_initial = _DummyModel()
    mock_model_final = _DummyModel()
    mock_loader.load_model.return_value = mock_model_initial
    mock_loader.load_weights.return_value = mock_model_final

    mock_adapter_class = mock.MagicMock()
    mock_adapter_class.__name__ = "MockAdapter"
    mock_adapter_instance = mock.MagicMock()
    mock_adapter_instance.model = mock_model_final
    mock_adapter_class.return_value = mock_adapter_instance

    ul = _make_universal(monkeypatch, {"torchvision": mock_loader})

    with mock.patch(
        "segmentation_robustness_framework.loaders.models.universal_loader.get_adapter", return_value=mock_adapter_class
    ):
        returned = ul.load_model("torchvision", {"cfg": 1}, weights_path="weights.pth", weight_type="encoder")

    mock_loader.load_model.assert_called_once_with({"cfg": 1})
    mock_loader.load_weights.assert_called_once_with(mock_model_initial, "weights.pth", "encoder")
    assert hasattr(returned, "model")
    assert returned.model is mock_model_final


def test_universal_loader_handles_bundle(monkeypatch):
    mock_loader = mock.MagicMock()
    inner_model = _DummyModel()
    bundle = SimpleNamespace(model=inner_model)
    mock_loader.load_model.return_value = bundle

    mock_adapter_class = mock.MagicMock()
    mock_adapter_class.__name__ = "MockAdapter"
    mock_adapter_instance = mock.MagicMock()
    mock_adapter_instance.model = inner_model
    mock_adapter_class.return_value = mock_adapter_instance

    ul = _make_universal(monkeypatch, {"custom": mock_loader})

    with mock.patch(
        "segmentation_robustness_framework.loaders.models.universal_loader.get_adapter", return_value=mock_adapter_class
    ):
        returned = ul.load_model("custom", {"any": "cfg"})

    assert hasattr(returned, "model")
    assert returned.model is inner_model
    mock_loader.load_weights.assert_not_called()


def test_universal_loader_load_model_raises(monkeypatch):
    mock_loader = mock.MagicMock()
    mock_loader.load_model.side_effect = ValueError("test")

    ul = _make_universal(monkeypatch, {"torchvision": mock_loader})
    with pytest.raises(ValueError):
        ul.load_model("torchvision", {"any": "cfg"})


def test_universal_loader_load_weights_raises(monkeypatch, caplog):
    mock_loader = mock.MagicMock()
    dummy_model = _DummyModel()
    mock_loader.load_model.return_value = dummy_model
    mock_loader.load_weights.side_effect = RuntimeError("fail weights")

    ul = _make_universal(monkeypatch, {"smp": mock_loader})

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError, match="fail weights"):
            ul.load_model("smp", {"cfg": 1}, weights_path="weights.pth")

        assert any("Failed to load weights for smp model" in r.getMessage() for r in caplog.records)
