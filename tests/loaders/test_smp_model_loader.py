import logging
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from segmentation_robustness_framework.loaders.models.smp_loader import SMPModelLoader


class FakeResult:
    def __init__(self, missing_keys=None, unexpected_keys=None):
        self.missing_keys = missing_keys or []
        self.unexpected_keys = unexpected_keys or []


def _patch_smp(monkeypatch, model=None):
    if model is None:
        model = mock.MagicMock(spec=torch.nn.Module)
        model.classifier = mock.MagicMock()
        model.classifier.out_channels = 1
        model.classifier.in_channels = 4
        model.encoder = mock.MagicMock()

    mock_smp = mock.MagicMock()
    mock_smp.from_pretrained.return_value = model
    mock_smp.create_model.return_value = model

    monkeypatch.setattr(
        "segmentation_robustness_framework.loaders.models.smp_loader.importlib.import_module",
        lambda name: mock_smp,
    )

    SMPModelLoader._smp = None  # type: ignore
    return mock_smp, model


def _patch_torch_load(monkeypatch, checkpoint):
    def _fake_torch_load(path, map_location="cpu", weights_only=True):
        _fake_torch_load.called_with_path = path  # type: ignore[attr-defined]
        return checkpoint

    monkeypatch.setattr("torch.load", _fake_torch_load)
    return _fake_torch_load


def test_import_smp_raises_import_error(monkeypatch):
    monkeypatch.setattr(
        "segmentation_robustness_framework.loaders.models.smp_loader.importlib.import_module",
        lambda name: (_ for _ in ()).throw(ImportError("not installed")),
    )
    SMPModelLoader._smp = None
    with pytest.raises(ImportError):
        SMPModelLoader._import_smp()


def test_import_smp_cached_module_returned(monkeypatch):
    mock_smp = mock.MagicMock()
    monkeypatch.setattr(
        "segmentation_robustness_framework.loaders.models.smp_loader.importlib.import_module",
        lambda name: mock_smp,
    )
    SMPModelLoader._smp = None
    first = SMPModelLoader._import_smp()
    second = SMPModelLoader._import_smp()
    assert first is second is mock_smp


def test_smp_model_loader_load_model_from_checkpoint(monkeypatch):
    mock_smp, model = _patch_smp(monkeypatch)
    loader = SMPModelLoader()
    cfg = {"checkpoint": "smp-hub/upernet-convnext-tiny"}

    returned = loader.load_model(cfg)

    mock_smp.from_pretrained.assert_called_once_with("smp-hub/upernet-convnext-tiny")
    assert returned is model


def test_smp_model_loader_load_model_create(monkeypatch):
    mock_smp, model = _patch_smp(monkeypatch)
    cfg = {
        "architecture": "unet",
        "encoder_name": "resnet50",
        "encoder_weights": "imagenet",
        "classes": 3,
        "activation": None,
    }
    loader = SMPModelLoader()
    returned = loader.load_model(cfg)

    mock_smp.create_model.assert_called_once_with(
        arch="unet",
        encoder_name="resnet50",
        encoder_weights="imagenet",
        classes=3,
        activation=None,
    )
    assert returned is model


def test_smp_model_loader_adjusts_classifier_out_channels(monkeypatch):
    conv = torch.nn.Conv2d(4, 1, 1)
    model = SimpleNamespace(
        classifier=conv,
        encoder=mock.MagicMock(),
    )

    mock_smp, _ = _patch_smp(monkeypatch, model=model)

    cfg = {
        "architecture": "unet",
        "encoder_name": "resnet34",
        "classes": 5,
    }

    loader = SMPModelLoader()
    returned = loader.load_model(cfg)

    assert returned.classifier.out_channels == 5


def test_smp_model_loader_checkpoint_error_reraised(monkeypatch, caplog):
    model = mock.MagicMock()
    mock_smp = mock.MagicMock()
    mock_smp.from_pretrained.side_effect = RuntimeError("broken")
    monkeypatch.setattr(
        "segmentation_robustness_framework.loaders.models.smp_loader.importlib.import_module",
        lambda name: mock_smp,
    )
    SMPModelLoader._smp = None

    loader = SMPModelLoader()
    cfg = {"checkpoint": "smp-hub/bogus"}

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError, match="Could not load checkpoint"):
            loader.load_model(cfg)

        assert any("Could not load checkpoint" in r.getMessage() for r in caplog.records)


def test_smp_model_loader_load_weights_full(monkeypatch):
    state = {"layer.weight": torch.randn(1)}
    _patch_torch_load(monkeypatch, {"state_dict": state})

    model = mock.MagicMock()
    model.load_state_dict.return_value = ([], [])

    loader = SMPModelLoader()
    returned = loader.load_weights(model, "weights.pth", weight_type="full")

    model.load_state_dict.assert_called_once_with(state, strict=False)
    assert returned is model


def test_smp_model_loader_load_weights_encoder(monkeypatch):
    enc_state = {
        "encoder.conv.weight": torch.randn(1),
        "encoder.conv.bias": torch.randn(1),
        "something_else": torch.randn(1),  # should be ignored
    }
    _patch_torch_load(monkeypatch, enc_state)

    encoder = mock.MagicMock()
    model = SimpleNamespace(encoder=encoder)
    encoder.load_state_dict.return_value = ([], [])

    loader = SMPModelLoader()
    loader.load_weights(model, "enc.pth", weight_type="encoder")

    args, kwargs = encoder.load_state_dict.call_args
    assert all(not k.startswith("encoder.") for k in args[0].keys())
    assert kwargs.get("strict") is False


def test_smp_model_loader_load_weights_unknown(monkeypatch):
    _patch_torch_load(monkeypatch, {})
    model = mock.MagicMock()

    loader = SMPModelLoader()
    loader.load_weights(model, "some.pth", weight_type="unknown")

    model.load_state_dict.assert_not_called()


def test_smp_model_loader_load_weights_raises(monkeypatch):
    monkeypatch.setattr("torch.load", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("fail")))
    model = mock.MagicMock()
    loader = SMPModelLoader()

    with pytest.raises(RuntimeError):
        loader.load_weights(model, "bad.pth")


def test_smp_model_loader_load_weights_full_logs_missing_keys(monkeypatch, caplog):
    _patch_torch_load(monkeypatch, {"state_dict": {}})

    model = mock.MagicMock()
    model.load_state_dict.return_value = FakeResult(missing_keys=["missing.key"], unexpected_keys=[])

    loader = SMPModelLoader()
    with caplog.at_level(logging.WARNING):
        loader.load_weights(model, "dummy.pth", weight_type="full")

        assert any("Missing keys when loading full model weights" in r.getMessage() for r in caplog.records)


def test_smp_model_loader_load_weights_full_logs_unexpected_keys(monkeypatch, caplog):
    _patch_torch_load(monkeypatch, {"state_dict": {}})

    model = mock.MagicMock()
    model.load_state_dict.return_value = FakeResult(missing_keys=[], unexpected_keys=["unexpected.key"])

    loader = SMPModelLoader()
    with caplog.at_level(logging.WARNING):
        loader.load_weights(model, "dummy.pth", weight_type="full")

        assert any("Unexpected keys when loading full model weights" in r.getMessage() for r in caplog.records)


def test_smp_model_loader_load_weights_encoder_logs_missing_keys(monkeypatch, caplog):
    enc_state = {"encoder.dummy": torch.randn(1)}
    _patch_torch_load(monkeypatch, enc_state)

    encoder = mock.MagicMock()
    encoder.load_state_dict.return_value = FakeResult(missing_keys=["missing.key"], unexpected_keys=[])
    model = SimpleNamespace(encoder=encoder)

    loader = SMPModelLoader()
    with caplog.at_level(logging.WARNING):
        loader.load_weights(model, "enc.pth", weight_type="encoder")

        assert any("Missing keys when loading encoder weights" in r.getMessage() for r in caplog.records)


def test_smp_model_loader_load_weights_encoder_logs_unexpected_keys(monkeypatch, caplog):
    enc_state = {"encoder.dummy": torch.randn(1)}
    _patch_torch_load(monkeypatch, enc_state)

    encoder = mock.MagicMock()
    encoder.load_state_dict.return_value = FakeResult(missing_keys=[], unexpected_keys=["unexpected.key"])
    model = SimpleNamespace(encoder=encoder)

    loader = SMPModelLoader()
    with caplog.at_level(logging.WARNING):
        loader.load_weights(model, "enc.pth", weight_type="encoder")

        assert any("Unexpected keys when loading encoder weights" in r.getMessage() for r in caplog.records)
