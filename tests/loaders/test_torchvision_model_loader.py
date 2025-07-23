import logging
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import pytest
import torch
from segmentation_robustness_framework.loaders.models.torchvision_loader import TorchvisionModelLoader
from torch import nn


class _DummyWeights:
    DEFAULT = object()


def _make_dummy_model(classifier_type: str = "sequential", out_channels: int = 21):
    if classifier_type == "sequential":
        classifier = nn.Sequential(nn.Conv2d(4, out_channels, 1))
    elif classifier_type == "conv":
        classifier = nn.Conv2d(4, out_channels, 1)
    else:
        raise ValueError("Unsupported classifier_type")

    model = SimpleNamespace()
    model.classifier = classifier
    model.backbone = mock.MagicMock()
    return model


def _patch_supported(monkeypatch, name: str, model):
    fn = mock.MagicMock(return_value=model)
    monkeypatch.setitem(TorchvisionModelLoader.SUPPORTED_MODELS, name, fn)
    monkeypatch.setitem(TorchvisionModelLoader.TORCHVISION_WEIGHTS_ENUMS, name, _DummyWeights)
    return fn


def _patch_torch_load(monkeypatch, checkpoint):
    def _fake_load(path, map_location="cpu", weights_only=True):
        _fake_load.called_with_path = path  # type: ignore[attr-defined]
        return checkpoint

    monkeypatch.setattr("torch.load", _fake_load)
    return _fake_load


def test_torchvision_loader_unsupported_model_raises(monkeypatch):
    loader = TorchvisionModelLoader()
    with pytest.raises(ValueError):
        loader.load_model({"name": "nonexistent_model"})


def test_torchvision_loader_load_model_uses_default_weights(monkeypatch):
    name = "deeplabv3_resnet50"
    mock_fn = _patch_supported(monkeypatch, name, _make_dummy_model())

    loader = TorchvisionModelLoader()
    cfg = {"name": name}
    loader.load_model(cfg)

    kwargs = mock_fn.call_args.kwargs
    assert kwargs["weights"] is _DummyWeights.DEFAULT
    assert kwargs["num_classes"] == 21


def test_torchvision_loader_load_model_weights_string_DEFAULT(monkeypatch):
    name = "deeplabv3_resnet50"
    mock_fn = _patch_supported(monkeypatch, name, _make_dummy_model())

    loader = TorchvisionModelLoader()
    cfg = {"name": name, "weights": "DEFAULT"}
    loader.load_model(cfg)

    kwargs = mock_fn.call_args.kwargs
    assert kwargs["weights"] is _DummyWeights.DEFAULT


def test_torchvision_loader_load_model_adjusts_classifier(monkeypatch):
    name = "deeplabv3_resnet50"
    model = _make_dummy_model(out_channels=21)
    _patch_supported(monkeypatch, name, model)

    loader = TorchvisionModelLoader()
    cfg = {"name": name, "num_classes": 3}
    returned = loader.load_model(cfg)

    conv = returned.classifier[-1]  # type: ignore[index]
    assert isinstance(conv, nn.Conv2d)
    assert conv.out_channels == 3


def test_torchvision_loader_load_weights_full(monkeypatch):
    state = {"layer.weight": torch.randn(1)}
    _patch_torch_load(monkeypatch, {"state_dict": state})

    model = mock.MagicMock()
    model.load_state_dict.return_value = ([], [])

    loader = TorchvisionModelLoader()
    returned = loader.load_weights(model, "weights.pth", weight_type="full")

    model.load_state_dict.assert_called_once_with(state, strict=False)
    assert returned is model


def test_torchvision_loader_load_weights_full_missing_unexpected(monkeypatch, caplog):
    _patch_torch_load(monkeypatch, {"state_dict": {}})
    model = mock.MagicMock()
    model.load_state_dict.return_value = (["missing.key"], ["unexpected.key"])

    loader = TorchvisionModelLoader()
    with caplog.at_level(logging.WARNING):
        loader.load_weights(model, "dummy", weight_type="full")
        assert any("Missing keys when loading weights" in r.getMessage() for r in caplog.records)
        assert any("Unexpected keys when loading weights" in r.getMessage() for r in caplog.records)


def test_torchvision_loader_load_weights_encoder(monkeypatch):
    enc_state = {
        "backbone.conv.weight": torch.randn(1),
        "backbone.conv.bias": torch.randn(1),
    }
    _patch_torch_load(monkeypatch, enc_state)

    backbone = mock.MagicMock()
    model = SimpleNamespace(backbone=backbone)
    backbone.load_state_dict.return_value = ([], [])

    loader = TorchvisionModelLoader()
    loader.load_weights(model, "enc.pth", weight_type="encoder")

    args, kwargs = backbone.load_state_dict.call_args
    assert all(not k.startswith("backbone.") for k in args[0].keys())
    assert kwargs.get("strict") is False


def test_torchvision_loader_load_weights_encoder_missing_unexpected(monkeypatch, caplog):
    enc_state = {"backbone.conv.weight": torch.randn(1)}
    _patch_torch_load(monkeypatch, enc_state)

    backbone = mock.MagicMock()
    backbone.load_state_dict.return_value = (["missing"], ["unexpected"])
    model = SimpleNamespace(backbone=backbone)

    loader = TorchvisionModelLoader()
    with caplog.at_level(logging.WARNING):
        loader.load_weights(model, "enc.pth", weight_type="encoder")
        assert any("Missing keys when loading encoder weights" in r.getMessage() for r in caplog.records)
        assert any("Unexpected keys when loading encoder weights" in r.getMessage() for r in caplog.records)


def test_torchvision_loader_load_weights_unknown_type(monkeypatch):
    _patch_torch_load(monkeypatch, {})
    model = mock.MagicMock()

    loader = TorchvisionModelLoader()
    loader.load_weights(model, "some.pth", weight_type="unknown")

    model.load_state_dict.assert_not_called()


def test_torchvision_loader_load_weights_raises(monkeypatch):
    monkeypatch.setattr("torch.load", lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("fail")))
    model = mock.MagicMock()
    loader = TorchvisionModelLoader()

    with pytest.raises(RuntimeError):
        loader.load_weights(model, "bad.pth")


def test_torchvision_loader_lraspp_default_num_classes(monkeypatch):
    name = "lraspp_mobilenet_v3_large"
    model = _make_dummy_model(out_channels=21)
    _patch_supported(monkeypatch, name, model)

    loader = TorchvisionModelLoader()
    cfg = {"name": name}
    returned = loader.load_model(cfg)

    conv = returned.classifier[-1] if isinstance(returned.classifier, nn.Sequential) else returned.classifier  # type: ignore[index]
    assert isinstance(conv, nn.Conv2d)
    assert conv.out_channels == 21


def test_torchvision_loader_unknown_classifier_logs_warning(monkeypatch, caplog):
    name = "deeplabv3_resnet50"
    model = _make_dummy_model(classifier_type="conv", out_channels=21)
    _patch_supported(monkeypatch, name, model)

    loader = TorchvisionModelLoader()
    cfg = {"name": name, "num_classes": 5}

    with caplog.at_level(logging.WARNING):
        returned = loader.load_model(cfg)
        assert any("Unknown classifier type" in r.getMessage() for r in caplog.records)

    conv = returned.classifier if isinstance(returned.classifier, nn.Conv2d) else returned.classifier[-1]  # type: ignore[index]
    assert conv.out_channels == 21


class _NoDefault:
    pass


def test_torchvision_loader_weights_default_fallback_branch(monkeypatch):
    name = "deeplabv3_resnet50"
    model = _make_dummy_model()
    fn = mock.MagicMock(return_value=model)
    monkeypatch.setitem(TorchvisionModelLoader.SUPPORTED_MODELS, name, fn)
    monkeypatch.setitem(TorchvisionModelLoader.TORCHVISION_WEIGHTS_ENUMS, name, _NoDefault)

    loader = TorchvisionModelLoader()
    cfg = {"name": name, "weights": "default"}
    with pytest.raises(ValueError, match="Invalid weights: default"):
        loader.load_model(cfg)


class _DummyAuxModel(nn.Module):
    def __init__(self, out_channels):
        super().__init__()
        self.aux_classifier = nn.Sequential(nn.Conv2d(10, out_channels, 1))


class _DummyTVModel(nn.Module):
    def __init__(self, out_channels=21):
        super().__init__()
        self.classifier = nn.Sequential(nn.Conv2d(10, out_channels, 1))

    def forward(self, x):
        return x


def test_torchvision_loader_aux_classifier_branch(monkeypatch):
    def fake_model_fn(*, weights=None, num_classes=21):
        return _DummyAuxModel(out_channels=num_classes)

    loader = TorchvisionModelLoader()
    monkeypatch.setitem(loader.SUPPORTED_MODELS, "deeplabv3_resnet101", fake_model_fn)
    monkeypatch.setitem(loader.TORCHVISION_WEIGHTS_ENUMS, "deeplabv3_resnet101", None)
    cfg = {"name": "deeplabv3_resnet101", "num_classes": 5, "weights": None}
    model = loader.load_model(cfg)
    if isinstance(model.aux_classifier, nn.Sequential):
        layer = model.aux_classifier[-1]
    else:
        layer = model.aux_classifier
    assert layer.out_channels == 5


def test_torchvision_loader_weights_string(monkeypatch):
    loader = TorchvisionModelLoader()
    captured = {}

    class _Enum:
        CUSTOM = "custom_weight_enum"

    def fake_model_fn(*, weights=None, num_classes=21):
        captured["weights"] = weights
        return _DummyTVModel(out_channels=num_classes)

    monkeypatch.setitem(loader.SUPPORTED_MODELS, "fcn_resnet101", fake_model_fn)
    monkeypatch.setitem(loader.TORCHVISION_WEIGHTS_ENUMS, "fcn_resnet101", _Enum)
    cfg = {"name": "fcn_resnet101", "num_classes": 21, "weights": "CUSTOM"}
    loader.load_model(cfg)
    assert captured["weights"] == _Enum.CUSTOM


def test_torchvision_load_weights_state_dict(monkeypatch, tmp_path: Path):
    loader = TorchvisionModelLoader()
    model = _DummyTVModel()
    dummy_sd = {"some.weight": torch.zeros(1)}

    def fake_load_state_dict(sd, strict=False):
        fake_load_state_dict.called = sd
        return [], []

    fake_load_state_dict.called = None
    model.load_state_dict = fake_load_state_dict  # type: ignore[assignment]
    monkeypatch.setattr(torch, "load", lambda *a, **kw: {"state_dict": dummy_sd})
    loader.load_weights(model, "dummy_path.pth", weight_type="full")
    assert fake_load_state_dict.called == dummy_sd


@pytest.mark.parametrize(
    "model_name,num_classes",
    [
        ("fcn_resnet50", 21),
        ("fcn_resnet50", 5),
        ("deeplabv3_resnet50", 3),
        ("lraspp_mobilenet_v3_large", 21),
    ],
)
def test_torchvision_loader_num_classes(model_name: str, num_classes: int):
    loader = TorchvisionModelLoader()
    cfg = {"name": model_name, "num_classes": num_classes, "weights": None}
    model = loader.load_model(cfg)
    if hasattr(model, "classifier"):
        cls = model.classifier
        if isinstance(cls, nn.Sequential):
            layer = cls[-1]
            assert isinstance(layer, nn.Conv2d)
            assert layer.out_channels == num_classes
        elif hasattr(cls, "high_classifier"):
            layer = cls.high_classifier  # LRASPP case
            assert isinstance(layer, nn.Conv2d)
            assert layer.out_channels == num_classes
    elif hasattr(model, "aux_classifier"):
        aux = model.aux_classifier
        if isinstance(aux, nn.Sequential):
            layer = aux[-1]
        else:
            layer = aux
        assert isinstance(layer, nn.Conv2d)
        assert layer.out_channels == num_classes


def test_torchvision_loader_encoder_weights(tmp_path: Path):
    loader = TorchvisionModelLoader()
    cfg = {"name": "fcn_resnet50", "num_classes": 21, "weights": None}
    model = loader.load_model(cfg)
    backbone_state_dict = {k: v.clone() for k, v in model.state_dict().items() if k.startswith("backbone.")}
    ckpt_path = tmp_path / "encoder.pth"
    torch.save(backbone_state_dict, ckpt_path)
    for k, p in model.backbone.state_dict().items():
        p.zero_()
    loader.load_weights(model, str(ckpt_path), weight_type="encoder")
    assert any(p.abs().sum() > 0 for p in model.backbone.parameters())


def test_torchvision_loader_weights_enum_default(monkeypatch):
    loader = TorchvisionModelLoader()
    captured = {}

    class _Enum:
        DEFAULT = "enum_default"

    def fake_model_fn(*, weights=None, num_classes=21):
        captured["weights"] = weights
        return _DummyTVModel(out_channels=num_classes)

    monkeypatch.setitem(loader.SUPPORTED_MODELS, "fcn_resnet50", fake_model_fn)
    monkeypatch.setitem(loader.TORCHVISION_WEIGHTS_ENUMS, "fcn_resnet50", _Enum)
    cfg = {"name": "fcn_resnet50", "num_classes": 21}
    loader.load_model(cfg)
    assert captured["weights"] == _Enum.DEFAULT


def test_torchvision_loader_low_high_classifier_rewire(monkeypatch):
    class _DummyModel:
        def __init__(self):
            self.classifier = SimpleNamespace(
                low_classifier=nn.Conv2d(4, 21, 1),
                high_classifier=nn.Conv2d(4, 21, 1),
            )

    def fake_model_fn(*, weights=None, num_classes=21):
        return _DummyModel()

    loader = TorchvisionModelLoader()
    monkeypatch.setitem(loader.SUPPORTED_MODELS, "fcn_resnet50", fake_model_fn)
    monkeypatch.setitem(loader.TORCHVISION_WEIGHTS_ENUMS, "fcn_resnet50", None)
    cfg = {"name": "fcn_resnet50", "num_classes": 5, "weights": None}
    model = loader.load_model(cfg)
    assert model.classifier.low_classifier.out_channels == 5
    assert model.classifier.high_classifier.out_channels == 5


def test_torchvision_loader_load_weights_model_key(monkeypatch, tmp_path: Path):
    loader = TorchvisionModelLoader()
    model = _DummyTVModel()
    dummy_sd = {"some.weight": torch.zeros(1)}

    def fake_load_state_dict(sd, strict=False):
        fake_load_state_dict.called = sd
        return [], []

    fake_load_state_dict.called = None
    model.load_state_dict = fake_load_state_dict  # type: ignore[assignment]
    monkeypatch.setattr(torch, "load", lambda *a, **kw: {"model": dummy_sd})
    loader.load_weights(model, "dummy_path.pth", weight_type="full")
    assert fake_load_state_dict.called == dummy_sd
