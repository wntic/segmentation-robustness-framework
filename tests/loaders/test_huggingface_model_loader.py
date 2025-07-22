import logging
import unittest.mock as mock

import pytest
import torch
from segmentation_robustness_framework.loaders.models.hf_bundle import HFSegmentationBundle
from segmentation_robustness_framework.loaders.models.huggingface_loader import HuggingFaceModelLoader


def patch_transformers(monkeypatch, num_labels=150, processor_attrs=None):
    mock_transformers = mock.MagicMock()
    mock_model = mock.MagicMock()

    class DummyConfig:
        def __init__(self, num_labels):
            self.num_labels = num_labels

    mock_model.config = DummyConfig(num_labels)
    mock_transformers.AutoModelForSemanticSegmentation.from_pretrained.return_value = mock_model
    mock_transformers.AutoModelForInstanceSegmentation.from_pretrained.return_value = mock_model
    mock_transformers.AutoModel.from_pretrained.return_value = mock_model
    mock_transformers.AutoConfig.from_pretrained.return_value = DummyConfig(num_labels)
    mock_processor = mock.MagicMock()
    if processor_attrs:
        for k, v in processor_attrs.items():
            setattr(mock_processor, k, v)
    mock_transformers.AutoImageProcessor.from_pretrained.return_value = mock_processor
    monkeypatch.setattr(
        "segmentation_robustness_framework.loaders.models.huggingface_loader.importlib.import_module",
        lambda name: mock_transformers,
    )
    return mock_model, mock_processor


def test_import_transformers_raises_import_error():
    with mock.patch(
        "segmentation_robustness_framework.loaders.models.huggingface_loader.importlib.import_module"
    ) as mock_import:
        mock_import.side_effect = ImportError("test")
        with pytest.raises(ImportError):
            HuggingFaceModelLoader._import_transformers()
        mock_import.assert_called_once_with("transformers")


def test_import_transformers_cached_module_returned():
    with mock.patch(
        "segmentation_robustness_framework.loaders.models.huggingface_loader.importlib.import_module"
    ) as mock_import:
        assert HuggingFaceModelLoader._import_transformers() is not None


def test_huggingface_model_loader_load_model_semantic_segmentation(monkeypatch):
    mock_model, mock_processor = patch_transformers(monkeypatch, num_labels=150)
    loader = HuggingFaceModelLoader()
    cfg = {"model_name": "nvidia/segformer-b0-finetuned-ade-512-512", "num_labels": 150}
    bundle = loader.load_model(cfg)
    assert isinstance(bundle, HFSegmentationBundle)


def test_huggingface_model_loader_load_model_instance_segmentation(monkeypatch):
    mock_model, mock_processor = patch_transformers(monkeypatch, num_labels=42)
    loader = HuggingFaceModelLoader()
    cfg = {"model_name": "facebook/maskformer-swin-tiny-coco", "task": "instance_segmentation"}
    bundle = loader.load_model(cfg)
    assert isinstance(bundle, HFSegmentationBundle)


def test_huggingface_model_loader_load_model_auto_task(monkeypatch):
    mock_model, mock_processor = patch_transformers(monkeypatch, num_labels=99)
    loader = HuggingFaceModelLoader()
    cfg = {"model_name": "facebook/maskformer-swin-tiny-coco", "task": "unknown_task"}
    bundle = loader.load_model(cfg)
    assert isinstance(bundle, HFSegmentationBundle)


def test_huggingface_model_loader_load_model_with_config_overrides(monkeypatch):
    mock_model, mock_processor = patch_transformers(monkeypatch, num_labels=150)
    loader = HuggingFaceModelLoader()
    cfg = {"model_name": "facebook/maskformer-swin-tiny-coco", "config_overrides": {"num_labels": 150}}
    bundle = loader.load_model(cfg)
    assert isinstance(bundle, HFSegmentationBundle)


def test_huggingface_model_loader_load_model_with_processor_overrides(monkeypatch):
    processor_attrs = {"size": (512, 512)}
    mock_model, mock_processor = patch_transformers(monkeypatch, processor_attrs=processor_attrs)
    loader = HuggingFaceModelLoader()
    cfg = {"model_name": "facebook/maskformer-swin-tiny-coco", "processor_overrides": {"size": (512, 512)}}
    bundle = loader.load_model(cfg)
    assert isinstance(bundle, HFSegmentationBundle)
    assert hasattr(bundle.processor, "size")
    assert bundle.processor.size == (512, 512)


def test_huggingface_model_loader_load_model_without_processor(monkeypatch):
    mock_model, mock_processor = patch_transformers(monkeypatch)
    loader = HuggingFaceModelLoader()
    cfg = {"model_name": "facebook/maskformer-swin-tiny-coco", "return_processor": False}
    model = loader.load_model(cfg)
    assert not isinstance(model, HFSegmentationBundle)
    assert hasattr(model, "config")


def test_huggingface_model_loader_load_model_logs_and_reraises(monkeypatch, caplog):
    """When an internal call raises, load_model should log the error and re-raise."""

    mock_transformers = mock.MagicMock()
    mock_transformers.AutoConfig.from_pretrained.side_effect = RuntimeError("Simulated error")

    from segmentation_robustness_framework.loaders.models.huggingface_loader import HuggingFaceModelLoader as _HFL

    _HFL._hf = None

    monkeypatch.setattr(
        "segmentation_robustness_framework.loaders.models.huggingface_loader.importlib.import_module",
        lambda name: mock_transformers,
    )

    loader = HuggingFaceModelLoader()
    cfg = {"model_name": "some/model"}

    with caplog.at_level(logging.ERROR):
        with pytest.raises(RuntimeError, match="Simulated error"):
            loader.load_model(cfg)

        assert any(
            "Failed to load HuggingFace model: Simulated error" in record.getMessage() for record in caplog.records
        )


def _patch_torch_load(monkeypatch, checkpoint_dict):
    """Utility to patch torch.load and return checkpoint_dict."""

    def _fake_torch_load(path, map_location="cpu"):
        _fake_torch_load.called_with_path = path  # type: ignore[attr-defined]
        return checkpoint_dict

    monkeypatch.setattr("torch.load", _fake_torch_load)
    return _fake_torch_load


def test_huggingface_model_loader_load_weights_state_dict(monkeypatch):
    """load_weights should extract checkpoint['state_dict'] and pass it to model.load_state_dict"""
    ckpt = {"state_dict": {"layer.weight": torch.randn(1)}}
    _fake_load = _patch_torch_load(monkeypatch, ckpt)

    mock_model = mock.MagicMock()
    loader = HuggingFaceModelLoader()
    returned = loader.load_weights(mock_model, "dummy_path.pth")

    mock_model.load_state_dict.assert_called_once()
    args, kwargs = mock_model.load_state_dict.call_args
    assert args[0] == ckpt["state_dict"]
    assert kwargs.get("strict") is False
    assert _fake_load.called_with_path == "dummy_path.pth"
    assert returned is mock_model


def test_huggingface_model_loader_load_weights_raw_state_dict(monkeypatch):
    """load_weights should handle checkpoints that are already state dicts."""
    ckpt = {"encoder.weight": torch.randn(1)}
    _patch_torch_load(monkeypatch, ckpt)

    mock_model = mock.MagicMock()
    loader = HuggingFaceModelLoader()
    loader.load_weights(mock_model, "another_dummy.pth")

    mock_model.load_state_dict.assert_called_once()
    args, kwargs = mock_model.load_state_dict.call_args
    assert args[0] == ckpt
    assert kwargs.get("strict") is False


def test_huggingface_model_loader_load_weights_raises_exception(monkeypatch):
    mock_model = mock.MagicMock()
    loader = HuggingFaceModelLoader()
    with pytest.raises(Exception):
        loader.load_weights(mock_model, "dummy_path.pth")
