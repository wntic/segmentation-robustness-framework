import pytest
from segmentation_robustness_framework.loaders.models import CustomModelLoader

from tests.utils.dummy_model import DummyModel, DummyModelWithoutEncoder


def test_custom_model_loader_load_model():
    loader = CustomModelLoader()
    cfg = {"model_class": DummyModel, "model_args": [3, 8], "model_kwargs": {"num_classes": 21}}
    model = loader.load_model(cfg)
    assert isinstance(model, DummyModel)
    assert model.encoder.conv.in_channels == 3
    assert model.encoder.conv.out_channels == 8
    assert model.classifier.out_channels == 21


def test_custom_model_loader_load_model_with_str():
    loader = CustomModelLoader()
    cfg = {
        "model_class": "tests.utils.dummy_model.DummyModel",
        "model_args": [3, 8],
        "model_kwargs": {"num_classes": 21},
    }
    model = loader.load_model(cfg)
    assert isinstance(model, DummyModel)
    assert model.encoder.conv.in_channels == 3
    assert model.encoder.conv.out_channels == 8
    assert model.classifier.out_channels == 21


def test_custom_model_loader_load_model_raises_value_error():
    loader = CustomModelLoader()
    cfg = {"model_class": 1, "model_args": [3, 8], "model_kwargs": {"num_classes": 21}}
    with pytest.raises(ValueError):
        loader.load_model(cfg)


def test_custom_model_loader_load_weights_checkpoint():
    loader = CustomModelLoader()
    cfg = {"model_class": DummyModel, "model_args": [3, 8], "model_kwargs": {"num_classes": 21}}
    model = loader.load_model(cfg)
    weights_path = "tests/data/dummy_model_checkpoint.pth"
    loader.load_weights(model, weights_path, weight_type="full")
    assert model.encoder.conv.weight.data.shape == (8, 3, 3, 3)
    assert model.encoder.conv.bias.data.shape == (8,)
    assert model.classifier.weight.data.shape == (21, 8, 1, 1)
    assert model.classifier.bias.data.shape == (21,)


def test_custom_model_loader_load_weights_full():
    loader = CustomModelLoader()
    cfg = {"model_class": DummyModel, "model_args": [3, 8], "model_kwargs": {"num_classes": 21}}
    model = loader.load_model(cfg)
    weights_path = "tests/data/dummy_model_weights.pth"
    loader.load_weights(model, weights_path, weight_type="full")
    assert model.encoder.conv.weight.data.shape == (8, 3, 3, 3)
    assert model.encoder.conv.bias.data.shape == (8,)
    assert model.classifier.weight.data.shape == (21, 8, 1, 1)
    assert model.classifier.bias.data.shape == (21,)


def test_custom_model_loader_load_weights_encoder():
    loader = CustomModelLoader()
    cfg = {"model_class": DummyModel, "model_args": [3, 8], "model_kwargs": {"num_classes": 21}}
    model = loader.load_model(cfg)
    weights_path = "tests/data/dummy_encoder_weights.pth"
    loader.load_weights(model, weights_path, weight_type="encoder")
    assert model.encoder.conv.weight.data.shape == (8, 3, 3, 3)
    assert model.encoder.conv.bias.data.shape == (8,)
    assert model.classifier.weight.data.shape == (21, 8, 1, 1)
    assert model.classifier.bias.data.shape == (21,)


def test_custom_model_loader_load_weights_encoder_without_encoder():
    loader = CustomModelLoader()
    cfg = {"model_class": DummyModelWithoutEncoder, "model_args": [3], "model_kwargs": {"num_classes": 21}}
    model = loader.load_model(cfg)
    weights_path = "tests/data/dummy_encoder_weights.pth"
    loader.load_weights(model, weights_path, weight_type="encoder")
    assert model.classifier.weight.data.shape == (21, 3, 1, 1)
    assert model.classifier.bias.data.shape == (21,)


def test_custom_model_loader_load_weights_unknown_weight_type():
    loader = CustomModelLoader()
    cfg = {"model_class": DummyModel, "model_args": [3, 8], "model_kwargs": {"num_classes": 21}}
    model = loader.load_model(cfg)
    weights_path = "tests/data/dummy_encoder_weights.pth"
    with pytest.raises(ValueError):
        loader.load_weights(model, weights_path, weight_type="unknown")
