from unittest.mock import MagicMock, patch

import pytest
from segmentation_robustness_framework.utils.loader_utils import resolve_model_class


def test_resolve_model_class_with_dot_notation():
    result = resolve_model_class("torch.nn.Conv2d")

    import torch.nn

    assert result == torch.nn.Conv2d


def test_resolve_model_class_without_dot_notation():
    result = resolve_model_class("MyCustomClass")
    assert result == "MyCustomClass"


def test_resolve_model_class_with_builtin_module():
    result = resolve_model_class("collections.OrderedDict")

    from collections import OrderedDict

    assert result == OrderedDict


def test_resolve_model_class_with_custom_module():
    mock_module = MagicMock()
    mock_class = MagicMock()
    mock_module.TestClass = mock_class

    with patch("importlib.import_module", return_value=mock_module):
        result = resolve_model_class("test_module.TestClass")

    assert result == mock_class


def test_resolve_model_class_with_multiple_dots():
    mock_module = MagicMock()
    mock_class = MagicMock()
    mock_module.DeepClass = mock_class

    with patch("importlib.import_module", return_value=mock_module):
        result = resolve_model_class("deep.nested.module.DeepClass")

    assert result == mock_class


def test_resolve_model_class_import_error():
    with patch("importlib.import_module", side_effect=ImportError("No module named 'nonexistent'")):
        with pytest.raises(ImportError, match="No module named 'nonexistent'"):
            resolve_model_class("nonexistent.module.Class")


def test_resolve_model_class_attribute_error():
    class MockModule:
        pass

    mock_module = MockModule()

    with patch("importlib.import_module", return_value=mock_module):
        with pytest.raises(AttributeError):
            resolve_model_class("existing_module.NonexistentClass")


def test_resolve_model_class_empty_string():
    result = resolve_model_class("")
    assert result == ""


def test_resolve_model_class_single_dot():
    with pytest.raises(ValueError, match="Empty module name"):
        resolve_model_class(".")


def test_resolve_model_class_ends_with_dot():
    with pytest.raises(ModuleNotFoundError, match="No module named 'module'"):
        resolve_model_class("module.")


def test_resolve_model_class_starts_with_dot():
    with pytest.raises(ValueError, match="Empty module name"):
        resolve_model_class(".class")


def test_resolve_model_class_with_underscores():
    mock_module = MagicMock()
    mock_class = MagicMock()
    mock_module.My_Test_Class = mock_class

    with patch("importlib.import_module", return_value=mock_module):
        result = resolve_model_class("test_module.My_Test_Class")

    assert result == mock_class


def test_resolve_model_class_with_numbers():
    mock_module = MagicMock()
    mock_class = MagicMock()
    mock_module.Class123 = mock_class

    with patch("importlib.import_module", return_value=mock_module):
        result = resolve_model_class("test_module.Class123")

    assert result == mock_class


def test_resolve_model_class_importlib_called_correctly():
    with patch("importlib.import_module") as mock_import:
        mock_module = MagicMock()
        mock_class = MagicMock()
        mock_module.TestClass = mock_class
        mock_import.return_value = mock_module

        resolve_model_class("my.module.TestClass")

        mock_import.assert_called_once_with("my.module")


def test_resolve_model_class_with_special_characters():
    with pytest.raises(ModuleNotFoundError, match="No module named 'module'"):
        resolve_model_class("module.Class$Name")


def test_resolve_model_class_unicode():
    with pytest.raises(ModuleNotFoundError, match="No module named 'module'"):
        resolve_model_class("module.ClaßName")


def test_resolve_model_class_very_long_path():
    mock_module = MagicMock()
    mock_class = MagicMock()
    mock_module.LongClassName = mock_class

    long_path = "very.deeply.nested.module.path.with.many.levels.LongClassName"

    with patch("importlib.import_module", return_value=mock_module):
        result = resolve_model_class(long_path)

    assert result == mock_class
