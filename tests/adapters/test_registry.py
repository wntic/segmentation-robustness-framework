import pytest
from segmentation_robustness_framework.adapters.registry import get_adapter, register_adapter


def test_register_adapter():
    @register_adapter("test_adapter1")
    class TestAdapter1:
        pass

    assert get_adapter("test_adapter1") is TestAdapter1


def test_get_adapter():
    @register_adapter("test_adapter2")
    class TestAdapter2:
        pass

    assert get_adapter("test_adapter2") is TestAdapter2


def test_alredy_registered():
    with pytest.raises(ValueError):

        @register_adapter("test_adapter3")
        class TestAdapter3:
            pass

        @register_adapter("test_adapter3")
        class TestAdapter4:
            pass


def test_get_adapter_not_registered():
    with pytest.raises(KeyError):
        get_adapter("not_registered_adapter")
