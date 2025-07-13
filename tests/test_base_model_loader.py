import pytest
from segmentation_robustness_framework.loaders.models import BaseModelLoader


def test_base_model_loader_is_abstract():
    with pytest.raises(TypeError):
        BaseModelLoader()  # type: ignore
