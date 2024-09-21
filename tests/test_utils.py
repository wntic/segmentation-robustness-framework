import os
import tempfile

import numpy as np
import pytest
import torch
from segmentation_robustness_framework import utils
from PIL import Image


@pytest.fixture
def sample_image_path():
    image = Image.new("RGB", (100, 100), color="red")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as temp_file:
        image.save(temp_file, format="JPEG")
        temp_file_path = temp_file.name

    yield temp_file_path
    os.remove(temp_file_path)


class TestImagePreprocessing:
    def test_preprocess_image_with_shape(self, sample_image_path):
        image_tensor = utils.image_preprocessing.preprocess_image(sample_image_path, (128, 128))
        assert image_tensor.shape == (1, 3, 128, 128)
        assert isinstance(image_tensor, torch.Tensor)

    def test_preprocess_image_without_shape(self, sample_image_path):
        image_tensor = utils.image_preprocessing.preprocess_image(sample_image_path)
        assert image_tensor.shape[2] % 8 == 0
        assert image_tensor.shape[3] % 8 == 0
        assert isinstance(image_tensor, torch.Tensor)

    def test_preprocess_image_invalid_path(self):
        with pytest.raises(TypeError):
            utils.image_preprocessing.preprocess_image(123)

    def test_preprocess_image_invalid_shape(self, sample_image_path):
        with pytest.raises(ValueError):
            utils.image_preprocessing.preprocess_image(sample_image_path, (128,))
        with pytest.raises(TypeError):
            utils.image_preprocessing.preprocess_image(sample_image_path, ("128", "128"))

    def test_preprocess_image_invalid_shape_type(self, sample_image_path):
        with pytest.raises(TypeError):
            utils.image_preprocessing.preprocess_image(sample_image_path, [128, 128])


class TestDenormalize:
    def test_denormalize(self):
        image = np.ndarray(shape=(3, 4, 3), dtype=float)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        expected = std * image + mean
        result = utils.visualization.denormalize(image)
        np.testing.assert_allclose(result, expected, rtol=1e-5)
