import os
import tempfile

import numpy as np
import pytest
import torch
from adversarial_segmentation_toolkit.utils.image_preprocessing import preprocess_image
from adversarial_segmentation_toolkit.utils.image_utils import denormalize
from PIL import Image

# from adversarial_segmentation_toolkit.utils.visualization import visualize_segmentation


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
        image_tensor = preprocess_image(sample_image_path, (128, 128))
        assert image_tensor.shape == (1, 3, 128, 128)
        assert isinstance(image_tensor, torch.Tensor)

    def test_preprocess_image_without_shape(self, sample_image_path):
        image_tensor = preprocess_image(sample_image_path)
        assert image_tensor.shape[2] % 8 == 0
        assert image_tensor.shape[3] % 8 == 0
        assert isinstance(image_tensor, torch.Tensor)

    def test_preprocess_image_invalid_path(self):
        with pytest.raises(TypeError):
            preprocess_image(123)

    def test_preprocess_image_invalid_shape(self, sample_image_path):
        with pytest.raises(ValueError):
            preprocess_image(sample_image_path, (128,))
        with pytest.raises(TypeError):
            preprocess_image(sample_image_path, ("128", "128"))

    def test_preprocess_image_invalid_shape_type(self, sample_image_path):
        with pytest.raises(TypeError):
            preprocess_image(sample_image_path, [128, 128])


class TestDenormalize:
    def test_denormalize(self):
        image = np.ndarray(shape=(3, 4, 3), dtype=float)
        mean = np.array([0.485, 0.456, 0.406])
        std = np.array([0.229, 0.224, 0.225])

        expected = std * image + mean
        result = denormalize(image)
        np.testing.assert_allclose(result, expected, rtol=1e-5)
