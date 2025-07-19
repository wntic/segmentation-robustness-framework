import pytest
import torch
import torch.nn as nn


class MockSegmentationModel(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        self.conv = nn.Conv2d(3, num_classes, kernel_size=3, padding=1)

    def forward(self, x):
        return self.conv(x)

    def eval(self):
        self.training = False
        return self

    def zero_grad(self):
        if self.conv.weight.grad is not None:
            self.conv.weight.grad.zero_()


@pytest.fixture
def model():
    model = MockSegmentationModel(num_classes=10)
    if torch.cuda.is_available():
        model = model.cuda()
    return model


def test_apply_with_different_batch_sizes(attack, model):
    device = next(model.parameters()).device
    batch_sizes = [1, 2, 4]

    for batch_size in batch_sizes:
        image = torch.randn(batch_size, 3, 32, 32, device=device)
        labels = torch.randint(0, 10, (batch_size, 32, 32), device=device)

        model.eval()
        result = attack.apply(image, labels)

        assert result.shape == image.shape
        assert not torch.allclose(result, image)


def test_apply_with_different_image_sizes(attack, model):
    device = next(model.parameters()).device
    sizes = [(16, 16), (32, 32), (64, 64)]

    for height, width in sizes:
        image = torch.randn(1, 3, height, width, device=device)
        labels = torch.randint(0, 10, (1, height, width), device=device)

        model.eval()
        result = attack.apply(image, labels)

        assert result.shape == image.shape
        assert not torch.allclose(result, image)


def test_apply_preserves_input(attack, model):
    device = next(model.parameters()).device
    image = torch.randn(1, 3, 32, 32, device=device)
    labels = torch.randint(0, 10, (1, 32, 32), device=device)

    image_original = image.clone()
    labels_original = labels.clone()

    model.eval()
    result = attack.apply(image, labels)

    assert torch.allclose(image, image_original)
    assert torch.allclose(labels, labels_original)
    assert not torch.allclose(result, image_original)


def test_apply_clamps_output(attack, model):
    device = next(model.parameters()).device
    image = torch.randn(1, 3, 32, 32, device=device)
    labels = torch.randint(0, 10, (1, 32, 32), device=device)

    model.eval()
    result = attack.apply(image, labels)

    assert torch.all(result >= 0)
    assert torch.all(result <= 1)


def test_apply_with_model_in_training_mode(attack, model):
    model.train()
    assert model.training

    device = next(model.parameters()).device
    image = torch.randn(1, 3, 32, 32, device=device)
    labels = torch.randint(0, 10, (1, 32, 32), device=device)

    result = attack.apply(image, labels)

    assert not model.training
    assert result.shape == image.shape


def test_apply_with_edge_case_labels(attack, model):
    device = next(model.parameters()).device
    image = torch.randn(1, 3, 32, 32, device=device)

    labels = torch.zeros(1, 32, 32, dtype=torch.long, device=device)
    model.eval()
    result = attack.apply(image, labels)
    assert result.shape == image.shape

    labels = torch.full((1, 32, 32), 9, dtype=torch.long, device=device)
    model.eval()
    result = attack.apply(image, labels)
    assert result.shape == image.shape


def test_apply_with_single_pixel_valid(attack, model):
    device = next(model.parameters()).device
    image = torch.randn(1, 3, 32, 32, device=device)
    labels = torch.full((1, 32, 32), -1, dtype=torch.long, device=device)
    labels[0, 0, 0] = 0

    model.eval()
    result = attack.apply(image, labels)

    assert result.shape == image.shape
    assert not torch.allclose(result, image)


def test_apply_with_model_output_shape(model):
    device = next(model.parameters()).device
    model_5_classes = MockSegmentationModel(num_classes=5)
    if torch.cuda.is_available():
        model_5_classes = model_5_classes.cuda()

    image = torch.randn(1, 3, 32, 32, device=device)
    labels = torch.randint(0, 5, (1, 32, 32), device=device)

    model_5_classes.eval()

    from segmentation_robustness_framework.attacks.attack import AdversarialAttack

    class TestAttack(AdversarialAttack):
        def apply(self, images, labels):
            return images + 0.1

    attack_5 = TestAttack(model_5_classes)
    result = attack_5.apply(image, labels)

    assert result.shape == image.shape


def test_apply_with_empty_batch(attack, model):
    device = next(model.parameters()).device
    image = torch.empty(0, 3, 32, 32, device=device)
    labels = torch.empty(0, 32, 32, dtype=torch.long, device=device)

    model.eval()
    result = attack.apply(image, labels)

    assert result.shape == image.shape


def test_apply_with_single_pixel_image(attack, model):
    device = next(model.parameters()).device
    image = torch.randn(1, 3, 1, 1, device=device)
    labels = torch.randint(0, 10, (1, 1, 1), device=device)

    model.eval()
    result = attack.apply(image, labels)

    assert result.shape == image.shape
    assert not torch.allclose(result, image)


def test_apply_with_very_large_image(attack, model):
    device = next(model.parameters()).device
    image = torch.randn(1, 3, 512, 512, device=device)
    labels = torch.randint(0, 10, (1, 512, 512), device=device)

    model.eval()
    result = attack.apply(image, labels)

    assert result.shape == image.shape
    assert not torch.allclose(result, image)


def test_apply_with_model_that_raises_exception():
    class FailingModel(nn.Module):
        def forward(self, x):
            raise RuntimeError("Model failed")

        def eval(self):
            return self

        def zero_grad(self):
            pass

    failing_model = FailingModel()
    if torch.cuda.is_available():
        failing_model = failing_model.cuda()

    from segmentation_robustness_framework.attacks.attack import AdversarialAttack

    class TestAttack(AdversarialAttack):
        def apply(self, images, labels):
            return images + 0.1

    attack = TestAttack(failing_model)

    device = next(failing_model.parameters()).device if list(failing_model.parameters()) else torch.device("cpu")
    image = torch.randn(1, 3, 32, 32, device=device)
    labels = torch.randint(0, 10, (1, 32, 32), device=device)

    with pytest.raises(RuntimeError, match="Model failed"):
        attack.apply(image, labels)
