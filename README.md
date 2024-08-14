# Adversarial Segmentation Toolkit

[![Ruff](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/ruff/main/assets/badge/v2.json)](https://github.com/astral-sh/ruff)

A tool for evaluating the robustness of image segmentation models to adversarial attacks.

## Installation

```bash
git clone https://github.com/wntic/adversarial-segmentation-toolkit.git
```

## Usage

Using segmentation models:

```python
from adversarial_segmentation_toolkit import models, preprocess_image


model = models.FCN("resnet50")
image = preprocess_image(image_path)
output_segmentation = model(image)

```

An example of using methods to generate adversarial examples:

```python
from adversarial_segmentation_toolkit import attacks, preprocess_image


atk = attacks.FGSM(model=model, eps=0.1)
adv_image = atk.attack(image, labels)

```

## Models
- FCN (weights: resnet50, resnet101)
- DeepLabV3 (weights: resnet50, resnet101, mobilenet_v3_large)

## License

[Apache License 2.0](https://choosealicense.com/licenses/apache-2.0/)
