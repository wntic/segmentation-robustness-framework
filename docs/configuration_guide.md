# Configuration Guide

This guide explains how to configure attacks and datasets for the Segmentation Robustness Framework using the new Pydantic-based configuration system.

## Table of Contents

- [Overview](#overview)
- [Attack Configurations](#attack-configurations)
- [Dataset Configurations](#dataset-configurations)
- [Complete Configuration Examples](#complete-configuration-examples)
- [Validation and Error Handling](#validation-and-error-handling)
- [Integration with Framework](#integration-with-framework)

## Overview

The framework provides type-safe configuration classes for attacks and datasets using Pydantic models. This ensures:

- **Type Safety**: All parameters are validated at runtime
- **Documentation**: Self-documenting configurations with clear field descriptions
- **Flexibility**: Support for single values or lists for parameter sweeping
- **Validation**: Automatic validation of parameter constraints

## Attack Configurations

### Supported Attack Types

1. **FGSM** (Fast Gradient Sign Method)
2. **PGD** (Projected Gradient Descent)
3. **R+FGSM** (Random Fast Gradient Sign Method)
4. **TPGD** (PGD based on KL-Divergence loss)

### FGSM Configuration

```python
from segmentation_robustness_framework.config import FGSMConfig

# Single epsilon value
fgsm_config = FGSMConfig(
    type="fgsm",
    epsilon=0.1
)

# Multiple epsilon values for parameter sweeping
fgsm_config = FGSMConfig(
    type="fgsm",
    epsilon=[0.01, 0.05, 0.1, 0.2]
)
```

**Parameters:**
- `epsilon` (float | list[float]): Maximum perturbation magnitude
- `targeted` (bool): Always False (FGSM doesn't support targeted attacks)

### PGD Configuration

```python
from segmentation_robustness_framework.config import PGDConfig

# Untargeted PGD
pgd_config = PGDConfig(
    type="pgd",
    epsilon=[0.05, 0.1, 0.2],
    alpha=[0.01, 0.02],
    steps=10,
    targeted=False
)

# Targeted PGD
pgd_targeted = PGDConfig(
    type="pgd",
    epsilon=[0.05, 0.1],
    alpha=0.01,
    steps=20,
    targeted=True,
    target_label=15  # Required for targeted attacks
)
```

**Parameters:**
- `epsilon` (float | list[float]): Maximum perturbation magnitude
- `alpha` (float | list[float]): Step size for each iteration
- `steps` (int): Number of iterations (≥ 1)
- `targeted` (bool): Whether to perform targeted attack
- `target_label` (int): Target class label (required if targeted=True)

### R+FGSM Configuration

```python
from segmentation_robustness_framework.config import RFGSMConfig

rfgsm_config = RFGSMConfig(
    type="rfgsm",
    epsilon=[0.05, 0.1, 0.2],
    alpha=0.02,
    steps=5,
    targeted=False
)
```

**Parameters:**
- `epsilon` (float | list[float]): Maximum perturbation magnitude
- `alpha` (float | list[float]): Step size for random initialization
- `steps` (int): Number of iterations (≥ 1)
- `targeted` (bool): Whether to perform targeted attack
- `target_label` (int): Target class label (required if targeted=True)

### TPGD Configuration

```python
from segmentation_robustness_framework.config import TPGDConfig

tpgd_config = TPGDConfig(
    type="tpgd",
    epsilon=[0.05, 0.1],
    alpha=0.01,
    steps=15
)
```

**Parameters:**
- `epsilon` (float | list[float]): Maximum perturbation magnitude
- `alpha` (float | list[float]): Step size for each iteration
- `steps` (int): Number of iterations (≥ 1)

## Dataset Configurations

### Supported Dataset Types

1. **VOC** (Pascal VOC 2012)
2. **Cityscapes**
3. **ADE20K**
4. **Stanford Background**

### VOC Configuration

```python
from segmentation_robustness_framework.config import VOCConfig

voc_config = VOCConfig(
    type="voc",
    root="/path/to/VOCdevkit/VOC2012/",
    split="val",  # "train", "val", or "trainval"
    image_shape=[512, 512],
    max_images=100,  # Optional: limit number of images
    download=True    # Optional: auto-download if not present
)
```

**Parameters:**
- `root` (str | Path): Path to VOC2012 directory
- `split` (str): Dataset split ("train", "val", "trainval")
- `image_shape` (list[int]): Desired image shape [height, width]
- `max_images` (int | None): Maximum number of images to process
- `download` (bool): Whether to download dataset if not present

### Cityscapes Configuration

```python
from segmentation_robustness_framework.config import CityscapesConfig

# NOTE: Cityscapes cannot be downloaded automatically due to required authorization on the official website.
# You must register and download the dataset manually from https://www.cityscapes-dataset.com/ and place it in the specified root directory.
cityscapes_config = CityscapesConfig(
    type="cityscapes",
    root="/path/to/cityscapes/",  # <-- Place manually downloaded Cityscapes data here
    split="val",  # "train", "val", or "test"
    mode="fine",  # "fine" or "coarse"
    image_shape=[1024, 2048],
    max_images=50
)
```

**Parameters:**
- `root` (str | Path): Path to Cityscapes directory
- `split` (str): Dataset split ("train", "val", "test")
- `mode` (str): Dataset mode ("fine" or "coarse")
- `image_shape` (list[int]): Desired image shape [height, width]
- `max_images` (int | None): Maximum number of images to process

### ADE20K Configuration

```python
from segmentation_robustness_framework.config import ADE20KConfig

ade20k_config = ADE20KConfig(
    type="ade20k",
    root="/path/to/ADEChallengeData2016/",
    split="val",  # "train" or "val"
    image_shape=[512, 512],
    max_images=200
)
```

**Parameters:**
- `root` (str | Path): Path to ADE20K directory
- `split` (str): Dataset split ("train" or "val")
- `image_shape` (list[int]): Desired image shape [height, width]
- `max_images` (int | None): Maximum number of images to process

### Stanford Background Configuration

```python
from segmentation_robustness_framework.config import StanfordBackgroundConfig

stanford_config = StanfordBackgroundConfig(
    type="stanford_background",
    root="/path/to/stanford_background/",
    split="train",  # Only "train" available
    image_shape=[256, 256],
    max_images=100
)
```

**Parameters:**
- `root` (str | Path): Path to Stanford Background directory
- `split` (str): Dataset split (only "train" available)
- `image_shape` (list[int]): Desired image shape [height, width]
- `max_images` (int | None): Maximum number of images to process

## Complete Configuration Examples

### Python Configuration

```python
from segmentation_robustness_framework.config import (
    FGSMConfig, PGDConfig, VOCConfig
)

# Create attack configurations
attacks = [
    FGSMConfig(type="fgsm", epsilon=[0.01, 0.05, 0.1]),
    PGDConfig(
        type="pgd",
        epsilon=[0.05, 0.1],
        alpha=0.01,
        steps=10,
        targeted=False
    ),
    PGDConfig(
        type="pgd",
        epsilon=[0.05, 0.1],
        alpha=0.01,
        steps=20,
        targeted=True,
        target_label=15
    )
]

# Create dataset configuration
dataset = VOCConfig(
    type="voc",
    root="/path/to/VOCdevkit/VOC2012/",
    split="val",
    image_shape=[512, 512],
    max_images=100
)

# Convert to dictionaries for framework use
attack_dicts = [attack.model_dump() for attack in attacks]
dataset_dict = dataset.model_dump()
```

### YAML Configuration

```yaml
model:
  type: "torchvision"
  name: "deeplabv3_resnet50"
  num_classes: 21
  weights: "DEFAULT"
  device: "cuda"

attacks:
  - type: "fgsm"
    epsilon: [0.01, 0.05, 0.1, 0.2]

  - type: "pgd"
    epsilon: [0.05, 0.1, 0.2]
    alpha: [0.01, 0.02]
    steps: 10
    targeted: false

  - type: "pgd"
    epsilon: [0.05, 0.1]
    alpha: 0.01
    steps: 20
    targeted: true
    target_label: 15

  - type: "rfgsm"
    epsilon: [0.05, 0.1]
    alpha: 0.02
    steps: 5
    targeted: false

dataset:
  type: "voc"
  root: "/path/to/VOCdevkit/VOC2012/"
  split: "val"
  image_shape: [512, 512]
  max_images: 100
  download: true
```

## Validation and Error Handling

The configuration classes provide automatic validation:

### Attack Validation

```python
# This will raise an error - FGSM doesn't support targeted attacks
try:
    invalid_fgsm = FGSMConfig(
        type="fgsm",
        epsilon=0.1,
        targeted=True  # ValueError: FGSM does not support targeted attacks
    )
except ValueError as e:
    print(f"Validation error: {e}")

# This will raise an error - PGD targeted attack without target_label
try:
    invalid_pgd = PGDConfig(
        type="pgd",
        epsilon=0.1,
        alpha=0.01,
        steps=10,
        targeted=True,
        # Missing target_label - ValueError: target_label is required for targeted PGD attacks
    )
except ValueError as e:
    print(f"Validation error: {e}")
```

### Dataset Validation

```python
# This will raise an error - invalid image shape
try:
    invalid_dataset = VOCConfig(
        type="voc",
        root="/path/to/voc/",
        split="val",
        image_shape=[512],  # Missing width - ValueError: image_shape must be a list of 2 positive integers
        max_images=100
    )
except ValueError as e:
    print(f"Validation error: {e}")
```

## Integration with Framework

### Using with Loaders

```python
from segmentation_robustness_framework.loaders.models import UniversalModelLoader
from segmentation_robustness_framework.loaders import AttackLoader, DatasetLoader

# Create configurations
attack_config = PGDConfig(
    type="pgd",
    epsilon=[0.05, 0.1],
    alpha=0.01,
    steps=10,
    targeted=False
)

dataset_config = VOCConfig(
    type="voc",
    root="/path/to/VOCdevkit/VOC2012/",
    split="val",
    image_shape=[512, 512],
    max_images=100
)

# Convert to dictionaries and use with loaders
attack_dict = attack_config.model_dump()
dataset_dict = dataset_config.model_dump()

# Use with framework components
loader = UniversalModelLoader()
# ... use attack_dict and dataset_dict with loaders
```

### Using with RobustEngine

```python
from segmentation_robustness_framework.engine import RobustEngine

# Create YAML configuration file
config_path = "my_config.yaml"
# ... write configuration to file

# Use with RobustEngine
engine = RobustEngine(config_path)
engine.run(save=True, show=True, metrics=["mean_iou", "recall_macro"])
```

## Best Practices

1. **Use Type Hints**: Always specify the correct types for parameters
2. **Validate Early**: Test configurations before running experiments
3. **Document Parameters**: Use descriptive parameter names and values
4. **Parameter Sweeping**: Use lists for parameters you want to sweep
5. **Error Handling**: Always handle validation errors gracefully

## Troubleshooting

### Common Issues

1. **Validation Errors**: Check parameter types and constraints
2. **Missing Parameters**: Ensure all required parameters are provided
3. **Invalid Paths**: Verify dataset paths exist and are accessible
4. **Memory Issues**: Use `max_images` to limit dataset size for large experiments

### Debugging Tips

1. Use `.model_dump()` to inspect configuration dictionaries
2. Check validation errors for specific parameter constraints
3. Test configurations with small datasets first
4. Verify all paths and dependencies are correct 