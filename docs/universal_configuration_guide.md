# Universal Configuration Guide

This guide explains how to use the universal configuration system that can handle **any** model, attack, or dataset from **any** source, making the framework truly extensible and flexible.

## Table of Contents

- [Overview](#overview)
- [Universal Configuration Structure](#universal-configuration-structure)
- [Supported Sources](#supported-sources)
- [Configuration Examples](#configuration-examples)
- [Dynamic Loading](#dynamic-loading)
- [Best Practices](#best-practices)
- [Troubleshooting](#troubleshooting)

## Overview

The universal configuration system provides **complete flexibility** to use any component from any source:

- ✅ **Built-in components** (torchvision, framework attacks, etc.)
- ✅ **Custom components** (your own models, attacks, datasets)
- ✅ **Third-party components** (advertorch, foolbox, transformers, etc.)
- ✅ **Mixed components** (combine any sources in one config)
- ✅ **Dynamic loading** (load components at runtime)
- ✅ **Type safety** (Pydantic validation)
- ✅ **Error handling** (clear error messages)

## Universal Configuration Structure

### Core Configuration Classes

```python
from segmentation_robustness_framework.config import (
    UniversalConfig,
    UniversalModelConfig,
    UniversalAttackConfig,
    UniversalDatasetConfig,
)
```

### UniversalModelConfig

```python
UniversalModelConfig(
    type="torchvision",                    # Model type identifier
    source="torchvision.models.segmentation",  # Module source
    identifier="deeplabv3_resnet50",      # Model identifier
    parameters={"num_classes": 21},        # Model parameters
    device="cuda"                         # Device
)
```

### UniversalAttackConfig

```python
UniversalAttackConfig(
    type="fgsm",                          # Attack type identifier
    source="segmentation_robustness_framework.attacks",  # Module source
    identifier="FGSM",                    # Attack identifier
    parameters={},                        # Attack-specific parameters
    epsilon=[0.01, 0.05, 0.1],          # Common parameters
    alpha=0.01,                          # Common parameters
    steps=10,                            # Common parameters
    targeted=False                        # Common parameters
)
```

### UniversalDatasetConfig

```python
UniversalDatasetConfig(
    type="voc",                           # Dataset type identifier
    source="segmentation_robustness_framework.datasets",  # Module source
    identifier="VOCSegmentation",         # Dataset identifier
    parameters={},                        # Dataset-specific parameters
    root="/path/to/dataset",             # Common parameters
    split="val",                         # Common parameters
    image_shape=[512, 512],              # Common parameters
    max_images=100                       # Common parameters
)
```

## Supported Sources

### Built-in Sources

```python
# Framework components
source="segmentation_robustness_framework.attacks"
source="segmentation_robustness_framework.datasets"
source="segmentation_robustness_framework.loaders.models"

# PyTorch components
source="torchvision.models.segmentation"
source="torchvision.datasets"
source="torch.nn"
```

### Third-party Sources

```python
# HuggingFace Transformers
source="transformers"
source="transformers.models.segformer"

# Adversarial attack libraries
source="advertorch.attacks"
source="foolbox.attacks"
source="art.attacks"

# Other libraries
source="timm"
source="segmentation_models_pytorch"
```

### Custom Sources

```python
# Your own modules
source="my_custom_models"
source="my_custom_attacks"
source="my_custom_datasets"

# Nested modules
source="my_deep_module.models.segmentation"
source="my_attack_module.attacks.adversarial"
```

## Configuration Examples

### 1. Built-in Components

```python
from segmentation_robustness_framework.config import UniversalConfig

config = UniversalConfig(
    model=UniversalModelConfig(
        type="torchvision",
        source="torchvision.models.segmentation",
        identifier="deeplabv3_resnet50",
        parameters={"num_classes": 21, "weights": "DEFAULT"},
        device="cuda"
    ),
    attacks=[
        UniversalAttackConfig(
            type="fgsm",
            source="segmentation_robustness_framework.attacks",
            identifier="FGSM",
            parameters={},
            epsilon=[0.01, 0.05, 0.1]
        ),
        UniversalAttackConfig(
            type="pgd",
            source="segmentation_robustness_framework.attacks",
            identifier="PGD",
            parameters={},
            epsilon=[0.05, 0.1],
            alpha=0.01,
            steps=10,
            targeted=False
        )
    ],
    dataset=UniversalDatasetConfig(
        type="voc",
        source="segmentation_robustness_framework.datasets",
        identifier="VOCSegmentation",
        parameters={},
        root="/path/to/VOCdevkit/VOC2012/",
        split="val",
        image_shape=[512, 512],
        max_images=100
    ),
    output_dir="./runs/builtin_example",
    metrics=["mean_iou", "pixel_accuracy"],
    save_results=True,
    show_results=False
)
```

### 2. Custom Components

```python
config = UniversalConfig(
    model=UniversalModelConfig(
        type="custom",
        source="my_custom_models",
        identifier="MyCustomSegmentationModel",
        parameters={
            "num_classes": 21,
            "backbone": "resnet50",
            "pretrained": True
        },
        device="cuda"
    ),
    attacks=[
        UniversalAttackConfig(
            type="custom_attack",
            source="my_custom_attacks",
            identifier="MyCustomAttack",
            parameters={
                "custom_param1": 0.1,
                "custom_param2": "value"
            },
            epsilon=[0.01, 0.05]
        )
    ],
    dataset=UniversalDatasetConfig(
        type="custom_dataset",
        source="my_custom_datasets",
        identifier="MyCustomDataset",
        parameters={
            "custom_transform": True,
            "augmentation": "strong"
        },
        root="/path/to/my/dataset",
        split="val",
        image_shape=[512, 512],
        max_images=100
    ),
    output_dir="./runs/custom_example",
    metrics=["mean_iou"],
    save_results=True,
    show_results=True
)
```

### 3. Third-party Components

```python
config = UniversalConfig(
    model=UniversalModelConfig(
        type="huggingface",
        source="transformers",
        identifier={
            "path": ["AutoModelForSemanticSegmentation", "from_pretrained"]
        },
        parameters={
            "pretrained_model_name_or_path": "nvidia/segformer-b2-finetuned-ade-512-512",
            "num_labels": 150
        },
        device="cuda"
    ),
    attacks=[
        UniversalAttackConfig(
            type="advertorch_attack",
            source="advertorch.attacks",
            identifier="PGD",
            parameters={
                "eps": 0.3,
                "alpha": 0.01,
                "steps": 40
            }
        ),
        UniversalAttackConfig(
            type="foolbox_attack",
            source="foolbox.attacks",
            identifier="FGSM",
            parameters={
                "epsilon": 0.1
            }
        )
    ],
    dataset=UniversalDatasetConfig(
        type="torchvision_dataset",
        source="torchvision.datasets",
        identifier="VOCSegmentation",
        parameters={
            "year": "2012",
            "image_set": "val",
            "download": True
        },
        root="/path/to/datasets",
        image_shape=[512, 512],
        max_images=50
    ),
    output_dir="./runs/third_party_example",
    metrics=["mean_iou", "dice_macro"],
    save_results=True,
    show_results=False
)
```

### 4. Mixed Components

```python
config = UniversalConfig(
    model=UniversalModelConfig(
        type="torchvision",
        source="torchvision.models.segmentation",
        identifier="deeplabv3_resnet50",
        parameters={"num_classes": 21, "weights": "DEFAULT"},
        device="cuda"
    ),
    attacks=[
        # Built-in attack
        UniversalAttackConfig(
            type="fgsm",
            source="segmentation_robustness_framework.attacks",
            identifier="FGSM",
            parameters={},
            epsilon=[0.01, 0.05, 0.1]
        ),
        # Custom attack
        UniversalAttackConfig(
            type="custom_attack",
            source="my_custom_attacks",
            identifier="MyCustomAttack",
            parameters={"custom_param": 0.1},
            epsilon=[0.05, 0.1]
        ),
        # Third-party attack
        UniversalAttackConfig(
            type="advertorch_pgd",
            source="advertorch.attacks",
            identifier="PGD",
            parameters={
                "eps": 0.3,
                "alpha": 0.01,
                "steps": 40
            }
        )
    ],
    dataset=UniversalDatasetConfig(
        type="custom_dataset",
        source="my_custom_datasets",
        identifier="MyCustomDataset",
        parameters={"custom_transform": True},
        root="/path/to/my/dataset",
        split="val",
        image_shape=[512, 512],
        max_images=100
    ),
    output_dir="./runs/mixed_example",
    metrics=["mean_iou", "pixel_accuracy", "dice_macro"],
    save_results=True,
    show_results=True
)
```

## Dynamic Loading

### Simple Identifiers

```python
# Direct class/function names
identifier="FGSM"
identifier="deeplabv3_resnet50"
identifier="VOCSegmentation"
```

### Complex Identifiers

```python
# Nested module paths
identifier={
    "path": ["models", "segmentation", "DeepLabV3"]
}

# Factory methods
identifier={
    "path": ["AutoModelForSemanticSegmentation", "from_pretrained"]
}

# Custom factory
identifier={
    "path": ["models", "segmentation"],
    "factory_method": "create_model"
}
```

### Parameter Combinations

The system automatically generates parameter combinations for parameter sweeping:

```python
# Single values
epsilon=0.1
alpha=0.01

# Multiple values (automatic combinations)
epsilon=[0.01, 0.05, 0.1]
alpha=[0.01, 0.02]

# Results in 6 combinations:
# (0.01, 0.01), (0.01, 0.02), (0.05, 0.01), (0.05, 0.02), (0.1, 0.01), (0.1, 0.02)
```

## YAML Configuration

### Complete YAML Example

```yaml
model:
  type: "torchvision"
  source: "torchvision.models.segmentation"
  identifier: "deeplabv3_resnet50"
  parameters:
    num_classes: 21
    weights: "DEFAULT"
  device: "cuda"

attacks:
  - type: "fgsm"
    source: "segmentation_robustness_framework.attacks"
    identifier: "FGSM"
    parameters: {}
    epsilon: [0.01, 0.05, 0.1, 0.2]

  - type: "pgd"
    source: "segmentation_robustness_framework.attacks"
    identifier: "PGD"
    parameters: {}
    epsilon: [0.05, 0.1, 0.2]
    alpha: [0.01, 0.02]
    steps: 10
    targeted: false

  - type: "custom_attack"
    source: "my_custom_attacks"
    identifier: "MyCustomAttack"
    parameters:
      custom_param1: 0.1
      custom_param2: "value"
    epsilon: [0.05, 0.1]

  - type: "advertorch_pgd"
    source: "advertorch.attacks"
    identifier: "PGD"
    parameters:
      eps: 0.3
      alpha: 0.01
      steps: 40

dataset:
  type: "custom_dataset"
  source: "my_custom_datasets"
  identifier: "MyCustomDataset"
  parameters:
    custom_transform: true
    augmentation: "strong"
  root: "/path/to/my/dataset"
  split: "val"
  image_shape: [512, 512]
  max_images: 100

output_dir: "./runs/universal_example"
metrics: ["mean_iou", "pixel_accuracy", "dice_macro"]
save_results: true
show_results: false
```

## Using with Loaders

```python
from segmentation_robustness_framework.loaders.universal_loader import (
    UniversalModelLoader,
    UniversalAttackLoader,
    UniversalDatasetLoader,
)

# Load model
model_loader = UniversalModelLoader()
model = model_loader.load_model(config.model.model_dump())

# Load attacks
attack_loader = UniversalAttackLoader(model)
attacks = attack_loader.load_attacks([attack.model_dump() for attack in config.attacks])

# Load dataset
dataset_loader = UniversalDatasetLoader()
dataset = dataset_loader.load_dataset(config.dataset.model_dump())
```

## Best Practices

### 1. Module Organization

```python
# Good: Clear module structure
source="my_custom_models.segmentation"
source="my_custom_attacks.adversarial"
source="my_custom_datasets.semantic"

# Good: Descriptive identifiers
identifier="DeepLabV3WithResNet50"
identifier="CustomFGSMAttack"
identifier="MyVOCDataset"
```

### 2. Parameter Management

```python
# Good: Separate common and specific parameters
UniversalAttackConfig(
    type="custom_attack",
    source="my_custom_attacks",
    identifier="MyCustomAttack",
    parameters={
        "custom_param1": 0.1,
        "custom_param2": "value"
    },
    epsilon=[0.01, 0.05],  # Common parameter
    steps=10                # Common parameter
)
```

### 3. Error Handling

```python
try:
    config = UniversalConfig(...)
    # Use configuration
except ValueError as e:
    print(f"Configuration validation error: {e}")
except ImportError as e:
    print(f"Module import error: {e}")
except AttributeError as e:
    print(f"Component not found: {e}")
```

### 4. Testing Configurations

```python
# Test with small datasets first
config.dataset.max_images = 5

# Test with single parameters
config.attacks[0].epsilon = 0.01  # Single value instead of list

# Test with minimal components
config.attacks = config.attacks[:1]  # Only one attack
```

## Troubleshooting

### Common Issues

1. **ImportError: Cannot import module**
   - Check if the module is installed
   - Verify the module path is correct
   - Ensure the module is in Python path

2. **AttributeError: Component not found**
   - Check if the identifier exists in the module
   - Verify the identifier spelling
   - Use `dir(module)` to see available components

3. **ValueError: Configuration validation error**
   - Check required fields are provided
   - Verify parameter types and constraints
   - Ensure at least one attack is specified

4. **RuntimeError: Model instantiation failed**
   - Check model parameters are correct
   - Verify model class signature
   - Test model instantiation separately

### Debugging Tips

1. **Test components individually**
```python
# Test model loading
model_loader = UniversalModelLoader()
model = model_loader.load_model(config.model.model_dump())

# Test attack loading
attack_loader = UniversalAttackLoader(model)
attacks = attack_loader.load_attacks([config.attacks[0].model_dump()])

# Test dataset loading
dataset_loader = UniversalDatasetLoader()
dataset = dataset_loader.load_dataset(config.dataset.model_dump())
```

2. **Use simple configurations first**
```python
# Start with built-in components
source="segmentation_robustness_framework.attacks"
identifier="FGSM"

# Then add custom components
source="my_custom_attacks"
identifier="MyCustomAttack"
```

3. **Check module availability**
```python
import importlib

# Test if module can be imported
try:
    module = importlib.import_module("my_custom_attacks")
    print("Module imported successfully")
    print(f"Available components: {dir(module)}")
except ImportError as e:
    print(f"Module import failed: {e}")
```

4. **Validate component instantiation**
```python
# Test component instantiation
try:
    attack_class = getattr(module, "MyCustomAttack")
    attack = attack_class(model=model, **parameters)
    print("Component instantiated successfully")
except Exception as e:
    print(f"Component instantiation failed: {e}")
```

## Advanced Features

### Complex Module Paths

```python
# Deep nested modules
identifier={
    "path": ["models", "segmentation", "backbones", "resnet", "DeepLabV3"]
}

# Factory methods with parameters
identifier={
    "path": ["AutoModelForSemanticSegmentation", "from_pretrained"],
    "factory_params": {
        "pretrained_model_name_or_path": "nvidia/segformer-b2-finetuned-ade-512-512"
    }
}
```

### Custom Factory Functions

```python
# Custom factory function
identifier={
    "path": ["models", "factory"],
    "factory_method": "create_model",
    "factory_params": {
        "backbone": "resnet50",
        "num_classes": 21
    }
}
```

### Dynamic Parameter Generation

```python
# Generate parameters dynamically
parameters={
    "num_classes": lambda: get_dataset_classes(),
    "backbone": lambda: select_best_backbone(),
    "pretrained": lambda: check_pretrained_availability()
}
```

The universal configuration system provides **unlimited flexibility** to use any component from any source, making your framework truly extensible and future-proof! 