# Model Loaders

The Segmentation Robustness Framework provides a flexible system for loading segmentation models from various sources, including Torchvision, segmentation_models_pytorch (SMP), HuggingFace Transformers, and custom user models.  
All loaders implement a common interface and can be used interchangeably.

---

## Table of Contents

- [Overview](#overview)
- [Configuring Model Loaders](#configuring-model-loaders)
- [Torchvision Loader](#torchvision-loader)
- [HuggingFace Loader](#huggingface-loader)
- [Custom Model Loader](#custom-model-loader)
- [Universal Model Loader](#universal-model-loader)
- [Loading Weights](#loading-weights)
- [Advanced Scenarios](#advanced-scenarios)
- [Troubleshooting](#troubleshooting)
- [API Reference](#api-reference)

---

## Overview

Each loader expects a configuration dictionary describing the model to load.  
You can use the dedicated loader classes directly, or use the `UniversalModelLoader` for a unified interface.

---

## Configuring Model Loaders

Below are example configuration dictionaries for each loader type.  
You can define these configs in your Python code or load them from YAML files.

### Torchvision Example

```python
torchvision_config = {
    "name": "deeplabv3_resnet50",  # Model name (required)
    "num_classes": 21,             # Number of output classes
    "weights": "DEFAULT",         # Pretrained weights (optional)
    # "device": "cuda"             # Device (optional, default: "cpu")
}
```

### HuggingFace Example

```python
huggingface_config = {
    "model_name": "nvidia/segformer-b2-finetuned-ade-512-512",  # Model id or path (required)
    "num_labels": 150,                                          # Number of output classes
    "task": "semantic_segmentation",                            # Task type (optional)
    "return_processor": True,                                   # Return processor (optional)
    "config_overrides": {                                       # Override config attributes (optional)
        "ignore_mismatched_sizes": True
    },
    "processor_overrides": {                                    # Override processor attributes (optional)
        "do_resize": False
    }
    # "device": "cuda"                                          # Device (optional, default: "cpu")
}
```

### Custom Model Example

Suppose you have a custom model class:

```python
from my_module import MyCustomSegmentationModel

custom_config = {
    "model_class": MyCustomSegmentationModel,  # Class or factory function (required)
    "model_args": [3, 21],                     # Positional args for model init
    "model_kwargs": {},                        # Keyword args for model init
    # "device": "cuda"                         # Device (optional, default: "cpu")
}
```

---

## Torchvision Loader

```python
from segmentation_robustness_framework.loaders.models import TorchvisionModelLoader

torchvision_config = {
    "name": "deeplabv3_resnet50",
    "num_classes": 21,
    "weights": "DEFAULT"
}

loader = TorchvisionModelLoader()
model = loader.load_model(torchvision_config)
```

---

## HuggingFace Loader

```python
from segmentation_robustness_framework.loaders.models import HuggingFaceModelLoader

huggingface_config = {
    "model_name": "nvidia/segformer-b2-finetuned-ade-512-512",
    "num_labels": 150,
    "config_overrides": {"ignore_mismatched_sizes": True},
    "processor_overrides": {"do_resize": False}
}

loader = HuggingFaceModelLoader()
bundle = loader.load_model(huggingface_config)
model = bundle.model
processor = bundle.processor

# Prepare input and run inference
from PIL import Image
img = Image.open("image.jpg")
inputs = processor(img, return_tensors="pt")
outputs = model(**inputs)
```

---

## Custom Model Loader

```python
from segmentation_robustness_framework.loaders.models import CustomModelLoader
from my_module import MyCustomSegmentationModel

custom_config = {
    "model_class": MyCustomSegmentationModel,
    "model_args": [3, 21],
    "model_kwargs": {}
}

loader = CustomModelLoader()
model = loader.load_model(custom_config)
```

---

## Universal Model Loader

The `UniversalModelLoader` provides a unified interface for all supported model types.

```python
from segmentation_robustness_framework.loaders.models import UniversalModelLoader

# Example: Torchvision
torchvision_config = {
    "name": "deeplabv3_resnet50",
    "num_classes": 21,
    "weights": "DEFAULT"
}

loader = UniversalModelLoader()
model = loader.load_model(
    model_type="torchvision",
    model_config=torchvision_config
)

# Example: HuggingFace
huggingface_config = {
    "model_name": "nvidia/segformer-b2-finetuned-ade-512-512",
    "num_labels": 150,
    "config_overrides": {"ignore_mismatched_sizes": True}
}
bundle = loader.load_model(
    model_type="huggingface",
    model_config=huggingface_config
)
model = bundle.model
processor = bundle.processor

# Example: Custom
from my_module import MyCustomSegmentationModel
custom_config = {
    "model_class": MyCustomSegmentationModel,
    "model_args": [3, 21],
    "model_kwargs": {}
}
model = loader.load_model(
    model_type="custom",
    model_config=custom_config
)
```

---

## Loading Weights

You can load weights at the same time as the model:

```python
model = loader.load_model(
    model_type="torchvision",
    model_config=torchvision_config,
    weights_path="path/to/weights.pth",
    weight_type="full"  # or "encoder"
)
```

Or load weights separately:

```python
model = loader.load_model(torchvision_config)
model = loader.load_weights(model, "path/to/weights.pth", weight_type="full")
```

---

## Advanced Scenarios

### Using Device Placement

All loaders accept a `"device"` key in the config (default: `"cpu"`).  
Move the model to GPU if needed:

```python
torchvision_config = {
    "name": "deeplabv3_resnet50",
    "num_classes": 21,
    "device": "cuda"
}
model = loader.load_model(torchvision_config)
model.to("cuda")
```

---

## Troubleshooting

- **Missing dependencies:**  
  If a loader is not available, ensure the required package is installed (e.g., `torchvision`, `transformers`, `segmentation_models_pytorch`).
- **Custom model class as string:**  
  If loading from YAML, resolve the class string to an actual Python class before passing to the loader.
- **Shape mismatch:**  
  Ensure `num_classes` matches your dataset and weights.

---

## API Reference

All loaders implement:

- `load_model(model_config: dict) -> nn.Module or Bundle`
- `load_weights(model: nn.Module, weights_path: str, weight_type: str = "full") -> nn.Module`

See the source code for detailed docstrings and advanced options.

---

## See Also

- [Custom Datasets Guide](custom_datasets_guide.md)
- [Custom Components Guide](custom_components.md)
- [Configuration Guide](configuration_guide.md)

---

**This page provides a comprehensive guide to using model loaders in the Segmentation Robustness Framework. Expand with more examples as your framework grows!**