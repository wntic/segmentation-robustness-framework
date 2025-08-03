# Model Loaders API

This page documents the model loading components of the Segmentation Robustness Framework.

## Model Loaders

::: segmentation_robustness_framework.loaders.models.universal_loader
    options:
        show_signature_annotations: true

::: segmentation_robustness_framework.loaders.models.torchvision_loader
    options:
        show_signature_annotations: true

::: segmentation_robustness_framework.loaders.models.smp_loader
    options:
        show_signature_annotations: true

::: segmentation_robustness_framework.loaders.models.huggingface_loader
    options:
        show_signature_annotations: true

::: segmentation_robustness_framework.loaders.models.custom_loader
    options:
        show_signature_annotations: true

## Model Loading Overview

The framework provides specialized loaders for different model types, each designed to handle the specific requirements and output formats of different model architectures.

### Universal Model Loader

The `UniversalModelLoader` is the main entry point for model loading. It automatically selects the appropriate specialized loader based on the model type.

```python
from segmentation_robustness_framework.loaders import UniversalModelLoader

# Load a torchvision model
model = UniversalModelLoader().load_model(
    model_type="torchvision",
    model_config={"name": "deeplabv3_resnet50", "num_classes": 21}
)

# Load an SMP model
model = UniversalModelLoader().load_model(
    model_type="smp",
    model_config={"architecture": "unet", "encoder_name": "resnet34", "classes": 21}
)

# Load a HuggingFace model
model = UniversalModelLoader().load_model(
    model_type="huggingface",
    model_config={"model_name": "nvidia/segformer-b0-finetuned-ade-512-512"}
)
```

### Supported Model Types

#### Torchvision Models

```python
# Available models
torchvision_models = [
    "deeplabv3_resnet50",
    "deeplabv3_resnet101",
    "deeplabv3_mobilenetv3_large",
    "fcn_resnet50",
    "fcn_resnet101",
    "lraspp_mobilenet_v3_large"
]

# Example usage
model = UniversalModelLoader().load_model(
    model_type="torchvision",
    model_config={"name": "deeplabv3_resnet50", "num_classes": 21}
)

# With custom weights
model = UniversalModelLoader().load_model(
    model_type="torchvision",
    model_config={
        "name": "deeplabv3_resnet50", 
        "num_classes": 21,
        "weights": "COCO_WITH_VOC_LABELS_V1"
    }
)
```

#### SMP Models

```python
# Available architectures
smp_architectures = [
    "unet", "unetplusplus", "manet", "linknet",
    "fpn", "pspnet", "pan", "deeplabv3", "deeplabv3plus"
]

# Available encoders
smp_encoders = [
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
    "resnext50_32x4d", "resnext101_32x8d",
    "timm-efficientnet-b0", "timm-efficientnet-b1", "timm-efficientnet-b2"
]

# Example usage
model = UniversalModelLoader().load_model(
    model_type="smp",
    model_config={
        "architecture": "unet",
        "encoder_name": "resnet34",
        "encoder_weights": "imagenet",
        "classes": 21
    }
)
```

#### HuggingFace Models

```python
# Example usage
model = UniversalModelLoader().load_model(
    model_type="huggingface",
    model_config={
        "model_name": "nvidia/segformer-b0-finetuned-ade-512-512",
        "trust_remote_code": True
    }
)

# With additional configuration
model = UniversalModelLoader().load_model(
    model_type="huggingface",
    model_config={
        "model_name": "nvidia/segformer-b0-finetuned-ade-512-512",
        "num_labels": 150,
        "config_overrides": {"ignore_mismatched_sizes": True},
        "processor_overrides": {"do_resize": False}
    }
)
```

### Custom Models

For custom models, you can use the `CustomModelLoader`:

```python
from segmentation_robustness_framework.loaders import CustomModelLoader

class MyCustomModel(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        # Your model architecture here
        pass
    
    def forward(self, x):
        # Your forward pass
        return logits

# Load custom model
model = CustomModelLoader().load_model(
    model_type="custom",
    model_config={"model_class": MyCustomModel, "num_classes": 21}
)
```

### Model Configuration

Each model type accepts different configuration parameters:

#### Torchvision Configuration

```yaml
model:
  type: torchvision
  config:
    name: deeplabv3_resnet50
    num_classes: 21
    weights: COCO_WITH_VOC_LABELS_V1  # Optional: specify weights
```

#### SMP Configuration

```yaml
model:
  type: smp
  architecture: unet
  encoder_name: resnet34
  encoder_weights: imagenet
  classes: 21
  activation: None  # Optional
```

#### HuggingFace Configuration

```yaml
model:
  type: huggingface
  config:
    model_name: nvidia/segformer-b0-finetuned-ade-512-512
    trust_remote_code: true
    num_labels: 150
```

### Error Handling

The model loaders include comprehensive error handling:

```python
try:
    model = UniversalModelLoader().load_model(
        model_type="torchvision",
        model_config={"name": "nonexistent_model"}
    )
except ValueError as e:
    print(f"Model loading failed: {e}")
    # Handle error appropriately
```
