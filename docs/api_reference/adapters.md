# Adapters API

This page documents the adapter components of the Segmentation Robustness Framework.

## Adapters

::: segmentation_robustness_framework.adapters.base_protocol
    options:
        show_signature_annotations: true

::: segmentation_robustness_framework.adapters.torchvision_adapter
    options:
        show_signature_annotations: true

::: segmentation_robustness_framework.adapters.smp_adapter
    options:
        show_signature_annotations: true

::: segmentation_robustness_framework.adapters.huggingface_adapter
    options:
        show_signature_annotations: true

::: segmentation_robustness_framework.adapters.custom_adapter
    options:
        show_signature_annotations: true

## Adapter Overview

Adapters provide a standardized interface for different model types, ensuring compatibility with the framework's evaluation pipeline. Each adapter implements the `SegmentationModelProtocol` interface.

### SegmentationModelProtocol

The base protocol that all adapters must implement:

```python
from typing import Protocol
import torch

class SegmentationModelProtocol(Protocol):
    """Standardized interface for segmentation models."""
    
    num_classes: int
    
    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw model outputs [B, C, H, W]"""
        ...
    
    def predictions(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted labels [B, H, W]"""
        ...
```

### Available Adapters

#### TorchvisionAdapter

Adapts torchvision segmentation models (DeepLab, FCN, LRASPP):

```python
from segmentation_robustness_framework.adapters import TorchvisionAdapter
import torchvision.models.segmentation as segmentation

# Create a torchvision model
model = segmentation.deeplabv3_resnet50(pretrained=True)

# Wrap with adapter
adapter = TorchvisionAdapter(model)

# Use in pipeline
logits = adapter.logits(x)  # [B, C, H, W]
predictions = adapter.predictions(x)  # [B, H, W]
```

#### SMPAdapter

Adapts segmentation_models_pytorch models (UNet, LinkNet, FPN, etc.):

```python
from segmentation_robustness_framework.adapters import SMPAdapter
import segmentation_models_pytorch as smp

# Create an SMP model
model = smp.Unet(
    encoder_name="resnet34",
    encoder_weights="imagenet",
    classes=21
)

# Wrap with adapter
adapter = SMPAdapter(model)

# Use in pipeline
logits = adapter.logits(x)  # [B, C, H, W]
predictions = adapter.predictions(x)  # [B, H, W]
```

#### HuggingFaceAdapter

Adapts HuggingFace transformer models:

```python
from segmentation_robustness_framework.adapters import HuggingFaceAdapter
from transformers import SegformerForSemanticSegmentation

# Create a HuggingFace model
model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512"
)

# Wrap with adapter
adapter = HuggingFaceAdapter(model)

# Use in pipeline
logits = adapter.logits(x)  # [B, C, H, W]
predictions = adapter.predictions(x)  # [B, H, W]
```

#### CustomAdapter

Template for creating custom adapters:

```python
from segmentation_robustness_framework.adapters import CustomAdapter
import torch.nn as nn

class MyCustomModel(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        # Your model architecture here
        pass
    
    def forward(self, x):
        # Your forward pass
        return logits

# Create custom adapter
class MyCustomAdapter(CustomAdapter):
    def __init__(self, model: MyCustomModel):
        super().__init__(model)
        self.num_classes = 21
    
    def logits(self, x):
        return self.model(x)
    
    def predictions(self, x):
        logits = self.logits(x)
        return torch.argmax(logits, dim=1)

# Use custom adapter
model = MyCustomModel()
adapter = MyCustomAdapter(model)
```

### Adapter Registration

Register custom adapters for automatic discovery:

```python
from segmentation_robustness_framework.adapters import register_adapter

@register_adapter("my_custom")
class MyCustomAdapter(CustomAdapter):
    def __init__(self, model):
        super().__init__(model)
        self.num_classes = 21
    
    def logits(self, x):
        return self.model(x)
    
    def predictions(self, x):
        logits = self.logits(x)
        return torch.argmax(logits, dim=1)

# Now you can use it with the registered name
model = UniversalModelLoader().load_model(
    model_type="my_custom",  # Uses the registered adapter
    model_config={"model_class": MyCustomModel}
)
```

### Adapter Usage in Pipeline

Adapters are automatically used by the model loaders:

```python
from segmentation_robustness_framework.loaders import UniversalModelLoader

# The loader automatically creates the appropriate adapter
model = UniversalModelLoader().load_model(
    model_type="torchvision",
    model_config={"name": "deeplabv3_resnet50"}
)

# The model is already wrapped with the correct adapter
logits = model.logits(x)
predictions = model.predictions(x)
```

### How Adapter Selection Works

1. **Automatic Selection**: The `UniversalModelLoader` automatically selects the appropriate adapter based on the `model_type`
2. **Registry Lookup**: It uses `get_adapter(model_type)` to find the registered adapter
3. **Default Mapping**: Built-in adapters are pre-registered with their model type names
4. **Custom Override**: You can pass `adapter_cls` parameter to override the default adapter
5. **Protocol Check**: If the model already implements `SegmentationModelProtocol`, no adapter is applied

```python
# The selection process:
model_type = "torchvision"
adapter_cls = get_adapter(model_type)  # Returns TorchvisionAdapter
model = adapter_cls(raw_model)  # Wraps the model
```

### Adapter Selection

Adapters are automatically selected based on the model type. The framework uses the following mapping:

- `torchvision` → `TorchvisionAdapter`
- `smp` → `SMPAdapter`
- `huggingface` → `HuggingFaceAdapter`
- `custom_*` → `CustomAdapter`

```python
# The universal loader automatically selects the correct adapter
model = UniversalModelLoader().load_model(
    model_type="torchvision",  # Will use TorchvisionAdapter
    model_config={"name": "deeplabv3_resnet50"}
)

# For custom models, you can override the adapter
from segmentation_robustness_framework.adapters import MyCustomAdapter

model = UniversalModelLoader().load_model(
    model_type="custom",
    model_config={"model_class": MyCustomModel},
    adapter_cls=MyCustomAdapter  # Override default adapter
)
```

### Error Handling

Adapters include comprehensive error handling:

```python
try:
    adapter = TorchvisionAdapter(model)
    logits = adapter.logits(x)
except Exception as e:
    print(f"Adapter error: {e}")
    # Handle error appropriately
```

### Performance Considerations

- **Memory Efficiency**: Adapters are lightweight wrappers
- **GPU Compatibility**: All adapters support GPU acceleration
- **Batch Processing**: Optimized for batch inference
- **Gradient Flow**: Preserves gradients for adversarial training 