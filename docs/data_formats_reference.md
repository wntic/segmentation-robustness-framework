# Data Formats Reference

Quick reference for data formats used throughout the segmentation robustness framework.

## Dataset Data Formats

### Input Images
| Property | Specification |
|----------|---------------|
| **Format** | PIL Image |
| **Mode** | RGB |
| **Size** | Any (will be resized) |
| **Values** | 0-255 (uint8) |

### Input Masks
| Property | Specification |
|----------|---------------|
| **Format** | PIL Image or numpy array |
| **Shape** | `[H, W]` or `[C, H, W]` |
| **Values** | Class indices (0 to num_classes-1) |
| **Data Type** | uint8 or int64 |

### Processed Images (After Transform)
| Property | Specification |
|----------|---------------|
| **Format** | torch.Tensor |
| **Shape** | `[C, H, W]` |
| **Values** | Normalized to [0, 1] |
| **Data Type** | torch.float32 |

### Processed Masks (After Transform)
| Property | Specification |
|----------|---------------|
| **Format** | torch.Tensor |
| **Shape** | `[H, W]` |
| **Values** | Class indices (0 to num_classes-1) |
| **Data Type** | torch.long |
| **Special** | -1 for ignore_index |

## Attack Data Formats

### Input Images
| Property | Specification |
|----------|---------------|
| **Format** | torch.Tensor |
| **Shape** | `[B, C, H, W]` |
| **Values** | Normalized to [0, 1] |
| **Device** | Same as model |

### Input Labels
| Property | Specification |
|----------|---------------|
| **Format** | torch.Tensor |
| **Shape** | `[B, H, W]` |
| **Values** | Class indices (0 to num_classes-1) |
| **Data Type** | torch.long |
| **Special** | -1 for ignore_index |

### Model Outputs
| Property | Specification |
|----------|---------------|
| **Format** | torch.Tensor |
| **Shape** | `[B, num_classes, H, W]` |
| **Values** | Raw logits |
| **Data Type** | torch.float32 |

### Adversarial Images
| Property | Specification |
|----------|---------------|
| **Format** | torch.Tensor |
| **Shape** | `[B, C, H, W]` |
| **Values** | Clamped to [0, 1] |
| **Device** | Same as input |

## Color-Coded Masks

### RGB Mask Format
```python
# Example color palette
COLORS = [
    (0, 0, 0),      # Class 0: background
    (255, 0, 0),    # Class 1: object1
    (0, 255, 0),    # Class 2: object2
    (0, 0, 255),    # Class 3: object3
]

# Register with framework
register_dataset_colors("my_dataset", COLORS)
```

### Conversion Process
```
RGB Mask [H, W, 3] → Color Matching → Index Mask [H, W]
```

## Tensor Shape Transformations

### Dataset Pipeline
```
Raw Image [H, W, 3] → PIL Image → Transform → Tensor [C, H, W]
Raw Mask [H, W] → PIL Image → Transform → Tensor [H, W]
```

### Attack Pipeline
```
Images [B, C, H, W] → Model → Logits [B, num_classes, H, W]
Labels [B, H, W] → Reshape → Flat Labels [B*H*W]
Logits [B, num_classes, H, W] → Reshape → Flat Logits [B*H*W, num_classes]
```

### Loss Computation
```python
# Reshape for CrossEntropyLoss
B, C, H, W = outputs.shape
outputs_flat = outputs.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
labels_flat = labels.reshape(-1)  # [B*H*W]

# Filter valid pixels
valid_mask = labels >= 0
valid_outputs = outputs_flat[valid_mask.reshape(-1)]
valid_labels = labels_flat[valid_mask.reshape(-1)]
```

## Common Shape Patterns

### Single Sample
```python
image.shape = [3, 256, 256]      # [C, H, W]
mask.shape = [256, 256]          # [H, W]
```

### Batch Processing
```python
images.shape = [4, 3, 256, 256]  # [B, C, H, W]
labels.shape = [4, 256, 256]     # [B, H, W]
outputs.shape = [4, 21, 256, 256] # [B, num_classes, H, W]
```

### Loss Computation
```python
logits.shape = [1024, 21]        # [B*H*W, num_classes]
targets.shape = [1024]           # [B*H*W]
```

## Data Type Requirements

| Component | Input Type | Output Type |
|-----------|------------|-------------|
| Images | PIL.Image | torch.float32 |
| Masks | PIL.Image/numpy | torch.long |
| Model | torch.float32 | torch.float32 |
| Loss | torch.long | torch.float32 |

## Device Management

### Automatic Device Handling
```python
# Model automatically moves to device
model = model.to(device)

# Images and labels moved to same device
images = images.to(device)
labels = labels.to(device)

# Adversarial images stay on same device
adv_images = attack(images, labels)  # Same device as inputs
```

### Device Consistency Check
```python
# Ensure all tensors are on same device
assert images.device == labels.device == model.device
```

## Validation Functions

### Shape Validation
```python
def validate_shapes(images, labels, model_outputs):
    assert images.shape[0] == labels.shape[0]  # Same batch size
    assert images.shape[2:] == labels.shape[1:]  # Same spatial dimensions
    assert model_outputs.shape[0] == images.shape[0]  # Same batch size
    assert model_outputs.shape[2:] == images.shape[2:]  # Same spatial dimensions
```

### Value Range Validation
```python
def validate_values(images, labels, num_classes):
    assert torch.all(images >= 0) and torch.all(images <= 1)  # Normalized images
    assert torch.all(labels >= -1) and torch.all(labels < num_classes)  # Valid labels
```

### Device Validation
```python
def validate_device(images, labels, model):
    device = images.device
    assert labels.device == device
    assert next(model.parameters()).device == device
``` 