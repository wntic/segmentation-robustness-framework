# Custom Datasets with Automatic Mask Resizing

This guide explains how to create custom datasets that work seamlessly with the automatic mask resizing feature in the segmentation robustness framework.

## Overview

The framework automatically detects model output sizes and resizes masks accordingly. For custom datasets, you need to provide a dataset name for proper color mapping (if your masks use RGB colors).

## Creating Custom Datasets

### Basic Requirements

Your custom dataset should:
1. Inherit from `torch.utils.data.Dataset`
2. Return `(image, mask)` pairs where:
   - `image` is a tensor of shape `[C, H, W]`
   - `mask` is a tensor of shape `[H, W]` with class indices
3. Set a `name` attribute for color mapping (if needed)

### Example: Medical Dataset

```python
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from pathlib import Path

class CustomMedicalDataset(Dataset):
    """Custom medical dataset with automatic mask resizing support."""
    
    def __init__(self, root: str, split: str = "train", transform=None, target_transform=None):
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        # 🎯 IMPORTANT: Set name for automatic color mapping
        self.name = "medical_dataset"
        
        # Dataset-specific attributes
        self.num_classes = 4  # Background, tumor, healthy_tissue, blood_vessel
        self.images_dir = self.root / "images" / split
        self.masks_dir = self.root / "masks" / split
        
        # Get list of image files
        self.images = [f for f in os.listdir(self.images_dir) if f.endswith('.png')]
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image and mask
        img_path = self.images_dir / self.images[idx]
        mask_path = self.masks_dir / self.images[idx].replace('.png', '_mask.png')
        
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')  # Grayscale mask
        
        if self.transform:
            image = self.transform(image)
        
        if self.target_transform:
            mask = self.target_transform(mask)
        
        return image, mask
```

### Alternative Naming Conventions

The framework supports multiple ways to specify the dataset name:

```python
class MyDataset(Dataset):
    def __init__(self, ...):
        # Option 1: Public attribute
        self.name = "my_dataset"
        
        # Option 2: Alternative public attribute
        self.dataset_name = "my_dataset"
        
        # Option 3: Private attribute
        self._name = "my_dataset"
```

## Using Custom Datasets

### Basic Usage

```python
from segmentation_robustness_framework.engine.pipeline import SegmentationRobustnessPipeline
from segmentation_robustness_framework.utils.image_preprocessing import get_preprocessing_fn

# Create preprocessing functions
preprocess, target_preprocess = get_preprocessing_fn([512, 512])

# Create custom dataset
custom_dataset = CustomMedicalDataset(
    root="./data/medical",
    split="val",
    transform=preprocess,
    target_transform=target_preprocess,
)

# Create pipeline with automatic mask resizing
pipeline = SegmentationRobustnessPipeline(
    model=model,
    dataset=custom_dataset,
    attacks=attacks,
    metrics=metrics,
    auto_resize_masks=True,  # 🎯 Enable automatic resizing
)

# Run evaluation
pipeline.run()
```

### What Happens Automatically

1. **Model Output Detection**: The framework runs a dummy forward pass to detect the model's output spatial size
2. **Dataset Name Detection**: The framework tries multiple strategies to detect your dataset name:
   - Checks for `name`, `dataset_name`, or `_name` attributes
   - Falls back to class name inference for built-in datasets
   - Checks the dataset registry
   - Examines the root path
3. **Mask Resizing**: Masks are automatically resized to match the model's output size
4. **Color Mapping**: If your dataset name is detected, appropriate color mapping is applied

## Registering Color Palettes

If your custom dataset uses RGB color masks, you can register a color palette:

```python
from segmentation_robustness_framework.utils.image_preprocessing import register_dataset_colors

# Define color palette for your dataset
MEDICAL_COLORS = [
    (0, 0, 0),      # Background (black)
    (255, 0, 0),    # Tumor (red)
    (0, 255, 0),    # Healthy tissue (green)
    (0, 0, 255),    # Blood vessel (blue)
]

# Register the color palette
register_dataset_colors("medical_dataset", MEDICAL_COLORS)
```

## Dataset Name Detection Strategies

The framework uses these strategies (in order) to detect your dataset name:

1. **Public Attributes**: `dataset.name`, `dataset.dataset_name`
2. **Private Attributes**: `dataset._name`
3. **Class Name Inference**: For built-in datasets only
4. **Registry Lookup**: If your dataset is registered
5. **Path Analysis**: Examines `dataset.root` for dataset indicators
6. **Fallback**: Uses generic mask processing if no name is detected

## Best Practices

### 1. Always Set a Name Attribute

```python
class MyDataset(Dataset):
    def __init__(self, ...):
        self.name = "my_dataset"  # Always set this
```

### 2. Use Descriptive Names

```python
# Good
self.name = "medical_brain_tumor_dataset"

# Avoid
self.name = "dataset"
```

### 3. Register Color Palettes Early

```python
# Do this before creating your dataset
register_dataset_colors("medical_brain_tumor_dataset", MEDICAL_COLORS)
```

### 4. Handle Missing Data Gracefully

```python
def __getitem__(self, idx):
    try:
        # Load your data
        image = Image.open(img_path).convert('RGB')
        mask = Image.open(mask_path).convert('L')
    except Exception as e:
        logger.warning(f"Failed to load sample {idx}: {e}")
        # Return a default sample or skip
        return self._get_default_sample()
```

## Troubleshooting

### Dataset Name Not Detected

If your dataset name is not being detected:

```python
# Check what the framework detects
pipeline = SegmentationRobustnessPipeline(...)
detected_name = pipeline._detect_dataset_name()
print(f"Detected dataset name: {detected_name}")
```

### Color Mapping Issues

If colors are not being mapped correctly:

1. Ensure your dataset name matches the registered color palette
2. Check that your RGB values are in the correct format `(R, G, B)`
3. Verify the color palette has the correct number of classes

### Shape Mismatch Errors

If you still get shape mismatch errors:

1. Ensure `auto_resize_masks=True` is set
2. Check that your model follows the adapter protocol
3. Verify that your dataset returns tensors in the correct format

## Example: Complete Custom Dataset

See `example_custom_dataset.py` for a complete working example of custom datasets with automatic mask resizing. 