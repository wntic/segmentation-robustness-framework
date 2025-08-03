# Datasets API

This page documents the dataset components of the Segmentation Robustness Framework.

## Dataset Classes

::: segmentation_robustness_framework.datasets.voc
    options:
        show_signature_annotations: true

::: segmentation_robustness_framework.datasets.ade20k
    options:
        show_signature_annotations: true

::: segmentation_robustness_framework.datasets.cityscapes
    options:
        show_signature_annotations: true

::: segmentation_robustness_framework.datasets.stanford_background
    options:
        show_signature_annotations: true

## Dataset Loaders

::: segmentation_robustness_framework.loaders.dataset_loader
    options:
        show_signature_annotations: true

## Dataset Overview

The framework provides support for popular semantic segmentation datasets with automatic preprocessing and data loading.

### Available Datasets

#### VOC (PASCAL VOC 2012)

The PASCAL Visual Object Classes dataset:

```python
from segmentation_robustness_framework.datasets import VOCSegmentation

# Load VOC dataset
dataset = VOCSegmentation(
    root="./data",
    split="val",
    transform=transform,
    target_transform=target_transform
)

print(f"Dataset size: {len(dataset)}")
print(f"Number of classes: {dataset.num_classes}")
```

**Features:**
- 20 object classes + background
- High-quality pixel-level annotations
- Standard benchmark dataset
- Automatic download support

#### ADE20K (MIT Scene Parsing)

The MIT ADE20K dataset for scene parsing:

```python
from segmentation_robustness_framework.datasets import ADE20K

# Load ADE20K dataset
dataset = ADE20K(
    root="./data",
    split="val",
    transform=transform,
    target_transform=target_transform
)

print(f"Dataset size: {len(dataset)}")
print(f"Number of classes: {dataset.num_classes}")
```

**Features:**
- 150 semantic categories
- Complex scene understanding
- High-resolution images
- Detailed annotations

#### Cityscapes

Urban scene understanding dataset:

```python
from segmentation_robustness_framework.datasets import Cityscapes

# Load Cityscapes dataset
dataset = Cityscapes(
    root="./data",
    split="val",
    mode="fine",
    target_type="semantic",
    transform=transform,
    target_transform=target_transform
)

print(f"Dataset size: {len(dataset)}")
print(f"Number of classes: {dataset.num_classes}")
```

**Features:**
- 19 semantic categories
- High-resolution urban images
- Fine and coarse annotations
- Multiple annotation types

#### Stanford Background

Natural scene parsing dataset:

```python
from segmentation_robustness_framework.datasets import StanfordBackground

# Load Stanford Background dataset
dataset = StanfordBackground(
    root="./data",
    transform=transform,
    target_transform=target_transform
)

print(f"Dataset size: {len(dataset)}")
print(f"Number of classes: {dataset.num_classes}")
```

**Features:**
- 8 semantic categories
- Natural outdoor scenes
- High-quality annotations
- Compact dataset for testing

### Dataset Configuration

Configure datasets in YAML configuration files. The framework automatically applies preprocessing based on the `image_shape` parameter:

```yaml
dataset:
  name: voc
  split: val
  root: ./data
  image_shape: [512, 512]  # Automatically applies resize, normalize, and mask conversion
```

### Dataset Loading

Use the `DatasetLoader` for automatic dataset loading with preprocessing:

```python
from segmentation_robustness_framework.loaders import DatasetLoader

# Load dataset with configuration
dataset_config = {
    "name": "voc",
    "split": "val",
    "root": "./data",
    "image_shape": [512, 512]  # Automatically applies preprocessing
}

dataset_loader = DatasetLoader(dataset_config)
dataset = dataset_loader.load_dataset()
```

### Automatic Preprocessing

The framework automatically applies preprocessing based on the `image_shape` parameter:

```python
from segmentation_robustness_framework.utils.image_preprocessing import get_preprocessing_fn

# Get preprocessing functions (automatically called by DatasetLoader)
image_preprocess, target_preprocess = get_preprocessing_fn(
    image_shape=[512, 512],
    dataset_name="voc"
)

# The preprocessing includes:
# - Image resize to specified shape
# - Image normalization (ImageNet stats)
# - Mask resize to match image
# - RGB to index conversion for masks
# - Stride alignment (ensures dimensions are divisible by 8)
```

### What Gets Applied Automatically

When you specify `image_shape` in the dataset configuration, the framework automatically applies:

#### Image Preprocessing

- **Resize**: Images are resized to the specified `[height, width]`
- **Normalization**: Images are normalized using ImageNet statistics (`mean=[0.485, 0.456, 0.406]`, `std=[0.229, 0.224, 0.225]`)
- **Tensor Conversion**: Images are converted to PyTorch tensors

#### Mask Preprocessing

- **Resize**: Masks are resized to match the image dimensions
- **RGB to Index**: RGB masks are converted to class indices using dataset-specific color palettes
- **Stride Alignment**: Dimensions are adjusted to be divisible by 8 for model compatibility

#### Dataset-Specific Features

- **Color Mapping**: Each dataset has its own color palette for mask conversion
- **Ignore Index**: Proper handling of ignored pixels (usually index 255)
- **Error Handling**: Warnings for unmapped colors in masks

### Custom Datasets

Create custom datasets by inheriting from `torch.utils.data.Dataset`:

```python
import torch
from torch.utils.data import Dataset
from PIL import Image
import os

class MyCustomDataset(Dataset):
    def __init__(self, root, split="train", transform=None, target_transform=None):
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.num_classes = 21
        
        # Load your data
        self.images = []  # List of image paths
        self.masks = []   # List of mask paths
        
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image and mask
        image = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx])
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        
        return image, mask

# Use custom dataset
dataset = MyCustomDataset("./data", split="train")
```

### Dataset Registration

Register custom datasets for automatic discovery:

```python
from segmentation_robustness_framework.datasets import register_dataset

@register_dataset("my_custom")
class MyCustomDataset(Dataset):
    def __init__(self, root, split="train", transform=None, target_transform=None):
        # Your dataset implementation
        pass
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Your data loading logic
        pass

# Now you can use it in configuration
# dataset:
#   name: my_custom
#   split: train
```

### Dataset Usage in Pipeline

Datasets are automatically used by the pipeline:

```python
from segmentation_robustness_framework.pipeline import SegmentationRobustnessPipeline

# Create pipeline with dataset
pipeline = SegmentationRobustnessPipeline(
    model=model,
    dataset=dataset,  # Your dataset here
    attacks=[FGSM(model, eps=0.1)],
    metrics=[metrics.mean_iou],
    batch_size=4,
    device="cuda"
)

results = pipeline.run()
```

### Performance Considerations

- **Memory Efficiency**: Lazy loading for large datasets
- **GPU Compatibility**: Automatic device placement
- **Batch Processing**: Optimized for batch inference
- **Data Augmentation**: Built-in augmentation support
