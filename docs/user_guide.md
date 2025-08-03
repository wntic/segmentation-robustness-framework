# User Guide

Learn how to use the Segmentation Robustness Framework effectively for evaluating your models against adversarial attacks.

## 🎯 Overview

This guide teaches you how to:
- **Configure and run evaluations** using different approaches
- **Choose the right components** for your use case
- **Interpret results** and understand what they mean
- **Customize the framework** for your specific needs
- **Troubleshoot common issues** and optimize performance

## 📋 Table of Contents

1. [Basic Usage](#basic-usage)
2. [Configuration-Based Approach](#configuration-based-approach)
3. [Component Selection](#component-selection)
4. [Data Preprocessing](#data-preprocessing)
5. [Understanding Results](#understanding-results)
6. [Advanced Usage](#advanced-usage)
7. [Performance Optimization](#performance-optimization)
8. [Troubleshooting](#troubleshooting)

---

## 🚀 Basic Usage

### **Method 1: Direct Pipeline Usage**

The most straightforward approach for simple evaluations:

```python
import torch
from segmentation_robustness_framework import (
    SegmentationRobustnessPipeline,
    PipelineConfig,
    MetricsCollection
)
from segmentation_robustness_framework.loaders import (
    UniversalModelLoader,
    DatasetLoader,
    AttackLoader
)

# Step 1: Load your model
model_loader = UniversalModelLoader()
model = model_loader.load_model(
    model_type="torchvision",
    model_config={
        "name": "deeplabv3_resnet50",
        "num_classes": 21,
        "weights": "default"
    }
)

# Step 2: Load your dataset
dataset_loader = DatasetLoader({
    "name": "voc",
    "root": None,  # Will use cache directory
    "image_shape": [256, 256],  # Triggers automatic preprocessing
    "split": "val",
    "download": True
})
dataset = dataset_loader.load_dataset()

# The framework automatically applies preprocessing:
# - Resizes images to (256, 256) with stride alignment
# - Normalizes images using ImageNet statistics
# - Converts RGB masks to class indices (21 classes for VOC)
# - Handles ignore_index (255 → -1)

# Step 3: Set device and move model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Step 4: Create attacks (model must be on device first)
attack_loader = AttackLoader(model, [
    {"name": "fgsm", "eps": 2/255},
    {"name": "pgd", "eps": 4/255, "alpha": 1/255, "iters": 10}
])
attacks_nested = attack_loader.load_attacks()
# Flatten the nested list of attacks
attacks = [attack for attack_list in attacks_nested for attack in attack_list]

# Step 5: Set up metrics
metrics_collection = MetricsCollection(
    num_classes=21,  # Number of classes in VOC dataset
    ignore_index=255
)

# Create list of metric functions to evaluate
metrics = [
    metrics_collection.mean_iou,  # Uses default macro averaging
    metrics_collection.pixel_accuracy,
    metrics_collection.dice_score,  # Uses default macro averaging
    metrics_collection.precision,   # Uses default macro averaging
    metrics_collection.recall,      # Uses default macro averaging
]

# Alternative: Use specific averaging strategies
# metrics = [
#     lambda targets, preds: metrics_collection.mean_iou(targets, preds, average="macro"),
#     lambda targets, preds: metrics_collection.mean_iou(targets, preds, average="micro"),
#     lambda targets, preds: metrics_collection.dice_score(targets, preds, average="macro"),
#     lambda targets, preds: metrics_collection.dice_score(targets, preds, average="micro"),
#     metrics_collection.pixel_accuracy,
# ]

# Step 6: Create and run pipeline
pipeline = SegmentationRobustnessPipeline(
    model=model,
    dataset=dataset,
    attacks=attacks,
    metrics=metrics,
    batch_size=8,
    device=device,  # Use the same device as model
    output_dir="./results"
)

# Step 7: Run evaluation
results = pipeline.run(save=True)
print(f"Clean IoU: {results['clean']['mean_iou']:.3f}")

# Step 8: Verify preprocessing worked correctly
print("\n🔍 Preprocessing Verification:")
image, mask = dataset[0]  # Get first sample
print(f"Image shape: {image.shape}")  # Should be [3, 256, 256]
print(f"Image dtype: {image.dtype}")  # Should be torch.float32
print(f"Image range: [{image.min():.3f}, {image.max():.3f}]")  # Should be normalized
print(f"Mask shape: {mask.shape}")    # Should be [256, 256]
print(f"Mask dtype: {mask.dtype}")    # Should be torch.long
print(f"Mask values: {mask.unique()}") # Should be class indices (0-20 for VOC)
```

### **Method 2: Configuration-Based Usage**

For complex evaluations and reproducibility:

```python
from segmentation_robustness_framework import PipelineConfig

# Load configuration from file
config = PipelineConfig.from_yaml("my_config.yaml")

# Create and run pipeline
results = config.run_pipeline(save=True)

# Verify preprocessing worked correctly
pipeline = config.create_pipeline()
dataset = pipeline.dataset
image, mask = dataset[0]  # Get first sample
print(f"Image shape: {image.shape}")  # Should be [3, 256, 256]
print(f"Image range: [{image.min():.3f}, {image.max():.3f}]")  # Should be normalized
print(f"Mask shape: {mask.shape}")    # Should be [256, 256]
print(f"Mask values: {mask.unique()}") # Should be class indices (0-20 for VOC)
```

**Configuration file (`my_config.yaml`):**

```yaml
model:
  type: "torchvision"
  config:
    name: "deeplabv3_resnet50"
    num_classes: 21
    weights: "default"
  weights_path: null

dataset:
  name: "voc"
  split: "val"
  image_shape: [256, 256]  # Triggers automatic preprocessing
  download: true
  # The framework automatically:
  # - Resizes images to (256, 256) with stride alignment
  # - Normalizes images using ImageNet statistics
  # - Converts RGB masks to class indices (21 classes for VOC)
  # - Handles ignore_index (255 → -1)

attacks:
  - name: "fgsm"
    eps: 0.02
  - name: "pgd"
    eps: 0.04
    alpha: 0.01
    iters: 10

pipeline:
  batch_size: 8
  device: "cuda"
  output_dir: "./results"
  auto_resize_masks: true
  output_formats: ["json", "csv"]

metrics:
  ignore_index: 255
  include_pixel_accuracy: true
  # The framework automatically computes all available metrics:
  # - mean_iou (macro and micro)
  # - pixel_accuracy
  # - precision (macro and micro)
  # - recall (macro and micro)
  # - dice_score (macro and micro)
```

## ⚙️ Configuration-Based Approach

The configuration-based approach is recommended for most use cases because it provides:

- **Reproducibility**: Exact experiments can be repeated
- **Sharing**: Easy to share experiments with others
- **Complexity Management**: Handle complex setups without code

### **Creating Configuration Files**

#### **Basic Configuration Structure**

```yaml
# Required sections
model:          # Model configuration
dataset:        # Dataset configuration  
attacks:        # List of attacks to test
pipeline:       # Pipeline settings
metrics:        # Metrics configuration (optional)
```

#### **Model Configuration Examples**

**Torchvision Models:**
```yaml
model:
  type: "torchvision"
  config:
    name: "deeplabv3_resnet50"  # or fcn_resnet50, lraspp_mobilenet_v3_large
    num_classes: 21
    weights: "COCO_WITH_VOC_LABELS_V1"  # or "default", optional
```

**HuggingFace Models:**
```yaml
model:
  type: "huggingface"
  config:
    model_name: "nvidia/segformer-b2-finetuned-ade-512-512"
    num_labels: 150
    trust_remote_code: false
```

**SMP Models:**
```yaml
model:
  type: "smp"
  config:
    architecture: "unet"
    encoder_name: "resnet34"
    encoder_weights: "imagenet"
    classes: 9
```

**Custom Models:**
```yaml
model:
  type: "custom_myadapter"  # Use registered adapter name as type
  config:
    model_class: "path.to.MyModel"
    model_args:
      num_classes: 21
      pretrained: true
```

**Custom Adapter System:**

For custom models, you need to register an adapter and use its name as the model type:

```yaml
model:
  type: "custom_myadapter"  # Registered adapter name as type
  config:
    model_class: "path.to.MyModel"
    model_args:
      num_classes: 21
```

**Creating Custom Adapters:**

For custom models, you need to create and register an adapter that implements the `SegmentationModelProtocol`:

```python
import torch
from segmentation_robustness_framework.adapters.base_protocol import SegmentationModelProtocol
from segmentation_robustness_framework.adapters.registry import register_adapter

@register_adapter("custom_myadapter")
class MyCustomAdapter(torch.nn.Module, SegmentationModelProtocol):
    """Adapter for custom segmentation models."""
    
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.num_classes = self._extract_num_classes()
    
    def _extract_num_classes(self) -> int:
        """Extract number of classes from model."""
        # Try different common patterns
        if hasattr(self.model, 'num_classes'):
            return self.model.num_classes
        elif hasattr(self.model, 'classifier'):
            if hasattr(self.model.classifier, 'out_channels'):
                return self.model.classifier.out_channels
        elif hasattr(self.model, 'last_layer'):
            if hasattr(self.model.last_layer, 'out_channels'):
                return self.model.last_layer.out_channels
        
        # Default fallback
        return 1
    
    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits for input images."""
        return self.model(x)
    
    def predictions(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted class labels for input images."""
        logits = self.logits(x)
        return torch.argmax(logits, dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.logits(x)
```

**Key Points:**

- **Registration**: Use `@register_adapter("custom_name")` to register your adapter
- **Protocol**: Must implement `SegmentationModelProtocol` methods (`logits`, `predictions`)
- **Model Type**: Use the registered adapter name as the model type
- **Framework Logic**: 
  - Universal loader automatically finds adapter by model type name
  - No separate adapter field needed in config

#### **Dataset Configuration Examples**

**VOC Dataset:**
```yaml
dataset:
  name: "voc"
  split: "val"
  image_shape: [256, 256]
  download: true
```

**ADE20K Dataset:**
```yaml
dataset:
  name: "ade20k"
  split: "val"
  image_shape: [512, 512]
  download: true
```

**Cityscapes Dataset:**
```yaml
# Cityscapes dataset must be downloaded manually
dataset:
  name: "cityscapes"
  split: "val"
  mode: "fine"
  target_type: "semantic"
  image_shape: [512, 1024]
  root: "/path/to/cityscapes"  # Specify path to downloaded data
```

#### **Attack Configuration Examples**

**Single Attack:**
```yaml
attacks:
  - name: "fgsm"
    eps: 0.02
```

**Multiple Attacks:**
```yaml
attacks:
  - name: "fgsm"
    eps: 0.02
  - name: "pgd"
    eps: 0.04
    alpha: 0.01
    iters: 10
    targeted: false
  - name: "rfgsm"
    eps: 0.08
    alpha: 0.02
    iters: 10
  - name: "tpgd"
    eps: 0.08
    alpha: 0.02
    iters: 10
```

#### **Pipeline Configuration Examples**

**Basic Pipeline:**
```yaml
pipeline:
  batch_size: 8
  device: "cuda"
  output_dir: "./results"
```

**Advanced Pipeline:**
```yaml
pipeline:
  batch_size: 4
  device: "cuda"
  output_dir: "./results"
  auto_resize_masks: true
  output_formats: ["json", "csv"]
  num_workers: 4
  pin_memory: true
  metric_precision: 4
```

### **Running from Configuration**

#### **Python Code:**
```python
from segmentation_robustness_framework import PipelineConfig

# Load and run
config = PipelineConfig.from_yaml("config.yaml")
results = config.run_pipeline(save=True, show=False)
```

#### **Command Line:**
```bash
# Basic usage
python -m segmentation_robustness_framework.cli.run_config config.yaml

# With options
python -m segmentation_robustness_framework.cli.run_config config.yaml --save --show

# Override values
python -m segmentation_robustness_framework.cli.run_config config.yaml \
    --override pipeline.batch_size=4 pipeline.device=cpu

# Summary only (preview configuration)
python -m segmentation_robustness_framework.cli.run_config config.yaml --summary-only

# Alternative: Use unified CLI interface
python -m segmentation_robustness_framework.cli.main run config.yaml
python -m segmentation_robustness_framework.cli.main list --attacks
python -m segmentation_robustness_framework.cli.main test --loaders
```

---

## 🎯 Component Selection

### **Choosing the Right Model**

#### **Torchvision Models**

**Best for:** Quick prototyping, standard architectures

**Available:** DeepLabV3, FCN, LRASPP

```python
model_loader = UniversalModelLoader()
model = model_loader.load_model(
    model_type="torchvision",
    model_config={
        "name": "deeplabv3_resnet50",
        "num_classes": 21,
        "weights": "default"
    }
)
```

#### **HuggingFace Models**

**Best for:** State-of-the-art transformers, research

**Available:** SegFormer, DETR, and other transformer models

```python
model = model_loader.load_model(
    model_type="huggingface",
    model_config={
        "model_name": "nvidia/segformer-b2-finetuned-ade-512-512",
        "num_labels": 150
    }
)
```

#### **SMP Models**

**Best for:** Medical imaging, custom architectures

**Available:** UNet, LinkNet, PSPNet, and many more

```python
model = model_loader.load_model(
    model_type="smp",
    model_config={
        "architecture": "unet",
        "encoder_name": "resnet34",
        "encoder_weights": "imagenet",
        "classes": 9
    }
)
```

### **Choosing the Right Dataset**

#### **VOC (Pascal VOC 2012)**

- **Classes:** 21
- **Best for:** General object segmentation, benchmarking
- **Image Size:** `256x256` or `512x512`

```python
dataset_loader = DatasetLoader({
    "name": "voc",
    "split": "val",
    "image_shape": [256, 256]
})
dataset = dataset_loader.load_dataset()
```

#### **ADE20K**

- **Classes:** 150
- **Best for:** Scene parsing, complex scenes
- **Image Size:** `512x512` or `640x640`

```python
dataset_loader = DatasetLoader({
    "name": "ade20k",
    "split": "val",
    "image_shape": [512, 512]
})
dataset = dataset_loader.load_dataset()
```

#### **Cityscapes**

- **Classes:** 35 (19 for evaluation)
- **Best for:** Urban scene understanding, autonomous driving
- **Image Size:** `512x1024`
- **Note:** Must be downloaded manually from https://www.cityscapes-dataset.com/

```python
dataset_loader = DatasetLoader({
    "name": "cityscapes",
    "split": "val",
    "mode": "fine",
    "target_type": "semantic",
    "image_shape": [512, 1024],
    "root": "/path/to/cityscapes"
})
dataset = dataset_loader.load_dataset()
```

#### **Stanford Background**

- **Classes:** 9
- **Best for:** Natural scene segmentation, research
- **Image Size:** `256x256` or `320x320`

```python
dataset_loader = DatasetLoader({
    "name": "stanford_background",
    "split": "train",
    "image_shape": [256, 256]
})
dataset = dataset_loader.load_dataset()
```

### **Choosing the Right Attacks**

The framework provides several adversarial attacks for evaluating model robustness:

#### **FGSM (Fast Gradient Sign Method)**

- **Best for:** Quick evaluation, baseline testing
- **Parameters:** `eps` (perturbation magnitude)

```python
# Using AttackLoader
attack_loader = AttackLoader(model, [
    {"name": "fgsm", "eps": 2/255}
])
attacks_nested = attack_loader.load_attacks()
attacks = [attack for attack_list in attacks_nested for attack in attack_list]
```

#### **PGD (Projected Gradient Descent)**

- **Best for:** Comprehensive evaluation, research
- **Parameters:** `eps`, `alpha` (step size), `iters` (iterations), `targeted`

```python
# Using AttackLoader
attack_loader = AttackLoader(model, [
    {"name": "pgd", "eps": 4/255, "alpha": 1/255, "iters": 10, "targeted": False}
])
attacks_nested = attack_loader.load_attacks()
attacks = [attack for attack_list in attacks_nested for attack in attack_list]
```

#### **RFGSM (Random + Fast Gradient Sign Method)**

- **Best for:** More robust evaluation
- **Parameters:** `eps`, `alpha`, `iters`, `targeted`

```python
# Using AttackLoader
attack_loader = AttackLoader(model, [
    {"name": "rfgsm", "eps": 8/255, "alpha": 2/255, "iters": 10, "targeted": False}
])
attacks_nested = attack_loader.load_attacks()
attacks = [attack for attack_list in attacks_nested for attack in attack_list]
```

#### **TPGD (Theoretically Principled PGD)**

- **Best for:** Advanced research, theoretical analysis
- **Parameters:** `eps`, `alpha`, `iters`

```python
# Using AttackLoader
attack_loader = AttackLoader(model, [
    {"name": "tpgd", "eps": 8/255, "alpha": 2/255, "iters": 10}
])
attacks_nested = attack_loader.load_attacks()
attacks = [attack for attack_list in attacks_nested for attack in attack_list]
```

#### **Multiple Attacks**

```python
# Load multiple attacks at once
attack_loader = AttackLoader(model, [
    {"name": "fgsm", "eps": 2/255},
    {"name": "pgd", "eps": 4/255, "alpha": 1/255, "iters": 10},
    {"name": "rfgsm", "eps": 8/255, "alpha": 2/255, "iters": 10}
])
attacks_nested = attack_loader.load_attacks()  # Returns nested list
attacks = [attack for attack_list in attacks_nested for attack in attack_list]
```

### **Choosing the Right Metrics**

The framework provides comprehensive metrics for evaluating segmentation model performance:

#### **Available Built-in Metrics**

**Mean IoU (Intersection over Union):**

```python
metrics_collection = MetricsCollection(num_classes=21, ignore_index=255)
mean_iou = metrics_collection.mean_iou(targets, preds, average="macro")  # or "micro"
```

**Pixel Accuracy:**

```python
pixel_acc = metrics_collection.pixel_accuracy(targets, preds)
```

**Precision:**

```python
precision = metrics_collection.precision(targets, preds, average="macro")  # or "micro"
```

**Recall:**

```python
recall = metrics_collection.recall(targets, preds, average="macro")  # or "micro"
```

**Dice Score:**

```python
dice = metrics_collection.dice_score(targets, preds, average="macro")  # or "micro"
```

#### **Using Metrics in Pipeline**

```python
# Create metrics collection
metrics_collection = MetricsCollection(num_classes=21, ignore_index=255)

# Create list of metric functions for pipeline
metrics = [
    metrics_collection.mean_iou,  # Uses default macro averaging
    metrics_collection.pixel_accuracy,
    metrics_collection.dice_score,  # Uses default macro averaging
    metrics_collection.precision,   # Uses default macro averaging
    metrics_collection.recall       # Uses default macro averaging
]

# Use in pipeline
pipeline = SegmentationRobustnessPipeline(
    model=model,
    dataset=dataset,
    attacks=attacks,
    metrics=metrics,  # List of callable metric functions
    batch_size=8
)
```

#### **Custom Averaging Strategies**

```python
# Use specific averaging strategies
metrics = [
    lambda targets, preds: metrics_collection.mean_iou(targets, preds, average="macro"),
    lambda targets, preds: metrics_collection.mean_iou(targets, preds, average="micro"),
    lambda targets, preds: metrics_collection.dice_score(targets, preds, average="macro"),
    lambda targets, preds: metrics_collection.dice_score(targets, preds, average="micro"),
    metrics_collection.pixel_accuracy,
]
```

#### **Custom Metrics**

```python
from segmentation_robustness_framework.metrics import register_custom_metric

@register_custom_metric("my_custom_metric")
def my_custom_metric(targets, preds):
    """Custom metric implementation."""
    # Your custom metric logic here
    return score

# Use custom metric in pipeline
metrics = [
    metrics_collection.mean_iou,
    my_custom_metric  # Custom metric function
]
```

---

## 🔧 Data Preprocessing

Understanding how data preprocessing works is crucial for getting accurate results and optimizing performance.

### **Overview**

The framework automatically handles preprocessing for both images and segmentation masks. This includes:

- **Image resizing** to the specified dimensions
- **Normalization** using ImageNet statistics
- **Mask conversion** from RGB to class indices
- **Automatic mask resizing** to match model output dimensions

### **Default Preprocessing Pipeline**

#### **Image Preprocessing**

```python
from torchvision import transforms

# Default image preprocessing
preprocess = transforms.Compose([
    transforms.Resize((height, width), interpolation=transforms.InterpolationMode.NEAREST),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
```

**What happens:**

1. **Resize**: Images are resized to the specified dimensions
2. **ToTensor**: Convert PIL image to tensor with values [0, 1]
3. **Normalize**: Apply ImageNet normalization (mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])

#### **Mask Preprocessing**

```python
def target_preprocess(mask: Image.Image, ignore_index: int = None) -> torch.Tensor:
    # Convert RGB mask to class indices
    # Resize to match image dimensions
    # Handle ignore_index if specified
    return mask_tensor
```

**What happens:**

1. **RGB to Index**: Convert RGB color masks to class indices
2. **Resize**: Resize masks to match image dimensions
3. **Ignore Index**: Mark ignored pixels (usually 255) as -1

### **Automatic Preprocessing**

The framework automatically applies preprocessing when you load datasets:

```python
# Preprocessing is automatically applied
dataset_loader = DatasetLoader()
dataset = dataset_loader.load_dataset(
    name="voc",
    split="val",
    image_shape=(256, 256),  # This triggers preprocessing
    download=True
)
```

**Behind the scenes:**

```python
# The framework automatically calls:
preprocess, target_preprocess = get_preprocessing_fn(
    image_shape=(256, 256), 
    dataset_name="voc"
)
```

### **Image Shape Requirements**

#### **Stride Alignment**

The framework automatically adjusts image dimensions to be divisible by 8 (common stride requirement):

```python
# If you specify (250, 250), it becomes (256, 256)
# If you specify (512, 512), it stays (512, 512)
# If you specify (1000, 1000), it becomes (1000, 1000)

# The adjustment formula:
h = (h // 8 + 1) * 8 if h % 8 != 0 else h
w = (w // 8 + 1) * 8 if w % 8 != 0 else w
```

#### **Common Image Shapes**

```python
# Standard sizes that work well
image_shapes = [
    (256, 256),   # Good for quick testing
    (512, 512),   # Good balance of speed and quality
    (640, 640),   # High quality, slower
    (1024, 1024), # Very high quality, much slower
]

# For specific datasets
voc_shape = (256, 256)      # VOC works well at this size
ade20k_shape = (512, 512)   # ADE20K benefits from larger size
cityscapes_shape = (512, 1024)  # Cityscapes uses rectangular images
```

### **Dataset-Specific Preprocessing**

#### **VOC Dataset**

```python
# VOC uses RGB masks that need conversion to indices
dataset = dataset_loader.load_dataset(
    name="voc",
    image_shape=(256, 256),
    # Automatically converts RGB masks to 21 class indices
)
```

#### **ADE20K Dataset**

```python
# ADE20K also uses RGB masks
dataset = dataset_loader.load_dataset(
    name="ade20k",
    image_shape=(512, 512),
    # Automatically converts RGB masks to 150 class indices
)
```

#### **Cityscapes Dataset**

```python
# Cityscapes uses rectangular images
dataset = dataset_loader.load_dataset(
    name="cityscapes",
    image_shape=(512, 1024),  # Rectangular shape
    # Automatically handles rectangular preprocessing
)
```

### **Custom Preprocessing**

#### **Using Custom Transforms**

```python
from torchvision import transforms
from segmentation_robustness_framework.utils import image_preprocessing

# Get default preprocessing
default_preprocess, default_target_preprocess = image_preprocessing.get_preprocessing_fn(
    image_shape=(256, 256), 
    dataset_name="voc"
)

# Create custom preprocessing
custom_preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    # Custom normalization
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    # Add data augmentation
    transforms.RandomHorizontalFlip(p=0.5),
])

# Use in dataset
dataset = VOCSegmentation(
    root="./data",
    transform=custom_preprocess,
    target_transform=default_target_preprocess
)
```

#### **Custom Normalization**

```python
# For models trained with different normalization
custom_normalize = transforms.Normalize(
    mean=[0.5, 0.5, 0.5],  # Your model's mean
    std=[0.5, 0.5, 0.5]    # Your model's std
)

preprocess = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    custom_normalize,
])
```

### **Mask Processing Details**

#### **RGB to Index Conversion**

```python
# The framework automatically converts RGB masks to class indices
# For VOC dataset:
# RGB color (128, 0, 0) → class index 1
# RGB color (0, 128, 0) → class index 2
# etc.

# This happens automatically when you specify dataset_name
preprocess, target_preprocess = image_preprocessing.get_preprocessing_fn(
    image_shape=(256, 256),
    dataset_name="voc"  # Enables RGB to index conversion
)
```

#### **Handling Ignore Index**

```python
from PIL import Image

# Pixels with value 255 are typically ignored
# The framework automatically converts them to -1
target_preprocess = image_preprocessing.get_preprocessing_fn(
    image_shape=(256, 256),
    dataset_name="voc"
)[1]

# Usage
mask = Image.open("mask.png")
processed_mask = target_preprocess(mask, ignore_index=255)
# Pixels with value 255 become -1
```

### **Automatic Mask Resizing**

The framework automatically resizes masks to match model output dimensions:

```python
# If your model outputs (512, 512) but dataset masks are (256, 256)
# The framework automatically resizes masks to (512, 512)

pipeline = SegmentationRobustnessPipeline(
    model=model,
    dataset=dataset,
    attacks=attacks,
    metrics=metrics,
    auto_resize_masks=True  # Default is True
)
```

**When it happens:**

- Model output size ≠ dataset mask size
- `auto_resize_masks=True` (default)
- Framework detects the mismatch and resizes automatically

### **Preprocessing Best Practices**

#### **1. Choose Appropriate Image Sizes**

```python
# For quick testing
image_shape = (256, 256)

# For production evaluation
image_shape = (512, 512)

# For high-quality results
image_shape = (640, 640)
```

#### **2. Use Dataset-Specific Names**

```python
# Always specify dataset_name for proper mask processing
preprocess, target_preprocess = image_preprocessing.get_preprocessing_fn(
    image_shape=(256, 256),
    dataset_name="voc"  # Important for RGB to index conversion
)
```

#### **3. Handle Memory Constraints**

```python
# For large images, reduce batch size
pipeline = SegmentationRobustnessPipeline(
    model=model,
    dataset=dataset,
    attacks=attacks,
    metrics=metrics,
    batch_size=2,  # Smaller batch for large images
    device="cuda"
)
```

#### **4. Verify Preprocessing**

```python
# Check that preprocessing works correctly
dataset = dataset_loader.load_dataset(
    name="voc",
    image_shape=(256, 256)
)

# Test a sample
image, mask = dataset[0]
print(f"Image shape: {image.shape}")  # Should be [3, 256, 256]
print(f"Mask shape: {mask.shape}")    # Should be [256, 256]
print(f"Mask dtype: {mask.dtype}")    # Should be torch.long
print(f"Mask values: {mask.unique()}") # Should be class indices
```

### **Troubleshooting Preprocessing**

#### **Common Issues**

**1. Wrong Image Dimensions**

```python
# Problem: Model expects different input size
# Solution: Check model input requirements
model_input_size = (512, 512)
dataset = dataset_loader.load_dataset(
    name="voc",
    image_shape=model_input_size  # Match model expectations
)
```

**2. Mask Conversion Errors**

```python
# Problem: RGB masks not converted to indices
# Solution: Specify dataset_name
preprocess, target_preprocess = image_preprocessing.get_preprocessing_fn(
    image_shape=(256, 256),
    dataset_name="voc"  # Enables RGB to index conversion
)
```

**3. Memory Issues with Large Images**

```python
# Problem: Out of memory with large images
# Solution: Reduce batch size or image size
pipeline = SegmentationRobustnessPipeline(
    batch_size=1,  # Reduce batch size
    # Or use smaller image_shape in dataset
)
```

**4. Stride Alignment Issues**

```python
# Problem: Image dimensions not divisible by 8
# Solution: Framework automatically fixes this
# If you specify (250, 250), it becomes (256, 256)
# No action needed - it's automatic
```

---

## 📊 Understanding Results

### **Result Structure**

The pipeline returns a dictionary with the following structure:

```python
results = {
    "clean": {
        "mean_iou": 0.823,
        "pixel_accuracy": 0.956,
        "dice_score": 0.891,
        # ... other metrics
    },
    "attack_fgsm_eps_0p008": {  # Attack name includes parameters
        "mean_iou": 0.234,
        "pixel_accuracy": 0.567,
        "dice_score": 0.345,
        # ... other metrics
    },
    "attack_pgd_eps_0p016_alpha_0p004_iters_10": {
        "mean_iou": 0.123,
        "pixel_accuracy": 0.456,
        "dice_score": 0.234,
        # ... other metrics
    }
}
```

### **Key Metrics Explained**

#### **Mean IoU (Intersection over Union)**

- **Range:** 0.0 to 1.0 (higher is better)
- **Meaning:** Average overlap between predicted and ground truth masks
- **Interpretation:** 
  - 0.8+ = Excellent
  - 0.6-0.8 = Good
  - 0.4-0.6 = Fair
  - <0.4 = Poor

#### **Pixel Accuracy**

- **Range:** 0.0 to 1.0 (higher is better)
- **Meaning:** Percentage of correctly classified pixels
- **Interpretation:**
  - 0.95+ = Excellent
  - 0.85-0.95 = Good
  - 0.75-0.85 = Fair
  - <0.75 = Poor

#### **Dice Score**

- **Range:** 0.0 to 1.0 (higher is better)
- **Meaning:** Harmonic mean of precision and recall
- **Interpretation:**
  - 0.9+ = Excellent
  - 0.7-0.9 = Good
  - 0.5-0.7 = Fair
  - <0.5 = Poor

### **Robustness Analysis**

#### **Robustness Drop**

```python
# Calculate how much performance drops under attack
clean_iou = results['clean']['mean_iou']
attack_iou = results['attack_fgsm_eps_0p008']['mean_iou']
robustness_drop = clean_iou - attack_iou

print(f"Clean IoU: {clean_iou:.3f}")
print(f"Attack IoU: {attack_iou:.3f}")
print(f"Robustness drop: {robustness_drop:.3f}")
```

#### **Robustness Percentage**

```python
# Calculate percentage of performance retained
robustness_percentage = (attack_iou / clean_iou) * 100
print(f"Robustness: {robustness_percentage:.1f}%")
```

### **Interpreting Results**

#### **Good Robustness**

```python
# Clean IoU: 0.823, Attack IoU: 0.712
# Robustness drop: 0.111 (13.5% drop)
# This model shows good robustness
```

#### **Poor Robustness**

```python
# Clean IoU: 0.823, Attack IoU: 0.234
# Robustness drop: 0.589 (71.6% drop)
# This model is vulnerable to attacks
```

#### **Comparing Attacks**

```python
# FGSM drop: 0.111
# PGD drop: 0.234
# PGD is more effective at fooling this model
```

---

## 🔧 Advanced Usage

### **Custom Model Integration**

#### **Using Custom Models**

To use custom models with the framework, you need to:

1. **Create an adapter** that implements `SegmentationModelProtocol`
2. **Register the adapter** with a unique name
3. **Use the universal model loader** with the registered adapter name

```python
import torch
import torch.nn as nn
from segmentation_robustness_framework.adapters.base_protocol import SegmentationModelProtocol
from segmentation_robustness_framework.adapters.registry import register_adapter
from segmentation_robustness_framework.loaders.models.universal_loader import UniversalModelLoader

# Step 1: Define your custom model
class MyCustomModel(nn.Module):
    def __init__(self, num_classes):
        super().__init__()
        self.num_classes = num_classes
        # Your model architecture here
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, 1)
        )
    
    def forward(self, x):
        return self.backbone(x)

# Step 2: Create and register adapter
@register_adapter("custom_my_model")
class MyCustomAdapter(torch.nn.Module, SegmentationModelProtocol):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.num_classes = self._extract_num_classes()
    
    def _extract_num_classes(self) -> int:
        """Extract number of classes from model."""
        if hasattr(self.model, 'num_classes'):
            return self.model.num_classes
        # Try to infer from last layer
        for module in reversed(list(self.model.modules())):
            if hasattr(module, 'out_channels'):
                return module.out_channels
        return 1  # Default fallback
    
    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits for input images."""
        return self.model(x)
    
    def predictions(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted class labels for input images."""
        logits = self.logits(x)
        return torch.argmax(logits, dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.logits(x)

# Step 3: Use universal model loader
model_loader = UniversalModelLoader()
model = model_loader.load_model(
    model_type="custom_my_model",  # Use registered adapter name as type
    model_config={
        "model_class": "path.to.MyCustomModel",
        "model_args": [21],  # num_classes
        "model_kwargs": {"pretrained": True}
    }
)
```

#### **Alternative: Direct Adapter Usage**

You can also pass the adapter class directly to the universal loader:

```python
# Without registration, pass adapter class directly
model = model_loader.load_model(
    model_type="custom",  # Must contain "custom"
    model_config={
        "model_class": "path.to.MyCustomModel",
        "model_args": [21]
    },
    adapter_cls=MyCustomAdapter  # Direct class reference
)
```

### **Custom Metrics**

#### **Registering Custom Metrics**
```python
from segmentation_robustness_framework.metrics import register_custom_metric

@register_custom_metric("my_custom_metric")
def my_custom_metric(targets, predictions):
    """Calculate custom metric."""
    # Your metric calculation
    return score

# Use in pipeline
metrics_collection = MetricsCollection(num_classes=21, ignore_index=255)

# Create list of metric functions including custom metric
metrics = [
    metrics_collection.mean_iou,
    metrics_collection.pixel_accuracy,
    my_custom_metric  # Custom metric function
]

# Use in pipeline
pipeline = SegmentationRobustnessPipeline(
    model=model,
    dataset=dataset,
    attacks=attacks,
    metrics=metrics,  # List of callable functions
    batch_size=8
)
```

### **Batch Processing for Large Datasets**

#### **Memory-Efficient Processing**
```python
# Use smaller batch sizes for large models
pipeline = SegmentationRobustnessPipeline(
    model=model,
    dataset=dataset,
    attacks=attacks,
    metrics=metrics,
    batch_size=2,  # Smaller batch size
    device="cuda"
)
```

#### **Multi-GPU Processing**

**Note**: The current framework implementation does not have built-in multi-GPU support. The pipeline is designed for single-GPU evaluation.

**Current Limitations:**

- The pipeline uses a single device specified by the `device` parameter
- No automatic distribution across multiple GPUs
- Attacks and metrics are not optimized for multi-GPU setups

**Single-GPU Usage (Recommended):**

```python
# Use single GPU with optimized settings
pipeline = SegmentationRobustnessPipeline(
    model=model,
    dataset=dataset,
    attacks=attacks,
    metrics=metrics,
    batch_size=8,  # Adjust based on GPU memory
    device="cuda:0"  # Specify specific GPU
)
```

**Future Multi-GPU Support:**

Multi-GPU support is planned for future versions and will include:

- Automatic model distribution across available GPUs
- Parallel attack evaluation
- Distributed metric computation

### **Custom Attack Parameters**

#### **Experimenting with Attack Strengths**

```python
# Test different perturbation magnitudes
attack_configs = []
for eps in [1/255, 2/255, 4/255, 8/255]:
    attack_configs.append({"name": "fgsm", "eps": eps})

# Load all attacks at once
attack_loader = AttackLoader(model, attack_configs)
attacks_nested = attack_loader.load_attacks()
attacks = [attack for attack_list in attacks_nested for attack in attack_list]

pipeline = SegmentationRobustnessPipeline(
    model=model,
    dataset=dataset,
    attacks=attacks,
    metrics=metrics
)
```

#### **Targeted vs Untargeted Attacks**

```python
# Load multiple attack configurations
attack_configs = [
    {"name": "pgd", "eps": 4/255, "alpha": 1/255, "iters": 10, "targeted": False},
    {"name": "pgd", "eps": 4/255, "alpha": 1/255, "iters": 10, "targeted": True}
]

# Load attacks
attack_loader = AttackLoader(model, attack_configs)
attacks_nested = attack_loader.load_attacks()
attacks = [attack for attack_list in attacks_nested for attack in attack_list]
```

---

## ⚡ Performance Optimization

### **Memory Management**

#### **GPU Memory Optimization**

```python
# Clear GPU cache between batches
import torch

def memory_cleanup():
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

# Use in pipeline
pipeline = SegmentationRobustnessPipeline(
    model=model,
    dataset=dataset,
    attacks=attacks,
    metrics=metrics,
    batch_size=4,  # Adjust based on GPU memory
    device="cuda"
)
```

#### **Batch Size Tuning**

```python
# Start with small batch size and increase
batch_sizes = [1, 2, 4, 8]

for batch_size in batch_sizes:
    try:
        pipeline = SegmentationRobustnessPipeline(
            model=model,
            dataset=dataset,
            attacks=attacks,
            metrics=metrics,
            batch_size=batch_size,
            device="cuda"
        )
        results = pipeline.run()
        print(f"Success with batch_size={batch_size}")
        break
    except RuntimeError as e:
        if "out of memory" in str(e):
            print(f"OOM with batch_size={batch_size}")
            continue
        raise e
```

### **Computation Optimization**

```python
# Optimize data loading
pipeline = SegmentationRobustnessPipeline(
    model=model,
    dataset=dataset,
    attacks=attacks,
    metrics=metrics,
    batch_size=8,
    num_workers=4,  # Parallel data loading
    pin_memory=True,  # Faster GPU transfer
    device="cuda"
)
```

---

## 🔍 Troubleshooting

### **Common Issues and Solutions**

#### **Out of Memory (OOM) Errors**

**Problem:** `RuntimeError: CUDA out of memory`

**Solutions:**

```python
# 1. Reduce batch size
pipeline = SegmentationRobustnessPipeline(
    batch_size=1,  # Start with 1
    device="cuda"
)

# 2. Clear GPU cache
import torch
torch.cuda.empty_cache()

# 3. Use CPU if GPU memory is insufficient
pipeline = SegmentationRobustnessPipeline(
    device="cpu"
)

# 4. Use gradient checkpointing for large models
model.gradient_checkpointing_enable()
```

#### **Model Loading Issues**

**Problem:** `ModuleNotFoundError: No module named 'model_name'`

**Solutions:**

```python
# 1. Check model name spelling
model = model_loader.load_model(
    model_type="torchvision",
    model_name="deeplabv3_resnet50"  # Correct spelling
)

# 2. Install required dependencies
pip install torchvision
pip install transformers  # For HuggingFace models
pip install segmentation-models-pytorch  # For SMP models

# 3. Use custom model loader for custom models
model = model_loader.load_model(
    model_type="custom",
    model_class="path.to.MyModel"
)
```

#### **Dataset Loading Issues**

**Problem:** `FileNotFoundError: Dataset not found`

**Solutions:**

```python
# 1. Enable automatic download
dataset_loader = DatasetLoader({
    "name": "voc",
    "split": "val",
    "image_shape": [256, 256],
    "download": True  # This will download the dataset
})
dataset = dataset_loader.load_dataset()

# 2. Specify correct path for manually downloaded datasets
dataset_loader = DatasetLoader({
    "name": "cityscapes",
    "split": "val",
    "image_shape": [512, 1024],
    "root": "/path/to/cityscapes"  # Specify correct path
})
dataset = dataset_loader.load_dataset()

# 3. Check dataset availability
from segmentation_robustness_framework.cli import list_components_main
list_components_main()  # Lists available datasets
```

#### **Attack Configuration Issues**

**Problem:** `ValueError: Invalid attack parameters`

**Solutions:**

```python
# 1. Check parameter ranges
attack_loader = AttackLoader(model, [
    {"name": "fgsm", "eps": 2/255}  # eps should be between 0 and 1
])
attacks_nested = attack_loader.load_attacks()
attacks = [attack for attack_list in attacks_nested for attack in attack_list]

# 2. Use correct parameter names
attack_loader = AttackLoader(model, [
    {"name": "pgd", "eps": 4/255, "alpha": 1/255, "iters": 10, "targeted": False}
])
attacks_nested = attack_loader.load_attacks()
attacks = [attack for attack_list in attacks_nested for attack in attack_list]

# 3. Check attack availability
from segmentation_robustness_framework.cli import list_components_main
list_components_main()  # Lists available attacks
```

#### **Configuration File Issues**

**Problem:** `YAMLError: Invalid YAML syntax`

**Solutions:**

```yaml
# 1. Check YAML syntax
model:
  type: "torchvision"  # Use quotes for strings
  config:
    name: "deeplabv3_resnet50"
    num_classes: 21  # Numbers don't need quotes

# 2. Validate configuration structure
python -m segmentation_robustness_framework.cli.run_config config.yaml --summary-only
```

#### **Performance Issues**

**Problem:** Evaluation is very slow

**Solutions:**

```python
# 1. Use GPU if available
pipeline = SegmentationRobustnessPipeline(
    device="cuda"  # Use GPU
)

# 2. Increase batch size if memory allows
pipeline = SegmentationRobustnessPipeline(
    batch_size=8  # Larger batch size
)

# 3. Use fewer attacks for faster evaluation
attack_loader = AttackLoader(model, [{"name": "fgsm", "eps": 2/255}])
attacks_nested = attack_loader.load_attacks()
attacks = [attack for attack_list in attacks_nested for attack in attack_list]

# 4. Use smaller dataset for testing
dataset_loader = DatasetLoader({
    "name": "voc",
    "split": "val",
    "image_shape": [256, 256],
    "download": True
})
dataset = dataset_loader.load_dataset()
```

### **Debugging Tips**

#### **Enable Verbose Logging**

```python
import logging
logging.basicConfig(level=logging.INFO)

# This will show detailed progress
pipeline = SegmentationRobustnessPipeline(...)
results = pipeline.run()
```

#### **Check Component Loading**

```python
# Test each component individually
model_loader = UniversalModelLoader()
model = model_loader.load_model(
    model_type="torchvision",
    model_config={"name": "deeplabv3_resnet50", "num_classes": 21}
)
print(f"Model loaded: {type(model)}")

dataset_loader = DatasetLoader({
    "name": "voc",
    "split": "val",
    "image_shape": [256, 256]
})
dataset = dataset_loader.load_dataset()
print(f"Dataset loaded: {len(dataset)} samples")

attack_loader = AttackLoader(model, [{"name": "fgsm", "eps": 2/255}])
attacks_nested = attack_loader.load_attacks()
attacks = [attack for attack_list in attacks_nested for attack in attack_list]
print(f"Attacks loaded: {len(attacks)} attacks")
```

#### **Validate Configuration**

```python
# Check configuration before running
config = PipelineConfig.from_yaml("config.yaml")
summary = config.get_config_summary()
print(summary)
```

---

## 📚 Next Steps

Now that you understand the basics, explore:

1. **[Advanced Examples](examples/advanced_examples.md)** - Learn how to add your own models, datasets, and attacks
2. **[API Reference](api_reference/index.md)** - Detailed documentation of all classes and methods
3. **[Configuration Guide](configuration_guide.md)** - Advanced configuration options and techniques
4. **[Examples Gallery](examples/basic_examples.md)** - Real-world examples and use cases

## 🤝 Getting Help

- **GitHub Issues**: Report bugs and request features
- **Documentation**: Check the [API Reference](api_reference/index.md) for detailed information
- **Examples**: See [Examples Gallery](examples/basic_examples.md) for real-world use cases
- **Community**: Join discussions on GitHub Discussions
