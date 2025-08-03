# User Guide

This comprehensive guide covers all aspects of using the Segmentation Robustness Framework, from basic usage to advanced customization.

## 📚 Table of Contents

1. [Getting Started](#getting-started)
2. [Basic Usage](#basic-usage)
3. [Model Loading](#model-loading)
4. [Dataset Management](#dataset-management)
5. [Attack Configuration](#attack-configuration)
6. [Metrics and Evaluation](#metrics-and-evaluation)
7. [Pipeline Configuration](#pipeline-configuration)
8. [Results and Visualization](#results-and-visualization)
9. [Advanced Features](#advanced-features)
10. [Troubleshooting](#troubleshooting)

## 🚀 Getting Started

### Prerequisites

Before using the framework, ensure you have:

```bash
# Core dependencies
pip install torch torchvision
pip install segmentation-robustness-framework[full]

# Optional dependencies for specific features
pip install transformers  # For HuggingFace models
pip install segmentation-models-pytorch  # For SMP models
```

### Basic Import Structure

```python
# Core framework imports
from segmentation_robustness_framework.engine.pipeline import SegmentationRobustnessPipeline
from segmentation_robustness_framework.utils.metrics import MetricsCollection
from segmentation_robustness_framework.attacks import FGSM, PGD, RFGSM, TPGD
from segmentation_robustness_framework.datasets import VOCSegmentation, ADE20K, Cityscapes, StanfordBackground
from segmentation_robustness_framework.loaders.models.universal_loader import UniversalModelLoader

# Adapters for different model types
from segmentation_robustness_framework.adapters import (
    TorchvisionAdapter, SMPAdapter, HuggingFaceAdapter, CustomAdapter
)
```

## 🔧 Basic Usage

### Complete Example

Here's a complete example that demonstrates the basic workflow:

```python
from segmentation_robustness_framework.engine.pipeline import SegmentationRobustnessPipeline
from segmentation_robustness_framework.utils.metrics import MetricsCollection
from segmentation_robustness_framework.attacks import FGSM
from segmentation_robustness_framework.datasets import VOCSegmentation
from segmentation_robustness_framework.loaders.models.universal_loader import UniversalModelLoader
from segmentation_robustness_framework.utils import image_preprocessing
import torch

# 1. Load model
loader = UniversalModelLoader()
model = loader.load_model(
    model_type="torchvision",
    model_config={"name": "deeplabv3_resnet50", "num_classes": 21}
)

# 2. Set device and move model to it (IMPORTANT: Do this before creating attacks!)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# 3. Load dataset with preprocessing
preprocess, target_preprocess = image_preprocessing.get_preprocessing_fn(
    [512, 512], dataset_name="voc"
)
dataset = VOCSegmentation(
    split="val", 
    root="path/to/dataset/storage", 
    transform=preprocess,
    target_transform=target_preprocess,
    download=True
)

# 4. Setup attack (attacks will use the same device as the model)
attack = FGSM(model, eps=2/255)

# 5. Setup metrics
metrics_collection = MetricsCollection(num_classes=21)
metrics = [metrics_collection.mean_iou, metrics_collection.pixel_accuracy]

# 6. Create and run pipeline
pipeline = SegmentationRobustnessPipeline(
    model=model,
    dataset=dataset,
    attacks=[attack],
    metrics=metrics,
    batch_size=4,
    device=device
)

results = pipeline.run(save=True, show=True)
pipeline.print_summary()
```

## 🏗️ Model Loading

### Universal Model Loader

The `UniversalModelLoader` provides a unified interface for loading different model types:

```python
loader = UniversalModelLoader()

# Torchvision models
model = loader.load_model(
    model_type="torchvision",
    model_config={
        "name": "deeplabv3_resnet50",
        "num_classes": 21,
        "weights": "DEFAULT"  # or None, or specific weights
    }
)

# SMP models
model = loader.load_model(
    model_type="smp",
    model_config={
        "architecture": "unet",
        "encoder_name": "resnet34",
        "classes": 21,
        "encoder_weights": "imagenet"
    }
)

# HuggingFace models
model = loader.load_model(
    model_type="huggingface",
    model_config={
        "model_name": "nvidia/segformer-b0-finetuned-ade-512-512",
        "num_labels": 150
    }
)

# Custom models
model = loader.load_model(
    model_type="custom",
    model_config={
        "model_class": "path.to.MyModel",
        "model_args": [arg1, arg2],
        "model_kwargs": {"param": "value"}
    }
)

# IMPORTANT: Set device after loading model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
```

### Direct Model Loading

You can also load models directly and wrap them with adapters:

```python
import torchvision.models.segmentation as tv_segmentation
import segmentation_models_pytorch as smp

# Torchvision model with adapter
base_model = tv_segmentation.deeplabv3_resnet50(num_classes=21)
model = TorchvisionAdapter(base_model)

# SMP model with adapter
base_model = smp.Unet(encoder_name="resnet34", classes=21)
model = SMPAdapter(base_model)

# HuggingFace model with adapter
from transformers import SegformerForSemanticSegmentation
base_model = SegformerForSemanticSegmentation.from_pretrained(
    "nvidia/segformer-b0-finetuned-ade-512-512"
)
model = HuggingFaceAdapter(base_model)

# Custom model with adapter
from segmentation_robustness_framework.adapters import CustomAdapter
base_model = MyCustomSegmentationModel(num_classes=21)
model = CustomAdapter(base_model, num_classes=21)

# IMPORTANT: Set device after creating adapters
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
```

### Custom Model Workflow

For custom models, you need to create your own adapter that implements the `SegmentationModelProtocol`. The framework provides a template `CustomAdapter` that you can extend.

#### Step 1: Create Your Custom Adapter

```python
from segmentation_robustness_framework.adapters import CustomAdapter
from segmentation_robustness_framework.adapters.registry import register_adapter

# Option 1: Extend the template adapter
@register_adapter("my_custom")
class MyCustomAdapter(CustomAdapter):
    def logits(self, x: torch.Tensor) -> torch.Tensor:
        # Handle your model's specific output format
        output = self.model(x)
        if isinstance(output, dict):
            return output['logits']  # If your model returns a dict
        elif hasattr(output, 'logits'):
            return output.logits     # If your model returns an object with logits
        else:
            return output            # If your model returns logits directly

# Option 2: Create a completely custom adapter
@register_adapter("my_special_model")
class MySpecialAdapter(torch.nn.Module, SegmentationModelProtocol):
    def __init__(self, model: torch.nn.Module, num_classes: int = 21):
        super().__init__()
        self.model = model
        self.num_classes = num_classes
    
    def logits(self, x: torch.Tensor) -> torch.Tensor:
        # Your custom logic here
        return self.model(x)
    
    def predictions(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.logits(x)
        return torch.argmax(logits, dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.logits(x)
```

#### Step 2: Use Your Custom Adapter

```python
# Load your custom model
base_model = MyCustomSegmentationModel(num_classes=21)

# Create and register your adapter
adapter = MyCustomAdapter(base_model, num_classes=21)

# Set device and create attacks
device = "cuda" if torch.cuda.is_available() else "cpu"
adapter = adapter.to(device)
attacks = [FGSM(adapter, eps=2/255)]

# Use in pipeline
pipeline = SegmentationRobustnessPipeline(
    model=adapter, dataset=dataset, attacks=attacks, metrics=metrics, device=device
)
results = pipeline.run()
```

#### Step 3: Alternative - Direct Usage Without Registration

```python
# If you don't want to register your adapter, use it directly
base_model = MyCustomSegmentationModel(num_classes=21)
adapter = MyCustomAdapter(base_model, num_classes=21)

# Set device and create attacks
device = "cuda" if torch.cuda.is_available() else "cpu"
adapter = adapter.to(device)
attacks = [FGSM(adapter, eps=2/255)]

pipeline = SegmentationRobustnessPipeline(
    model=adapter, dataset=dataset, attacks=attacks, metrics=metrics, device=device
)
results = pipeline.run()
```

#### Common Custom Model Output Formats

Different models return outputs in different formats. Here are common patterns:

```python
# Model returns logits directly: [B, num_classes, H, W]
class SimpleAdapter(CustomAdapter):
    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

# Model returns dict with 'logits' key
class DictOutputAdapter(CustomAdapter):
    def logits(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        return output['logits']

# Model returns object with .logits attribute
class ObjectOutputAdapter(CustomAdapter):
    def logits(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        return output.logits

# Model returns tuple (logits, aux_output)
class TupleOutputAdapter(CustomAdapter):
    def logits(self, x: torch.Tensor) -> torch.Tensor:
        output = self.model(x)
        return output[0]  # Return main logits, ignore aux output
```

#### Important Notes

1. **Your adapter must implement `SegmentationModelProtocol`** - This ensures compatibility with the framework
2. **The `logits` method must return shape `[B, num_classes, H, W]`** - This is the standard format expected by attacks and metrics
3. **Set the device on your adapter before creating attacks** - Attacks derive their device from the model
4. **Register your adapter if you want to use it with the universal loader** - Otherwise, use it directly in the pipeline

### Model Configuration Options

#### Torchvision Models

```python
# Available models
SUPPORTED_MODELS = {
    "deeplabv3_resnet50": deeplabv3_resnet50,
    "deeplabv3_resnet101": deeplabv3_resnet101,
    "deeplabv3_mobilenetv3_large": deeplabv3_mobilenet_v3_large,
    "fcn_resnet50": fcn_resnet50,
    "fcn_resnet101": fcn_resnet101,
    "lraspp_mobilenet_v3_large": lraspp_mobilenet_v3_large,
}

# Configuration options
config = {
    "name": "deeplabv3_resnet50",
    "num_classes": 21,
    "weights": "DEFAULT",  # or None, or specific weights
    "device": "cuda"
}
```

#### SMP Models

```python
# Configuration options
config = {
    "architecture": "unet",  # unet, linknet, pspnet, etc.
    "encoder_name": "resnet34",  # resnet18, resnet34, resnet50, etc.
    "encoder_weights": "imagenet",  # imagenet, None
    "classes": 21,
    "activation": None,  # sigmoid, softmax, etc.
}
```

#### HuggingFace Models

```python
# Configuration options
config = {
    "model_name": "nvidia/segformer-b0-finetuned-ade-512-512",
    "num_labels": 150,
    "trust_remote_code": False,
    "task": "semantic_segmentation",
    "return_processor": True,
    "config_overrides": {},
    "processor_overrides": {},
}
```

## 📊 Dataset Management

### Built-in Datasets

The framework provides several built-in datasets with automatic preprocessing:

```python
from segmentation_robustness_framework.utils import image_preprocessing

# Get preprocessing functions
preprocess, target_preprocess = image_preprocessing.get_preprocessing_fn(
    [512, 512], dataset_name="voc"
)

# VOC dataset (21 classes)
dataset = VOCSegmentation(
    split="val",  # train, val, trainval
    root="path/to/dataset/storage",
    transform=preprocess,
    target_transform=target_preprocess,
    download=True
)

# ADE20K dataset (150 classes)
preprocess, target_preprocess = image_preprocessing.get_preprocessing_fn(
    [512, 512], dataset_name="ade20k"
)
dataset = ADE20K(
    split="val",  # train, val
    root="path/to/dataset/storage",
    transform=preprocess,
    target_transform=target_preprocess,
    download=True
)

# Cityscapes dataset (35 classes)
# NOTE: Cityscapes cannot be downloaded automatically due to required authorization on the official website.
# You must register and download the dataset manually from https://www.cityscapes-dataset.com/ and place it in the specified root directory.
preprocess, target_preprocess = image_preprocessing.get_preprocessing_fn(
    [512, 512], dataset_name="cityscapes"
)
dataset = Cityscapes(
    root="path/to/dataset/storage",  # <-- Place manually downloaded Cityscapes data here
    split="val",  # train, val, test, train_extra
    mode="fine",  # fine, coarse
    target_type="semantic",  # semantic, instance, color, polygon
    transform=preprocess,
    target_transform=target_preprocess
)

# Stanford Background dataset (9 classes)
preprocess, target_preprocess = image_preprocessing.get_preprocessing_fn(
    [512, 512], dataset_name="stanford_background"
)
dataset = StanfordBackground(
    root="path/to/dataset/storage",
    transform=preprocess,
    target_transform=target_preprocess,
    download=True
)
```

### Dataset Configuration

```python
# Common configuration options
dataset_config = {
    "transform": preprocess,  # Image transformations
    "target_transform": target_preprocess,  # Mask transformations
    "download": True,  # Auto-download if not present
    "root": "path/to/dataset/storage"  # Dataset location
}
```

### Custom Datasets

Create your own dataset by inheriting from `torch.utils.data.Dataset`:

```python
from torch.utils.data import Dataset
from segmentation_robustness_framework.datasets.registry import register_dataset

@register_dataset("my_dataset")
class MyCustomDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.num_classes = 10  # Set your number of classes
        
        # Load file paths
        self.images = [...]  # List of image paths
        self.masks = [...]   # List of mask paths
    
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
```

## ⚔️ Attack Configuration

### Built-in Attacks

The framework provides several adversarial attacks:

```python
# IMPORTANT: Set device on model before creating attacks
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# FGSM (Fast Gradient Sign Method)
fgsm = FGSM(model, eps=2/255)

# PGD (Projected Gradient Descent)
pgd = PGD(
    model, 
    eps=4/255, 
    alpha=1/255, 
    iters=10, 
    targeted=False
)

# RFGSM (Random + Fast Gradient Sign Method)
rfgsm = RFGSM(
    model, 
    eps=8/255, 
    alpha=2/255, 
    iters=10, 
    targeted=False
)

# TPGD (Theoretically Principled PGD)
tpgd = TPGD(
    model, 
    eps=8/255, 
    alpha=2/255, 
    iters=10
)
```

### Attack Parameters

#### FGSM Parameters
```python
FGSM(model, eps=2/255)
# eps: Maximum perturbation magnitude (0.0 to 1.0)
```

#### PGD Parameters
```python
PGD(model, eps=4/255, alpha=1/255, iters=10, targeted=False)
# eps: Maximum perturbation magnitude
# alpha: Step size for each iteration
# iters: Number of iterations
# targeted: Whether to perform targeted attack
```

#### RFGSM Parameters
```python
RFGSM(model, eps=8/255, alpha=2/255, iters=10, targeted=False)
# eps: Maximum perturbation magnitude
# alpha: Step size for each iteration
# iters: Number of iterations
# targeted: Whether to perform targeted attack
```

#### TPGD Parameters
```python
TPGD(model, eps=8/255, alpha=2/255, iters=10)
# eps: Maximum perturbation magnitude
# alpha: Step size for each iteration
# iters: Number of iterations
```

### Custom Attacks

Create your own attack by inheriting from `AdversarialAttack`:

```python
from segmentation_robustness_framework.attacks.attack import AdversarialAttack
from segmentation_robustness_framework.attacks.registry import register_attack

@register_attack("my_attack")
class MyCustomAttack(AdversarialAttack):
    def __init__(self, model, eps=8/255, targeted=False):
        super().__init__(model)
        self.eps = eps
        self.targeted = targeted
    
    def apply(self, images, labels):
        """Apply the attack.
        
        Args:
            images: Input images [B, C, H, W]
            labels: Target labels [B, H, W]
            
        Returns:
            Adversarial images [B, C, H, W]
        """
        self.model.eval()
        
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # Your attack implementation here
        # ...
        
        return adversarial_images
```

## 📈 Metrics and Evaluation

### Built-in Metrics

The `MetricsCollection` class provides several evaluation metrics:

```python
metrics_collection = MetricsCollection(num_classes=21)

# Available metrics
metrics = [
    metrics_collection.mean_iou,           # Mean Intersection over Union (macro/micro)
    metrics_collection.pixel_accuracy,     # Pixel accuracy
    metrics_collection.precision,          # Precision (macro/micro)
    metrics_collection.recall,             # Recall (macro/micro)
    metrics_collection.dice_score,         # Dice score (macro/micro)
]
```

### Metric Configuration

```python
# Configure metrics with different averaging strategies
metrics_collection = MetricsCollection(
    num_classes=21,
    ignore_index=255  # Ignore certain labels
)

# Get metrics with specific averaging
macro_iou = metrics_collection.get_metric_with_averaging("mean_iou", "macro")
micro_iou = metrics_collection.get_metric_with_averaging("mean_iou", "micro")

# Get all metrics with averaging
all_metrics = metrics_collection.get_all_metrics_with_averaging(
    include_pixel_accuracy=True
)
```

### Custom Metrics

Create your own evaluation metrics:

```python
def custom_metric(targets, predictions):
    """Custom evaluation metric.
    
    Args:
        targets: Ground truth labels [B, H, W]
        predictions: Predicted labels [B, H, W]
    
    Returns:
        float: Metric value
    """
    # Your metric implementation
    return score

# Use with pipeline
metrics = [metrics_collection.mean_iou, custom_metric]
```

## ⚙️ Pipeline Configuration

### Basic Pipeline Setup

```python
pipeline = SegmentationRobustnessPipeline(
    model=model,
    dataset=dataset,
    attacks=attacks,
    metrics=metrics,
    batch_size=4,
    device="cuda",
    output_dir="./results",
    auto_resize_masks=True,
    metric_names=None,  # Auto-generated if None
    output_formats=["json", "csv"]
)
```

### Pipeline Parameters

#### Required Parameters
- `model`: Segmentation model with adapter
- `dataset`: Dataset instance
- `attacks`: List of attack instances
- `metrics`: List of metric functions

#### Optional Parameters
- `batch_size`: Batch size for evaluation (default: 8)
- `device`: Device for computation (default: "cuda")
- `output_dir`: Directory for saving results (default: None)
- `auto_resize_masks`: Auto-resize masks to model output (default: True)
- `metric_names`: Names for metrics (default: None)
- `output_formats`: Output formats ["json", "csv"] (default: ["json", "csv"])

## 📊 Results and Visualization

### Running Evaluation

```python
# Run evaluation with different options
results = pipeline.run(
    save=True,    # Save results to files
    show=True     # Show visualizations
)

# Get summary
summary = pipeline.get_summary()
pipeline.print_summary()
```

### Understanding Results

The pipeline returns a structured results dictionary:

```python
results = {
    "clean": {
        "mean_iou": 0.85,
        "pixel_accuracy": 0.92,
        "precision": 0.87,
        "recall": 0.83,
        "dice_score": 0.85
    },
    "attack_FGSM_epsilon_0p008": {
        "mean_iou": 0.45,
        "pixel_accuracy": 0.67,
        "precision": 0.52,
        "recall": 0.48,
        "dice_score": 0.50
    },
    # ... more attack results
}
```

### Output Files

The pipeline generates several output files:

```
results/
├── run_20231201_143022_abc12345/
│   ├── clean_detailed.json                      # Detailed clean results
│   ├── attack_FGSM_epsilon_0p008_detailed.json  # FGSM results
│   ├── summary_results.json                     # Aggregated results
│   ├── comparison_table.csv                     # Results comparison
│   ├── mean_iou_comparison.png                  # IoU comparison plot
│   └── performance_heatmap.png                  # Performance heatmap
```

## 🚀 Advanced Features

### Multi-Attack Evaluation

```python
# Evaluate multiple attacks with different parameters
attacks = [
    FGSM(model, eps=2/255),
    FGSM(model, eps=4/255),
    FGSM(model, eps=8/255),
    PGD(model, eps=4/255, alpha=1/255, iters=10),
    PGD(model, eps=8/255, alpha=2/255, iters=20),
    RFGSM(model, eps=8/255, alpha=2/255, iters=10),
]

pipeline = SegmentationRobustnessPipeline(
    model=model,
    dataset=dataset,
    attacks=attacks,
    metrics=metrics,
    batch_size=4
)
```

### Multi-Model Evaluation

```python
# Evaluate multiple models
models = {
    "DeepLabV3": loader.load_model("torchvision", {"name": "deeplabv3_resnet50"}),
    "FCN": loader.load_model("torchvision", {"name": "fcn_resnet50"}),
    "UNet": loader.load_model("smp", {"architecture": "unet", "encoder_name": "resnet34"}),
}

for model_name, model in models.items():
    pipeline = SegmentationRobustnessPipeline(
        model=model,
        dataset=dataset,
        attacks=attacks,
        metrics=metrics,
        output_dir=f"./results/{model_name}"
    )
    results = pipeline.run()
```

### Custom Preprocessing

```python
# Custom image preprocessing
from torchvision import transforms

transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Custom mask preprocessing
def mask_transform(mask):
    # Your mask preprocessing
    return processed_mask

# Get preprocessing for VOC dataset
preprocess, target_preprocess = image_preprocessing.get_preprocessing_fn(
    [512, 512], dataset_name="voc"
)

dataset = VOCSegmentation(
    split="val",
    transform=preprocess,
    target_transform=target_preprocess
)
```

### Memory Optimization

```python
# Optimize for memory usage
import torch

# Set memory fraction
torch.cuda.set_per_process_memory_fraction(0.8)

# Use smaller batch size
pipeline = SegmentationRobustnessPipeline(
    model=model,
    dataset=dataset,
    attacks=attacks,
    metrics=metrics,
    batch_size=1  # Small batch size
)

# Clear cache periodically
torch.cuda.empty_cache()
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Device Mismatch Errors

```python
# ERROR: Attacks and model on different devices
# This happens when you create attacks before moving model to device

# ❌ WRONG - This will cause device mismatch errors
model = load_model()
attacks = [FGSM(model, eps=2/255)]  # Attack created before device setup
model = model.to("cuda")  # Model moved after attack creation

# ✅ CORRECT - Set device before creating attacks
model = load_model()
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)  # Move model to device first
attacks = [FGSM(model, eps=2/255)]  # Attacks will use same device as model

# Use same device in pipeline
pipeline = SegmentationRobustnessPipeline(
    model=model,
    dataset=dataset,
    attacks=attacks,
    metrics=metrics,
    device=device  # Must match model device
)
```

#### 2. CUDA Out of Memory

```python
# Solution: Reduce batch size
pipeline = SegmentationRobustnessPipeline(
    # ... other parameters ...
    batch_size=1  # Start with 1, increase if possible
)

# Or use CPU
device = "cpu"
model = model.to(device)
pipeline = SegmentationRobustnessPipeline(
    # ... other parameters ...
    device=device
)
```

#### 3. Model-Dataset Class Mismatch

```python
# Check class counts
print(f"Model classes: {model.num_classes}")
print(f"Dataset classes: {dataset.num_classes}")

# Ensure they match
assert model.num_classes == dataset.num_classes, "Class count mismatch!"
```

#### 4. Dataset Download Issues

```python
# Manual download
import os
os.makedirs("path/to/dataset/storage", exist_ok=True)

# Set download directory explicitly
preprocess, target_preprocess = image_preprocessing.get_preprocessing_fn(
    [512, 512], dataset_name="voc"
)

dataset = VOCSegmentation(
    split="val",
    root="path/to/dataset/storage",
    transform=preprocess,
    target_transform=target_preprocess,
    download=True
)
```

#### 5. Attack Convergence Issues

```python
# Adjust attack parameters
attack = PGD(
    model,
    eps=4/255,      # Reduce epsilon
    alpha=0.5/255,  # Reduce step size
    iters=5         # Reduce iterations
)
```

### Device Management Best Practices

```python
# 1. Set device early and consistently
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# 2. Move model to device immediately after loading
model = load_model()
model = model.to(device)

# 3. Create attacks after model is on device
attacks = [
    FGSM(model, eps=2/255),
    PGD(model, eps=4/255, iters=10)
]

# 4. Use same device in pipeline
pipeline = SegmentationRobustnessPipeline(
    model=model,
    dataset=dataset,
    attacks=attacks,
    metrics=metrics,
    device=device  # Must match model device
)

# 5. For multi-GPU setups
if torch.cuda.device_count() > 1:
    device = "cuda:0"  # Use specific GPU
    model = model.to(device)
    print(f"Using GPU: {device}")
```

### Debugging Tips

```python
# Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# Print tensor shapes
print(f"Input shape: {images.shape}")
print(f"Target shape: {labels.shape}")
print(f"Output shape: {outputs.shape}")

# Check device placement
print(f"Model device: {next(model.parameters()).device}")
print(f"Input device: {images.device}")

# Verify all components are on same device
assert next(model.parameters()).device == images.device, "Device mismatch!"
```

### Performance Optimization

```python
# Use mixed precision
from torch.cuda.amp import autocast

with autocast():
    outputs = model(images)

# Use DataParallel for multiple GPUs
model = torch.nn.DataParallel(model)

# Optimize for inference
model.eval()
with torch.no_grad():
    outputs = model(images)
```

---

**This user guide covers the essential aspects of the framework. For more advanced topics:**

- 📖 [Custom Components](custom_components.md) - Adding your own datasets and attacks
<!-- - 🧪 [Advanced Usage](advanced_usage.md) - Advanced features and optimization
- 📋 [API Reference](api_reference.md) - Complete API documentation  -->