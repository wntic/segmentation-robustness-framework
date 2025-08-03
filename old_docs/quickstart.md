# Quick Start Guide

Get up and running with the Segmentation Robustness Framework in under 5 minutes! This guide will walk you through your first evaluation.

## 🎯 What You'll Learn

By the end of this guide, you'll be able to:
    
   - ✅ Load a pre-trained segmentation model
   - ✅ Set up a dataset for evaluation
   - ✅ Configure and run adversarial attacks
   - ✅ Compute evaluation metrics
   - ✅ Save and visualize results

## 🚀 Your First Evaluation

### Step 1: Import the Framework

```python
# Core framework imports
from segmentation_robustness_framework.engine.pipeline import SegmentationRobustnessPipeline
from segmentation_robustness_framework.utils.metrics import MetricsCollection
from segmentation_robustness_framework.attacks import FGSM, PGD
from segmentation_robustness_framework.datasets import VOCSegmentation
from segmentation_robustness_framework.loaders.models.universal_loader import UniversalModelLoader
from segmentation_robustness_framework.adapters.torchvision_adapter import TorchvisionAdapter
from segmentation_robustness_framework.utils import image_preprocessing

# For direct model loading (not necessary)
import torch
import torchvision.models.segmentation as tv_segmentation
```

### Step 2: Load a Model

```python
# Option 1: Using Universal Model Loader (Recommended)
loader = UniversalModelLoader()
model = loader.load_model(
    model_type="torchvision",
    model_config={
        "name": "deeplabv3_resnet50",
        "num_classes": 21,
        "weights": "DEFAULT"
    }
)

# Option 2: Direct loading with adapter
base_model = tv_segmentation.deeplabv3_resnet50(
    num_classes=21, weights=tv_segmentation.DeepLabV3_ResNet50_Weights.COCO_WITH_VOC_LABELS_V1
)
model = TorchvisionAdapter(base_model)

print(f"✅ Model loaded with {model.num_classes} classes")
```

### Step 3: Set Up Dataset with Preprocessing

```python
# Define image size for preprocessing
image_shape = [512, 512]  # [height, width]

# Get preprocessing functions for VOC dataset
preprocess, target_preprocess = image_preprocessing.get_preprocessing_fn(
    image_shape, dataset_name="voc"
)

# Load VOC dataset with preprocessing
dataset = VOCSegmentation(
    split="val",
    root="./data",  # Dataset will be downloaded here
    transform=preprocess,           # Image preprocessing
    target_transform=target_preprocess,  # Mask preprocessing
    download=True  # Automatically download and extract to ./data/voc/
)

print(f"✅ Dataset loaded with {len(dataset)} samples")
```

### Step 4: Set Device and Move Model

```python
# Set device and move model to it (IMPORTANT: Do this before creating attacks!)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

print(f"✅ Model moved to {device}")
```

### Step 5: Configure Attacks

```python
# Create multiple attacks with different parameters
# Note: Attacks will automatically use the same device as the model
attacks = [
    FGSM(model, eps=2/255),      # Fast attack
    FGSM(model, eps=4/255),      # Stronger attack
    PGD(model, eps=4/255, alpha=1/255, iters=10),  # Iterative attack
]

print(f"✅ {len(attacks)} attacks configured")
```

### Step 6: Set Up Metrics

```python
# Create metrics collection
metrics_collection = MetricsCollection(num_classes=21)

# Define metrics to compute
metrics = [
    metrics_collection.mean_iou,           # Mean Intersection over Union
    metrics_collection.pixel_accuracy,     # Pixel accuracy
    metrics_collection.precision,          # Precision (macro)
    metrics_collection.recall,             # Recall (macro)
    metrics_collection.dice_score,         # Dice score (macro)
]

print(f"✅ {len(metrics)} metrics configured")
```

### Step 7: Create and Run Pipeline

```python
# Create evaluation pipeline
pipeline = SegmentationRobustnessPipeline(
    model=model,
    dataset=dataset,
    attacks=attacks,
    metrics=metrics,
    batch_size=4,           # Adjust based on your GPU memory
    device=device,          # Use the same device as the model
    output_dir="./results", # Results will be saved here
    auto_resize_masks=True  # Automatically resize masks to model output
)

print("🚀 Starting evaluation...")

# Run the evaluation
results = pipeline.run(save=True, show=True)

print("✅ Evaluation completed!")
```

### Step 8: View Results

```python
# Print summary
pipeline.print_summary()

# Get detailed results
summary = pipeline.get_summary()
print(f"Clean performance: {summary['clean_performance']}")
print(f"Attack evaluations: {len(summary['attack_performance'])}")

# Access specific results
clean_iou = results['clean']['mean_iou']
fgsm_iou = results['attack_FGSM_epsilon_0p008']['mean_iou']
print(f"Clean IoU: {clean_iou:.3f}")
print(f"FGSM IoU: {fgsm_iou:.3f}")
```

## 📊 Understanding the Results

### Output Structure

The pipeline generates several types of output:

```
results/
├── run_20231201_143022_abc12345/
│   ├── clean_detailed.json                      # Detailed clean evaluation results
│   ├── attack_FGSM_epsilon_0p008_detailed.json  # FGSM attack results
│   ├── attack_PGD_epsilon_0p016_detailed.json   # PGD attack results
│   ├── summary_results.json                     # Aggregated results
│   ├── comparison_table.csv                     # Results comparison
│   ├── mean_iou_comparison.png                  # Visualization plots
│   └── performance_heatmap.png                  # Performance heatmap
```

### Key Metrics

- **Mean IoU**: Intersection over Union - measures segmentation accuracy
- **Pixel Accuracy**: Percentage of correctly classified pixels
- **Precision**: True positives / (True positives + False positives)
- **Recall**: True positives / (True positives + False negatives)
- **Dice Score**: 2 × (Precision × Recall) / (Precision + Recall)

### Interpreting Results

```python
# Example results interpretation
clean_performance = results['clean']
attack_performance = results['attack_FGSM_epsilon_0p008']

# Calculate robustness degradation
degradation = (clean_performance['mean_iou'] - attack_performance['mean_iou']) / clean_performance['mean_iou'] * 100
print(f"Model robustness degradation: {degradation:.1f}%")
```

## 🔧 Customization Examples

### Different Model Types

```python
# SMP (Segmentation Models PyTorch) model
model = loader.load_model(
    model_type="smp",
    model_config={
        "architecture": "unet",
        "encoder_name": "resnet34",
        "encoder_weights": "imagenet",
        "classes": 21,
        "activation": None,
    }
)

# HuggingFace model
model = loader.load_model(
    model_type="huggingface",
    model_config={
        "model_name": "nvidia/segformer-b0-finetuned-ade-512-512",
        "num_labels": 150
    }
)

# IMPORTANT: Set device after loading model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
```

### Custom Models

For custom models, you need to create your own adapter. See the [User Guide](user_guide.md#custom-model-workflow) for detailed instructions.

```python
# Example: Create a custom adapter for your model
from segmentation_robustness_framework.adapters import CustomAdapter

class MyCustomAdapter(CustomAdapter):
    def logits(self, x: torch.Tensor) -> torch.Tensor:
        # Handle your model's specific output format
        return self.model(x)

# Use your custom adapter
base_model = MyCustomSegmentationModel(num_classes=21)
model = MyCustomAdapter(base_model, num_classes=21)

# Set device and create attacks
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
```

### Different Datasets

```python
# ADE20K dataset
from segmentation_robustness_framework.datasets import ADE20K

# Get preprocessing for ADE20K
preprocess, target_preprocess = image_preprocessing.get_preprocessing_fn(
    [512, 512], dataset_name="ade20k"
)

dataset = ADE20K(
    split="val",
    root="./data",
    transform=preprocess,
    target_transform=target_preprocess,
    download=True,  # Automatically download and extract to ./data/ade20k/ADEChallengeData2016/
)

# Stanford Background dataset
from segmentation_robustness_framework.datasets import StanfordBackground

# Get preprocessing for Stanford Background
preprocess, target_preprocess = image_preprocessing.get_preprocessing_fn(
    [512, 512], dataset_name="stanford_background"
)

dataset = StanfordBackground(
    root="./data",
    transform=preprocess,
    target_transform=target_preprocess,
    download=True  # Automatically download and extract to ./data/stanford_background/stanford_background/
)
```

### Dataset Path Logic

The framework handles dataset paths differently based on the `download` parameter:

**When `download=True` (default):**
- Creates nested directory structure for organization
- VOC: `root/voc/VOCdevkit/VOC2012/`
- ADE20K: `root/ade20k/ADEChallengeData2016/`
- Stanford Background: `root/stanford_background/stanford_background/`

**When `download=False`:**
- Uses the exact path specified by `root`
- Useful when you have pre-downloaded datasets
- Example: `root="/path/to/existing/voc/dataset"`

```python
# Using pre-downloaded dataset
dataset = VOCSegmentation(
    split="val",
    root="/path/to/existing/voc/dataset",  # Exact path to dataset
    transform=preprocess,
    target_transform=target_preprocess,
    download=False  # Use existing dataset at exact path
)
```

### Different Attacks

```python
# RFGSM (Random + Fast Gradient Sign Method)
from segmentation_robustness_framework.attacks import RFGSM
rfgsm_attack = RFGSM(model, eps=8/255, alpha=2/255, iters=10)

# TPGD (Theoretically Principled PGD)
from segmentation_robustness_framework.attacks import TPGD
tpgd_attack = TPGD(model, eps=8/255, alpha=2/255, iters=10)
```

### Custom Metrics

```python
# Create custom metric
def custom_metric(targets, predictions):
    """Custom evaluation metric."""
    # Your metric implementation
    return score

# Use with pipeline
metrics = [metrics_collection.mean_iou, custom_metric]
```

## 🎛️ Advanced Configuration

### Batch Processing

```python
# Process in smaller batches for memory efficiency
pipeline = SegmentationRobustnessPipeline(
    model=model,
    dataset=dataset,
    attacks=attacks,
    metrics=metrics,
    batch_size=2,  # Smaller batch size
    device=device   # Use the same device as model
)
```

### Multiple Output Formats

```python
# Save results in multiple formats
pipeline = SegmentationRobustnessPipeline(
    # ... other parameters ...
    output_formats=["json", "csv"],  # Save both JSON and CSV
)
```

### Custom Device Management

```python
# Use specific GPU
import torch
torch.cuda.set_device(0)  # Use GPU 0

device = "cuda:0"  # Specify GPU
model = model.to(device)  # Move model to GPU

pipeline = SegmentationRobustnessPipeline(
    # ... other parameters ...
    device=device  # Use the same device as model
)
```

## 🐛 Common Issues and Solutions

### Memory Issues

```python
# Reduce batch size
batch_size = 1  # Start with 1, increase if memory allows

# Use CPU if GPU memory is insufficient
device = "cpu"
model = model.to(device)

# Clear GPU cache
torch.cuda.empty_cache()
```

### Device Mismatch Issues

```python
# Ensure model and attacks use the same device
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Create attacks AFTER moving model to device
attacks = [FGSM(model, eps=2/255), PGD(model, eps=4/255)]

# Use same device in pipeline
pipeline = SegmentationRobustnessPipeline(
    model=model,
    dataset=dataset,
    attacks=attacks,
    metrics=metrics,
    device=device  # Must match model device
)
```

### Preprocessing Issues

```python
# Ensure preprocessing is set up correctly
preprocess, target_preprocess = image_preprocessing.get_preprocessing_fn(
    [512, 512], dataset_name="voc"  # Use correct dataset name
)

# Check dataset compatibility
print(f"Dataset classes: {dataset.num_classes}")
print(f"Model classes: {model.num_classes}")
assert dataset.num_classes == model.num_classes, "Class count mismatch!"
```

## 🚀 Next Steps

Now that you've completed your first evaluation:

1. **Explore Examples**: Check the [Practical Example](practical_example.md) for more complex scenarios
2. **Custom Components**: Learn to add your own [datasets](custom_components.md#custom-datasets) and [attacks](custom_components.md#custom-attacks)
<!-- 3. **Advanced Usage**: Discover [advanced features](advanced_usage.md) and optimization techniques -->

---

**🎉 Congratulations!** You've successfully evaluated your first segmentation model for robustness against adversarial attacks. The framework is now ready for your research and development needs! 