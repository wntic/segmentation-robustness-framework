# Configuration Guide

This guide covers how to write configuration files for the Segmentation Robustness Framework.

## 📋 Overview

The framework uses a YAML/JSON-based configuration system that allows you to define complete evaluation pipelines without writing code. The configuration system handles model loading, dataset setup, attack configuration, and pipeline parameters automatically.

## 🏗️ Configuration Structure

### 📝 Basic Configuration Layout

```yaml
# Model configuration
model:
  type: "torchvision"
  config:
    name: "deeplabv3_resnet50"
    num_classes: 21
  weights_path: null
  weight_type: "full"
  adapter: null

# Dataset configuration
dataset:
  name: "voc"
  root: "./data/VOCdevkit/VOC2012"
  split: "val"
  image_shape: [512, 512]
  download: true

# Attack configurations
attacks:
  - name: "fgsm"
    eps: 0.02
  - name: "pgd"
    eps: 0.02
    alpha: 0.01
    iters: 10
    targeted: false

# Pipeline configuration
pipeline:
  batch_size: 8
  device: "cuda"
  output_dir: "./runs"
  auto_resize_masks: true
  output_formats: ["json", "csv"]
  metric_precision: 4
  num_workers: 0
  pin_memory: false
  persistent_workers: false

# Metrics configuration
metrics:
  ignore_index: 255
  selected_metrics:
    - "mean_iou"
    - "pixel_accuracy"
    - {"name": "dice_score", "average": "micro"}
    - "custom_metric_name"
```

## 🤖 Model Configuration

### 🎯 Supported Model Types

The framework supports four model types:

1. **torchvision** - Pre-trained models from torchvision
2. **smp** - Models from Segmentation Models PyTorch
3. **huggingface** - Models from HuggingFace Transformers
4. **custom** - Your own custom models

### 🖼️ Torchvision Models

```yaml
model:
  type: "torchvision"
  config:
    name: "deeplabv3_resnet50"  # Model name
    num_classes: 21              # Number of segmentation classes
  weights_path: null             # Optional: path to custom weights
  weight_type: "full"            # "full" or "encoder"
  adapter: null                  # Optional: custom adapter class
```

**📋 Available Torchvision Models:**

- `deeplabv3_resnet50`
- `deeplabv3_resnet101`
- `fcn_resnet50`
- `fcn_resnet101`
- `lraspp_mobilenet_v3_large`

> **Note**: For a complete list of available models and their specifications, refer to the [PyTorch Torchvision documentation](https://pytorch.org/vision/stable/models.html#semantic-segmentation).

### 🏗️ SMP Models

```yaml
model:
  type: "smp"
  config:
    architecture: "Unet"           # Model architecture
    encoder_name: "resnet34"       # Encoder name
    encoder_weights: "imagenet"    # Pre-trained weights
    in_channels: 3                 # Input channels
    classes: 21                    # Number of classes
  weights_path: null
  weight_type: "full"
  adapter: null
```

**🏛️ Available SMP Architectures:**

- `Unet`
- `UnetPlusPlus`
- `MAnet`
- `Linknet`
- `FPN`
- `PSPNet`
- `PAN`
- `DeepLabV3`
- `DeepLabV3Plus`

**🔧 Available SMP Encoders:**

- `resnet18`, `resnet34`, `resnet50`, `resnet101`, `resnet152`
- `densenet121`, `densenet169`, `densenet201`
- `efficientnet-b0` through `efficientnet-b7`
- `mobilenet_v2`
- `xception`

> **Note**: For a complete list of available architectures, encoders, and their specifications, refer to the [Segmentation Models PyTorch documentation](https://github.com/qubvel/segmentation_models.pytorch).

### 🤗 HuggingFace Models

```yaml
model:
  type: "huggingface"
  config:
    model_name: "nvidia/segformer-b0-finetuned-ade-512-512"
    revision: "main"              # Optional: git revision
    trust_remote_code: false      # Optional: trust remote code
  weights_path: null
  weight_type: "full"
  adapter: null
```

**📚 Available HuggingFace Models:**

- `nvidia/segformer-b0-finetuned-ade-512-512`
- `nvidia/segformer-b1-finetuned-ade-512-512`
- `nvidia/segformer-b2-finetuned-ade-512-512`
- `nvidia/segformer-b3-finetuned-ade-512-512`
- `nvidia/segformer-b4-finetuned-ade-512-512`
- `nvidia/segformer-b5-finetuned-ade-512-512`

> **Note**: For a complete list of available models and their specifications, refer to the [HuggingFace Transformers documentation](https://huggingface.co/models?pipeline_tag=image-segmentation) and the [HuggingFace Hub](https://huggingface.co/models).

### 🔧 Custom Models

```yaml
model:
  type: "custom"
  config:
    model_class: "path.to.CustomModel"
    model_args:
      num_classes: 21
      pretrained: true
  weights_path: "./path/to/weights.pth"
  weight_type: "full"
  adapter: "path.to.CustomAdapter"
```

### ⚙️ Model Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `type` | str | Required | Model type: "torchvision", "smp", "huggingface", "custom" |
| `config` | dict | Required | Model-specific configuration |
| `weights_path` | str | null | Path to custom model weights |
| `weight_type` | str | "full" | "full" or "encoder" for SMP models |
| `adapter` | str | null | Custom adapter class path |

## 📊 Dataset Configuration

### 🗂️ Supported Datasets

The framework supports four built-in datasets:

1. **voc** - PASCAL VOC 2012
2. **ade20k** - MIT Scene Parsing
3. **cityscapes** - Urban Scene Understanding
4. **stanford_background** - Stanford Background Dataset

### 🏷️ VOC Dataset

```yaml
dataset:
  name: "voc"
  root: "./data/VOCdevkit/VOC2012"  # Dataset root directory
  split: "val"                       # "train", "val", "trainval"
  image_shape: [512, 512]           # [height, width]
  download: true                     # Auto-download if not found
```

### 🏙️ ADE20K Dataset

```yaml
dataset:
  name: "ade20k"
  root: "data/"
  split: "val"
  image_shape: [512, 512]
  download: true
```

### 🚗 Cityscapes Dataset

```yaml
dataset:
  name: "cityscapes"
  root: "data/cityscapes"
  split: "val"
  mode: "fine"
  target_type: "semantic"
  image_shape: [512, 512]
```

### 🌳 Stanford Background Dataset

```yaml
dataset:
  name: "stanford_background"
  root: "data/"
  image_shape: [512, 512]
  download: true
```

### ⚙️ Dataset Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `name` | str | Required | Dataset name: "voc", "ade20k", "cityscapes", "stanford_background" |
| `root` | str | null | Dataset root directory (uses cache if null) |
| `split` | str | "val" | Dataset split: "train", "val", "trainval" |
| `image_shape` | list | Required | Target image size [height, width] |
| `download` | bool | false | Auto-download dataset if not found |

## ⚔️ Attack Configuration

### 🎯 Supported Attacks

The framework supports multiple adversarial attacks:

1. **fgsm** - Fast Gradient Sign Method
2. **pgd** - Projected Gradient Descent
3. **rfgsm** - R-FGSM with random start
4. **tpgd** - Two-Phase Gradient Descent

### ⚡ FGSM Attack

```yaml
attacks:
  - name: "fgsm"
    eps: 0.02                    # Maximum perturbation magnitude
```

### 🔄 PGD Attack

```yaml
attacks:
  - name: "pgd"
    eps: 0.02                    # Maximum perturbation magnitude
    alpha: 0.01                  # Step size
    iters: 10                    # Number of iterations
    targeted: false               # Targeted or untargeted attack
```

### 🎲 RFGSM Attack

```yaml
attacks:
  - name: "rfgsm"
    eps: 0.02                    # Maximum perturbation magnitude
    alpha: 0.01                  # Step size
    iters: 10                    # Number of iterations
```

### ⚖️ TPGD Attack

```yaml
attacks:
  - name: "tpgd"
    eps: 0.02                    # Maximum perturbation magnitude
    alpha: 0.01                  # Step size
    iters: 10                    # Number of iterations
    targeted: false               # Targeted or untargeted attack
```

### 🔀 Multiple Attacks

```yaml
attacks:
  - name: "fgsm"
    eps: 0.02
  - name: "fgsm"
    eps: 0.05
  - name: "pgd"
    eps: 0.02
    alpha: 0.01
    iters: 10
    targeted: false
  - name: "pgd"
    eps: 0.05
    alpha: 0.02
    iters: 20
    targeted: false
```

### ⚙️ Attack Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `name` | str | Required | Attack name: "fgsm", "pgd", "rfgsm", "tpgd" |
| `eps` | float | Required | Maximum perturbation magnitude |
| `alpha` | float | 0.01 | Step size (PGD, RFGSM, TPGD) |
| `iters` | int | 10 | Number of iterations (PGD, RFGSM, TPGD) |
| `targeted` | bool | false | Targeted attack (PGD, TPGD) |

## 🔧 Pipeline Configuration

### ⚙️ Basic Pipeline Settings

```yaml
pipeline:
  batch_size: 8                    # Batch size for evaluation
  device: "cuda"                   # Device: "cuda", "cpu", "mps"
  output_dir: "./runs"             # Output directory
  auto_resize_masks: true          # Auto-resize masks to model output
  output_formats: ["json", "csv"]  # Output formats
  metric_precision: 4              # Decimal places for metrics
  num_workers: 0                   # DataLoader workers
  pin_memory: false                # Pin memory for GPU
  persistent_workers: false         # Keep workers alive
```

### 📋 Pipeline Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `batch_size` | int | 8 | Batch size for evaluation |
| `device` | str | "cpu" | Device: "cuda", "cpu", "mps" |
| `output_dir` | str | "./runs" | Output directory for results |
| `auto_resize_masks` | bool | true | Auto-resize masks to model output |
| `output_formats` | list | ["json"] | Output formats: ["json", "csv"] |
| `metric_precision` | int | 4 | Decimal places for metric values |
| `num_workers` | int | 0 | Number of DataLoader workers |
| `pin_memory` | bool | false | Pin memory for faster GPU transfer |
| `persistent_workers` | bool | false | Keep workers alive between epochs |

## 📈 Metrics Configuration

### 📊 Basic Metrics Setup

```yaml
metrics:
  ignore_index: 255                # Ignore index for evaluation
  selected_metrics:
    - "mean_iou"                   # Mean Intersection over Union
    - "pixel_accuracy"             # Pixel accuracy
    - {"name": "dice_score", "average": "micro"}  # Dice score with micro averaging
    - "custom_metric_name"         # Custom metric
```

### 📋 Available Metrics

**🔧 Built-in Metrics:**
- `mean_iou` - Mean Intersection over Union
- `pixel_accuracy` - Overall pixel accuracy
- `precision` - Precision (per-class or averaged)
- `recall` - Recall (per-class or averaged)
- `dice_score` - Dice coefficient (F1-score)

**📊 Averaging Options:**
- `macro` - Macro averaging (default)
- `micro` - Micro averaging

### 📝 Metric Configuration Examples

```yaml
# All default metrics with macro averaging
metrics:
  ignore_index: 255
  selected_metrics: null  # Use all default metrics

# Specific metrics with different averaging
metrics:
  ignore_index: 255
  selected_metrics:
    - "mean_iou"
    - "pixel_accuracy"
    - {"name": "precision", "average": "macro"}
    - {"name": "precision", "average": "micro"}
    - {"name": "recall", "average": "macro"}
    - {"name": "recall", "average": "micro"}
    - {"name": "dice_score", "average": "macro"}
    - {"name": "dice_score", "average": "micro"}

# Custom metrics only
metrics:
  ignore_index: 255
  selected_metrics:
    - "custom_iou_metric"
    - "custom_accuracy_metric"
```

### ⚙️ Metrics Configuration Options

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `ignore_index` | int | 255 | Index to ignore in evaluation |
| `selected_metrics` | list | null | List of metrics to compute (null = all) |
| `include_pixel_accuracy` | bool | true | Include pixel accuracy in default metrics |

## 📝 Complete Configuration Examples

### 🏷️ Basic VOC Evaluation

```yaml
model:
  type: "torchvision"
  config:
    name: "deeplabv3_resnet50"
    num_classes: 21

dataset:
  name: "voc"
  root: "data/VOCdevkit/VOC2012"
  split: "val"
  image_shape: [512, 512]
  download: false

attacks:
  - name: "fgsm"
    eps: 0.02
  - name: "pgd"
    eps: 0.02
    alpha: 0.01
    iters: 10
    targeted: false

pipeline:
  batch_size: 4
  device: "cuda"
  output_dir: "./runs/voc_evaluation"
  auto_resize_masks: true
  output_formats: ["json", "csv"]

metrics:
  ignore_index: 255
  selected_metrics:
    - "mean_iou"
    - "pixel_accuracy"
    - {"name": "dice_score", "average": "micro"}
```

### 🏙️ Advanced ADE20K Evaluation

```yaml
model:
  type: "smp"
  config:
    architecture: "UnetPlusPlus"
    encoder_name: "efficientnet-b0"
    encoder_weights: "imagenet"
    in_channels: 3
    classes: 150

dataset:
  name: "ade20k"
  root: "data/ADEChallengeData2016"
  split: "val"
  image_shape: [512, 512]
  download: false

attacks:
  - name: "fgsm"
    eps: 0.02
  - name: "fgsm"
    eps: 0.05
  - name: "pgd"
    eps: 0.02
    alpha: 0.01
    iters: 10
    targeted: false
  - name: "pgd"
    eps: 0.05
    alpha: 0.02
    iters: 20
    targeted: false
  - name: "rfgsm"
    eps: 0.02
    alpha: 0.01
    iters: 10

pipeline:
  batch_size: 2
  device: "cuda"
  output_dir: "./runs/ade20k_evaluation"
  auto_resize_masks: true
  output_formats: ["json", "csv"]
  metric_precision: 4
  num_workers: 2
  pin_memory: true
  persistent_workers: true

metrics:
  ignore_index: 255
  selected_metrics:
    - "mean_iou"
    - "pixel_accuracy"
    - {"name": "precision", "average": "macro"}
    - {"name": "precision", "average": "micro"}
    - {"name": "recall", "average": "macro"}
    - {"name": "recall", "average": "micro"}
    - {"name": "dice_score", "average": "macro"}
    - {"name": "dice_score", "average": "micro"}
```

### 🤗 HuggingFace Model Evaluation

```yaml
model:
  type: "huggingface"
  config:
    model_name: "nvidia/segformer-b0-finetuned-ade-512-512"
    revision: "main"
    trust_remote_code: false

dataset:
  name: "ade20k"
  root: "data/"
  split: "val"
  image_shape: [512, 512]
  download: true

attacks:
  - name: "fgsm"
    eps: 0.02
  - name: "pgd"
    eps: 0.02
    alpha: 0.01
    iters: 10
    targeted: false

pipeline:
  batch_size: 1  # Smaller batch size for large models
  device: "cuda"
  output_dir: "./runs/segformer_evaluation"
  auto_resize_masks: true
  output_formats: ["json"]

metrics:
  ignore_index: 255
  selected_metrics:
    - "mean_iou"
    - "pixel_accuracy"
```

## 🔄 Using Configuration Files

### 📄 Loading from YAML

```python
from segmentation_robustness_framework.pipeline.config import PipelineConfig

# Load configuration from YAML file
config = PipelineConfig.from_yaml("config.yaml")

# Create and run pipeline
pipeline = config.create_pipeline()
results = pipeline.run(save=True, show=False)
```

### 📋 Loading from JSON

```python
from segmentation_robustness_framework.pipeline.config import PipelineConfig

# Load configuration from JSON file
config = PipelineConfig.from_json("config.json")

# Create and run pipeline
pipeline = config.create_pipeline()
results = pipeline.run(save=True, show=False)
```

### 📚 Loading from Dictionary

```python
from segmentation_robustness_framework.pipeline.config import PipelineConfig

# Configuration as dictionary
config_dict = {
    "model": {
        "type": "torchvision",
        "config": {"name": "deeplabv3_resnet50", "num_classes": 21}
    },
    "dataset": {
        "name": "voc",
        "root": "data/",
        "split": "val",
        "image_shape": [512, 512],
        "download": True
    },
    "attacks": [{"name": "fgsm", "eps": 0.02}],
    "pipeline": {"batch_size": 4, "device": "cuda"}
}

# Create configuration from dictionary
config = PipelineConfig.from_dict(config_dict)

# Create and run pipeline
pipeline = config.create_pipeline()
results = pipeline.run()
```

### 💻 Command Line Usage

```bash
# Run evaluation from configuration file
python -m segmentation_robustness_framework.cli.main run config.yaml

# Run with custom output directory
python -m segmentation_robustness_framework.cli.main run config.yaml --output-dir ./custom_output

# Run with specific device
python -m segmentation_robustness_framework.cli.main run config.yaml --device cpu
```

## ⚠️ Common Configuration Issues

### 🔗 Model-Dataset Compatibility

**Issue**: Model and dataset have different numbers of classes

**Solution**: Ensure `num_classes` in model config matches dataset
```yaml
# VOC has 21 classes
model:
  type: "torchvision"
  config:
    name: "deeplabv3_resnet50"
    num_classes: 21  # Must match VOC

dataset:
  name: "voc"  # Has 21 classes
```

### 💾 Memory Issues

**Issue**: CUDA out of memory

**Solution**: Reduce batch size or use CPU
```yaml
pipeline:
  batch_size: 1  # Reduce from 8
  device: "cpu"   # Or use CPU instead of CUDA
```

### ⚔️ Attack Parameter Issues

**Issue**: Attack parameters are invalid

**Solution**: Check parameter ranges and relationships
```yaml
attacks:
  - name: "pgd"
    eps: 0.02      # Must be positive
    alpha: 0.01    # Must be <= eps
    iters: 10      # Must be positive
```

### 📁 Dataset Path Issues

**Issue**: Dataset not found

**Solution**: Enable download or specify correct path
```yaml
dataset:
  name: "voc"
  root: "./data/VOCdevkit/VOC2012"  # Correct path
  download: true                      # Auto-download if not found
```

## 🎯 Best Practices

### 📋 Configuration Organization

1. **Use descriptive names** for output directories
2. **Group related configurations** logically
3. **Include comments** for complex configurations
4. **Use consistent formatting** and indentation

### ⚡ Performance Optimization

1. **Start with small batch sizes** and increase gradually
2. **Use GPU** when available for faster evaluation
3. **Enable pin_memory** for GPU evaluations
4. **Use multiple workers** for data loading (but be careful with memory)

### ✅ Validation

1. **Test configurations** with small datasets first
2. **Verify model-dataset compatibility** before running
3. **Check memory requirements** for your hardware
4. **Validate attack parameters** are within reasonable ranges

### 🔧 Maintenance

1. **Version control** your configuration files
2. **Document custom configurations** with comments
3. **Keep backup configurations** for different scenarios
4. **Update configurations** when framework versions change

## 📚 Configuration Templates

### 🎯 Minimal Configuration

```yaml
model:
  type: "torchvision"
  config:
    name: "deeplabv3_resnet50"
    num_classes: 21

dataset:
  name: "voc"
  split: "val"
  image_shape: [512, 512]
  download: true

attacks:
  - name: "fgsm"
    eps: 0.02

pipeline:
  batch_size: 4
  device: "cuda"

metrics:
  ignore_index: 255
```

### 🏭 Production Configuration

```yaml
model:
  type: "smp"
  config:
    architecture: "UnetPlusPlus"
    encoder_name: "efficientnet-b0"
    encoder_weights: "imagenet"
    in_channels: 3
    classes: 21

dataset:
  name: "voc"
  root: "./data/VOCdevkit/VOC2012"
  split: "val"
  image_shape: [512, 512]
  download: false

attacks:
  - name: "fgsm"
    eps: 0.02
  - name: "fgsm"
    eps: 0.05
  - name: "pgd"
    eps: 0.02
    alpha: 0.01
    iters: 10
    targeted: false
  - name: "pgd"
    eps: 0.05
    alpha: 0.02
    iters: 20
    targeted: false

pipeline:
  batch_size: 8
  device: "cuda"
  output_dir: "./runs/production_evaluation"
  auto_resize_masks: true
  output_formats: ["json", "csv"]
  metric_precision: 4
  num_workers: 4
  pin_memory: true
  persistent_workers: true

metrics:
  ignore_index: 255
  selected_metrics:
    - "mean_iou"
    - "pixel_accuracy"
    - {"name": "precision", "average": "macro"}
    - {"name": "recall", "average": "macro"}
    - {"name": "dice_score", "average": "macro"}
```

Happy configuring! ⚙️✨
