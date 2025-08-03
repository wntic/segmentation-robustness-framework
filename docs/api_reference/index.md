# API Reference

This page provides comprehensive API documentation for the Segmentation Robustness Framework, automatically generated from the code's docstrings. 🔧

## 📚 Navigation

- 🔧 [Pipeline API](pipeline.md) - Core pipeline components and configuration
- 🤖 [Model Loaders API](model_loaders.md) - Model loading and adapter system
- 🔄 [Adapters API](adapters.md) - Model interface standardization
- ⚔️ [Attacks API](attacks.md) - Adversarial attack implementations
- 📊 [Datasets API](datasets.md) - Dataset loading and preprocessing
- 📈 [Metrics API](metrics.md) - Evaluation metrics and scoring
- 💻 [CLI API](cli.md) - Command-line interface

## 🚀 Quick Overview

The Segmentation Robustness Framework provides a comprehensive API for evaluating the robustness of semantic segmentation models against adversarial attacks.

### 🔧 Core Components

#### 🔄 Pipeline
The main orchestration component that coordinates the entire evaluation process:

```python
from segmentation_robustness_framework.pipeline import SegmentationRobustnessPipeline

pipeline = SegmentationRobustnessPipeline(
    model=model,
    dataset=dataset,
    attacks=[FGSM(model, eps=0.1)],
    metrics=[metrics.mean_iou],
    batch_size=4,
    device="cuda"
)

results = pipeline.run()
```

#### 🤖 Model Loaders
Specialized loaders for different model types:

```python
from segmentation_robustness_framework.loaders import UniversalModelLoader

# Load any supported model type
model = UniversalModelLoader().load_model(
    model_type="torchvision",
    model_config={"name": "deeplabv3_resnet50"}
)
```

#### 🔄 Adapters
Standardized interfaces for different model architectures:

```python
from segmentation_robustness_framework.adapters import TorchvisionAdapter

# Wrap model with adapter
adapter = TorchvisionAdapter(model)
logits = adapter.logits(x)
predictions = adapter.predictions(x)
```

#### ⚔️ Attacks
Adversarial attack implementations:

```python
from segmentation_robustness_framework.attacks import FGSM, PGD

# Create attacks
fgsm_attack = FGSM(model, eps=0.1)
pgd_attack = PGD(model, eps=0.1, alpha=0.01, iters=10)

# Apply attacks
adversarial_x = fgsm_attack.apply(x, y)
```

#### 📊 Datasets
Dataset loading and preprocessing:

```python
from segmentation_robustness_framework.loaders import DatasetLoader

# Load dataset
dataset_loader = DatasetLoader({"name": "voc", "split": "val"})
dataset = dataset_loader.load_dataset()
```

#### 📈 Metrics
Evaluation metrics for segmentation:

```python
from segmentation_robustness_framework.metrics import MetricsCollection

# Create metrics collection
metrics = MetricsCollection(num_classes=21, ignore_index=255)

# Get metric functions
metric_functions = [
    metrics.mean_iou,
    metrics.pixel_accuracy,
    metrics.precision,
    metrics.recall
]
```

#### CLI
Command-line interface for easy usage:

```bash
# Run evaluation from configuration
python -m segmentation_robustness_framework.cli.main run config.yaml

# List available components
python -m segmentation_robustness_framework.cli.main list

# Run tests
python -m segmentation_robustness_framework.cli.main test
```

## Getting Started

1. **Choose your model**: See [Model Loaders API](model_loaders.md) for supported models
2. **Select your dataset**: See [Datasets API](datasets.md) for available datasets
3. **Configure attacks**: See [Attacks API](attacks.md) for attack options
4. **Define metrics**: See [Metrics API](metrics.md) for evaluation metrics
5. **Run evaluation**: See [Pipeline API](pipeline.md) for execution
6. **Use CLI**: See [CLI API](cli.md) for command-line usage

## Configuration

All components can be configured using YAML configuration files:

```yaml

model:
  type: "torchvision"
  config:
    name: "deeplabv3_resnet50"
    num_classes: 21
    weights: "COCO_WITH_VOC_LABELS_V1"  # Use pre-trained weights

dataset:
  name: "voc"
  split: "val"
  image_shape: [256, 256]
  download: true

attacks:
  - name: "fgsm"
    eps: 0.02

pipeline:
  batch_size: 4
  device: "cuda"
  output_dir: "./runs/"
  auto_resize_masks: true
  output_formats: ["json", "csv"]

metrics:
  ignore_index: 255
  selected_metrics:
    - "mean_iou"  # Will use macro averaging by default
    - "precision"
    - "recall"
    - "pixel_accuracy"
    - {"name": "dice_score", "average": "micro"}  # Specify averaging strategy
    - "custom_dice_score"
    - "weighted_iou"
    - "f1_score"
  # include_pixel_accuracy: true  # Only used when selected_metrics is not specified
```

## Extensibility

The framework is designed to be easily extensible:

- **Custom Models**: See [Model Loaders API](model_loaders.md)
- **Custom Attacks**: See [Attacks API](attacks.md)
- **Custom Metrics**: See [Metrics API](metrics.md)
- **Custom Datasets**: See [Datasets API](datasets.md)

## Best Practices

1. **Use Configuration Files**: Define experiments in YAML for reproducibility
2. **Start Simple**: Begin with basic attacks and metrics
3. **Validate Results**: Always compare clean vs adversarial performance
4. **Monitor Resources**: Use appropriate batch sizes for your hardware
5. **Save Results**: Enable result saving for analysis

## Support

For questions and issues:

- Check the [User Guide](../user_guide.md) for detailed usage instructions
- Review the [Core Concepts](../core_concepts.md) for framework understanding
- Use the CLI `list` command to see available components
- Run tests with the CLI `test` command to verify installation
