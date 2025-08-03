# Basic Examples

This page provides basic examples of using the Segmentation Robustness Framework. 🚀

## ⚡ Quick Evaluation

Evaluate a model with a single attack:

```python
from segmentation_robustness_framework.pipeline import SegmentationRobustnessPipeline
from segmentation_robustness_framework.loaders import UniversalModelLoader, DatasetLoader, AttackLoader
from segmentation_robustness_framework.metrics import MetricsCollection
from segmentation_robustness_framework.attacks import FGSM

# Initialize components
model_loader = UniversalModelLoader()
model = model_loader.load_model(
    model_type="torchvision",
    model_config={"name": "deeplabv3_resnet50", "num_classes": 21}
)

dataset_loader = DatasetLoader({
    "name": "voc", 
    "split": "val", 
    "root": "./data",
    "image_shape": [512, 512],
    "download": True
})
dataset = dataset_loader.load_dataset()

metrics = MetricsCollection(num_classes=21, ignore_index=255)

# Create pipeline
pipeline = SegmentationRobustnessPipeline(
    model=model,
    dataset=dataset,
    attacks=[FGSM(model, eps=0.02)],
    metrics=[metrics.mean_iou, metrics.pixel_accuracy],
    batch_size=4,
    device="cuda"
)

# Run evaluation
results = pipeline.run()
print(f"Clean IoU: {results['clean']['mean_iou']:.3f}")
print(f"FGSM IoU: {results['attack_fgsm']['mean_iou']:.3f}")
```

## ⚙️ Configuration-Based Evaluation

Use a YAML configuration file:

```yaml
# config.yaml
pipeline:
  device: cuda
  batch_size: 4
  output_dir: results
  auto_resize_masks: true
  output_formats: ["json"]

model:
  type: torchvision
  config:
    name: deeplabv3_resnet50
    num_classes: 21

dataset:
  name: voc
  split: val
  root: ./data
  image_shape: [512, 512]
  download: true

attacks:
  - name: fgsm
    eps: 0.02
  - name: pgd
    eps: 0.02
    alpha: 0.02
    iters: 10
    targeted: false

metrics:
  ignore_index: 255
  selected_metrics:
    - mean_iou
    - pixel_accuracy
    - precision
    - recall
```

Run with CLI:

```bash
python -m segmentation_robustness_framework.cli.main run config.yaml
```

## 🔀 Multiple Attacks Comparison

Compare different attack strengths:

```python
from segmentation_robustness_framework.attacks import FGSM, PGD

# Create attacks with different parameters
attacks = [
    FGSM(model, eps=0.02),
    FGSM(model, eps=0.05),
    FGSM(model, eps=0.1),
    PGD(model, eps=0.02, alpha=0.02, iters=10, targeted=False),
    PGD(model, eps=0.05, alpha=0.02, iters=20, targeted=False)
]

pipeline = SegmentationRobustnessPipeline(
    model=model,
    dataset=dataset,
    attacks=attacks,
    metrics=[metrics.mean_iou],
    batch_size=4,
    device="cuda"
)

results = pipeline.run()

# Print results
for attack_name, metrics in results.items():
    if attack_name.startswith('attack_'):
        print(f"{attack_name}: {metrics['mean_iou']:.3f}")
```

## Custom Model Integration

Integrate your own model:

```python
import torch.nn as nn
from segmentation_robustness_framework.adapters import CustomAdapter

# Your custom model
class MySegmentationModel(nn.Module):
    def __init__(self, num_classes=21):
        super().__init__()
        # Your model architecture here
        pass
    
    def forward(self, x):
        # Your forward pass
        return logits

# Create adapter
class MyModelAdapter(CustomAdapter):
    def __init__(self, model: MySegmentationModel):
        super().__init__(model, num_classes=21)
    
    def logits(self, x):
        return self.model(x)
    
    def predictions(self, x):
        logits = self.logits(x)
        return torch.argmax(logits, dim=1)

# Use in pipeline
model = MySegmentationModel()
adapter = MyModelAdapter(model)
pipeline = SegmentationRobustnessPipeline(
    model=adapter,
    dataset=dataset,
    attacks=[FGSM(adapter, eps=0.02)],
    metrics=[metrics.mean_iou],
    batch_size=4,
    device="cuda"
)
```

## Custom Metric

Create your own metric:

```python
from segmentation_robustness_framework.metrics import register_custom_metric

@register_custom_metric("custom_dice")
def custom_dice_score(targets, predictions):
    """Custom Dice score implementation."""
    # Your implementation here
    return dice_score

# Use in pipeline
pipeline = SegmentationRobustnessPipeline(
    model=model,
    dataset=dataset,
    attacks=[FGSM(model, eps=0.02)],
    metrics=[custom_dice_score],
    batch_size=4,
    device="cuda"
)
```

## Dataset Preprocessing

Customize dataset preprocessing:

```python
from torchvision import transforms

# Dataset configuration with automatic preprocessing
dataset_config = {
    "name": "voc",
    "split": "val",
    "root": "./data",
    "image_shape": [512, 512],
    "download": True
}

dataset_loader = DatasetLoader(dataset_config)
dataset = dataset_loader.load_dataset()
```

## Batch Processing

Process data in batches with custom logic:

```python
# Note: The framework handles batch processing automatically
# Custom preprocessing can be applied through dataset transforms

pipeline = SegmentationRobustnessPipeline(
    model=model,
    dataset=dataset,
    attacks=[FGSM(model, eps=0.02)],
    metrics=[metrics.mean_iou],
    batch_size=4,
    device="cuda"
)
```

## Error Handling

Handle errors gracefully:

```python
try:
    results = pipeline.run()
except Exception as e:
    print(f"Evaluation failed: {e}")
    # Handle error appropriately
```

## Next Steps

- [Advanced Examples](advanced_examples.md) - More complex use cases
- [User Guide](../user_guide.md) - Comprehensive usage guide
- [API Reference](../api_reference/index.md) - Complete API documentation
