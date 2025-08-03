# Metrics API

This page documents the evaluation metrics components of the Segmentation Robustness Framework.

## Metrics Classes

::: segmentation_robustness_framework.metrics.base_metrics
    options:
        show_signature_annotations: true

::: segmentation_robustness_framework.metrics.custom_metrics
    options:
        show_signature_annotations: true

## Metrics Overview

The framework provides comprehensive evaluation metrics for semantic segmentation tasks, including both standard metrics and custom implementations.

### MetricsCollection

The main metrics container that provides standardized evaluation functions:

```python
from segmentation_robustness_framework.metrics import MetricsCollection

# Initialize metrics collection
metrics = MetricsCollection(num_classes=21, ignore_index=255)

# Get metric functions for pipeline
metric_functions = [
    metrics.mean_iou,
    metrics.pixel_accuracy,
    metrics.precision,
    metrics.recall,
    metrics.dice_score
]
```

### Available Metrics

#### Mean IoU (Intersection over Union)

The most commonly used metric for semantic segmentation:

```python
# Calculate mean IoU
iou = metrics.mean_iou(targets, predictions)
print(f"Mean IoU: {iou:.3f}")
```

**Features:**
- Handles class imbalance
- Robust to different class distributions
- Standard benchmark metric

#### Pixel Accuracy

Overall pixel-level accuracy:

```python
# Calculate pixel accuracy
accuracy = metrics.pixel_accuracy(targets, predictions)
print(f"Pixel Accuracy: {accuracy:.3f}")
```

**Features:**
- Simple and intuitive
- Fast computation
- Good for balanced datasets

#### Precision

Per-class precision scores:

```python
# Calculate precision
precision = metrics.precision(targets, predictions)
print(f"Precision: {precision:.3f}")
```

**Features:**
- Per-class evaluation
- Useful for imbalanced datasets
- Detailed performance analysis

#### Recall

Per-class recall scores:

```python
# Calculate recall
recall = metrics.recall(targets, predictions)
print(f"Recall: {recall:.3f}")
```

**Features:**
- Per-class evaluation
- Completeness measure
- Balanced with precision

#### Dice Score (F1-Score)

Harmonic mean of precision and recall:

```python
# Calculate dice score
dice = metrics.dice_score(targets, predictions)
print(f"Dice Score: {dice:.3f}")
```

**Features:**
- Balanced metric
- Good for imbalanced classes
- Medical imaging standard

### Custom Metrics

Create custom metrics by implementing metric functions:

```python
import torch
import torch.nn.functional as F

def custom_metric(targets: torch.Tensor, predictions: torch.Tensor, 
                  num_classes: int, ignore_index: int = 255) -> float:
    """Custom metric implementation."""
    
    # Remove ignored pixels
    mask = targets != ignore_index
    targets = targets[mask]
    predictions = predictions[mask]
    
    # Your custom metric calculation
    # Example: weighted accuracy
    correct = (targets == predictions).float()
    weighted_accuracy = correct.mean()
    
    return weighted_accuracy.item()

# Use custom metric in pipeline
pipeline = SegmentationRobustnessPipeline(
    model=model,
    dataset=dataset,
    attacks=[FGSM(model, eps=0.1)],
    metrics=[custom_metric],
    batch_size=4,
    device="cuda"
)
```

### Metric Registration

Register custom metrics for automatic discovery:

```python
from segmentation_robustness_framework.metrics import register_custom_metric

@register_custom_metric("weighted_accuracy")
def weighted_accuracy(targets: torch.Tensor, predictions: torch.Tensor, 
                     num_classes: int, ignore_index: int = 255) -> float:
    """Weighted accuracy metric."""
    
    # Remove ignored pixels
    mask = targets != ignore_index
    targets = targets[mask]
    predictions = predictions[mask]
    
    # Calculate weighted accuracy
    correct = (targets == predictions).float()
    weighted_accuracy = correct.mean()
    
    return weighted_accuracy.item()

# Now you can use it in configuration
# metrics:
#   - weighted_accuracy
```

### Metric Configuration

Configure metrics in YAML configuration files:

```yaml
metrics:
  ignore_index: 255
  selected_metrics:
    - mean_iou
    - pixel_accuracy
    - precision
    - recall
    - {"name": "dice_score", "average": "micro"}
    - weighted_accuracy  # Custom metric
```

### Metric Usage in Pipeline

Metrics are automatically used by the pipeline:

```python
from segmentation_robustness_framework.pipeline import SegmentationRobustnessPipeline
from segmentation_robustness_framework.metrics import MetricsCollection

# Create metrics collection
metrics = MetricsCollection(num_classes=21, ignore_index=255)

# Create pipeline with metrics
pipeline = SegmentationRobustnessPipeline(
    model=model,
    dataset=dataset,
    attacks=[FGSM(model, eps=0.1)],
    metrics=[
        metrics.mean_iou,
        metrics.pixel_accuracy,
        metrics.precision,
        metrics.recall,
        metrics.dice_score
    ],
    batch_size=4,
    device="cuda"
)

# Run evaluation
results = pipeline.run()

# Access results
clean_iou = results['clean']['mean_iou']
attack_iou = results['attack_fgsm']['mean_iou']

print(f"Clean IoU: {clean_iou:.3f}")
print(f"Attack IoU: {attack_iou:.3f}")
```

### Metric Aggregation

The framework provides different aggregation strategies:

```python
# Micro averaging (global)
micro_precision = metrics.precision(targets, predictions, average='micro')

# Macro averaging (per-class then average)
macro_precision = metrics.precision(targets, predictions, average='macro')

# Weighted averaging (per-class weighted by frequency)
weighted_precision = metrics.precision(targets, predictions, average='weighted')
```

### Performance Considerations

- **GPU Acceleration**: All metrics support GPU computation
- **Memory Efficiency**: Optimized for large batches
- **Batch Processing**: Efficient batch metric computation
- **Numerical Stability**: Robust to edge cases

### Metric Interpretation

Understanding metric results:

```python
# Good performance indicators
good_iou = 0.8      # 80% IoU is excellent
good_accuracy = 0.9  # 90% accuracy is very good
good_dice = 0.85     # 85% dice score is excellent

# Poor performance indicators
poor_iou = 0.3       # 30% IoU indicates issues
poor_accuracy = 0.5  # 50% accuracy is poor
poor_dice = 0.4      # 40% dice score is poor

# Robustness evaluation
robustness_ratio = attack_iou / clean_iou
if robustness_ratio > 0.8:
    print("Model is robust")
elif robustness_ratio > 0.5:
    print("Model has moderate robustness")
else:
    print("Model is vulnerable to attacks")
```
