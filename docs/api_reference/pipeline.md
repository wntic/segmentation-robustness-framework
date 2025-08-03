# Pipeline API

This page documents the pipeline components of the Segmentation Robustness Framework.

::: segmentation_robustness_framework.pipeline.core
    options:
        show_signature_annotations: true

---

::: segmentation_robustness_framework.pipeline.config
    options:
        show_signature_annotations: true

## Pipeline Configuration

The pipeline configuration system allows you to define complete experiments using YAML configuration files.

### Configuration Structure

```yaml
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

### Pipeline Execution

The pipeline orchestrates the entire evaluation process:

1. **Model Loading**: Loads the specified model with appropriate adapter
2. **Dataset Loading**: Loads and preprocesses the dataset
3. **Attack Generation**: Creates adversarial examples using specified attacks
4. **Evaluation**: Runs both clean and adversarial evaluation
5. **Reporting**: Generates comprehensive results and visualizations

### Results Structure

The pipeline returns a dictionary with the following structure:

```python
{
    'clean': {
        'mean_iou': 0.823,
        'pixel_accuracy': 0.956,
        'precision': 0.891,
        'recall': 0.845
    },
    'attack_fgsm': {
        'mean_iou': 0.452,
        'pixel_accuracy': 0.723,
        'precision': 0.567,
        'recall': 0.489
    },
    'attack_pgd': {
        'mean_iou': 0.231,
        'pixel_accuracy': 0.456,
        'precision': 0.234,
        'recall': 0.198
    }
}
```
