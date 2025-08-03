# CLI API

This page documents the command-line interface components of the Segmentation Robustness Framework.

## CLI Commands

The framework provides a unified command-line interface for all operations.

### Main CLI

```bash
# Main CLI entry point
python -m segmentation_robustness_framework.cli.main run config.yaml
python -m segmentation_robustness_framework.cli.main list --attacks
python -m segmentation_robustness_framework.cli.main test --coverage
```

#### Available Commands

- `run`: Execute pipeline from configuration file
- `list`: List available components
- `test`: Run tests

### Run Command

Execute a pipeline from a configuration file.

```bash
python -m segmentation_robustness_framework.cli.main run config.yaml [options]
```

#### Options

- `config_file`: Path to configuration file (required)
- `--save`: Save results (default: True)
- `--show`: Show visualizations
- `--verbose, -v`: Enable verbose logging
- `--override, -o`: Override configuration values (format: section.key=value)
- `--summary-only`: Only print configuration summary without running pipeline

#### Examples

```bash
# Basic usage
python -m segmentation_robustness_framework.cli.main run config.yaml

# With options
python -m segmentation_robustness_framework.cli.main run config.yaml --save --show --verbose

# Override configuration values
python -m segmentation_robustness_framework.cli.main run config.yaml --override pipeline.device cuda pipeline.batch_size 4
```

### List Command

List available components in the framework.

```bash
python -m segmentation_robustness_framework.cli.main list [options]
```

#### Options

- `--models`: List available models
- `--attacks`: List available attacks
- `--metrics`: List available metrics
- `--datasets`: List available datasets
- `--examples`: List available configuration examples

#### Examples

```bash
# List all components
python -m segmentation_robustness_framework.cli.main list

# List specific components
python -m segmentation_robustness_framework.cli.main list --attacks
python -m segmentation_robustness_framework.cli.main list --models --datasets
```

### Test Command

Run tests for the framework.

```bash
python -m segmentation_robustness_framework.cli.main test [options] [test_path]
```

#### Options

- `--loaders`: Run loader tests
- `--adapters`: Run adapter tests
- `--attacks`: Run attack tests
- `--metrics`: Run metric tests
- `--pipeline`: Run pipeline tests
- `--coverage`: Run tests with coverage report
- `--verbose, -v`: Enable verbose output
- `test_path`: Specific test file or test to run (optional)

#### Examples

```bash
# Run all tests
python -m segmentation_robustness_framework.cli.main test

# Run specific test categories
python -m segmentation_robustness_framework.cli.main test --loaders --attacks

# Run with coverage
python -m segmentation_robustness_framework.cli.main test --coverage --verbose

# Run specific test file
python -m segmentation_robustness_framework.cli.main test tests/test_pipeline.py
```

## Configuration Examples

### Basic Configuration

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
    alpha: 0.01
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

### Advanced Configuration

```yaml
# advanced_config.yaml
pipeline:
  device: cuda
  batch_size: 8
  output_dir: results/experiment_1
  auto_resize_masks: true
  output_formats: ["json", "csv"]

model:
  type: smp
  config:
    architecture: unet
    encoder_name: resnet34
    encoder_weights: imagenet
    classes: 21

dataset:
  name: ade20k
  split: val
  root: ./data
  image_shape: [512, 512]
  download: true

attacks:
  - name: fgsm
    eps: 0.02
  - name: fgsm
    eps: 0.05
  - name: fgsm
    eps: 0.1
  - name: pgd
    eps: 0.02
    alpha: 0.01
    iters: 10
    targeted: false
  - name: pgd
    eps: 0.05
    alpha: 0.01
    iters: 20
    targeted: false

metrics:
  ignore_index: 255
  selected_metrics:
    - mean_iou
    - pixel_accuracy
    - precision
    - recall
    - {"name": "dice_score", "average": "micro"}
```

## CLI Usage Examples

### Quick Start

```bash
# Run a basic evaluation
python -m segmentation_robustness_framework.cli.main run config.yaml

# Check available components
python -m segmentation_robustness_framework.cli.main list

# Run tests
python -m segmentation_robustness_framework.cli.main test
```

### Development Workflow

```bash
# Test specific components
python -m segmentation_robustness_framework.cli.main test --loaders --adapters

# Run with verbose output
python -m segmentation_robustness_framework.cli.main run config.yaml --verbose

# Override configuration for quick testing
python -m segmentation_robustness_framework.cli.main run config.yaml \
  --override pipeline.batch_size 2 \
  --override pipeline.device cpu
```

### Production Workflow

```bash
# Run with full configuration
python -m segmentation_robustness_framework.cli.main run production_config.yaml

# Save results and visualizations
python -m segmentation_robustness_framework.cli.main run production_config.yaml \
  --save \
  --show

# Run with custom output directory
python -m segmentation_robustness_framework.cli.main run production_config.yaml \
  --override pipeline.output_dir results/$(date +%Y%m%d_%H%M%S)
```

## Error Handling

The CLI includes comprehensive error handling:

```bash
# Invalid configuration file
python -m segmentation_robustness_framework.cli.main run invalid_config.yaml
# Error: Configuration file not found or invalid

# Missing required parameters
python -m segmentation_robustness_framework.cli.main run config.yaml \
  --override model.name nonexistent_model
# Error: Model not found

# GPU not available
python -m segmentation_robustness_framework.cli.main run config.yaml \
  --override pipeline.device cuda
# Error: CUDA not available, falling back to CPU
```

## Performance Optimization

### Memory Management

```bash
# Reduce batch size for memory constraints
python -m segmentation_robustness_framework.cli.main run config.yaml \
  --override pipeline.batch_size 1

# Use CPU for memory-intensive operations
python -m segmentation_robustness_framework.cli.main run config.yaml \
  --override pipeline.device cpu
```

### Parallel Processing

```bash
# Use multiple workers for data loading
python -m segmentation_robustness_framework.cli.main run config.yaml \
  --override pipeline.num_workers 4
```

## Logging and Debugging

### Verbose Logging

```bash
# Enable verbose output
python -m segmentation_robustness_framework.cli.main run config.yaml --verbose

# Debug specific components
python -m segmentation_robustness_framework.cli.main run config.yaml \
  --override pipeline.debug true
```

### Log Files

```bash
# Save logs to file
python -m segmentation_robustness_framework.cli.main run config.yaml \
  --override pipeline.log_file logs/experiment.log
```

## Integration with Other Tools

### Scripting

```bash
#!/bin/bash
# Run multiple experiments
for eps in 0.05 0.1 0.2; do
  python -m segmentation_robustness_framework.cli.main run config.yaml \
    --override attacks.0.eps $eps \
    --override pipeline.output_dir results/eps_$eps
done
```

### Automation

```bash
# Run tests before deployment
python -m segmentation_robustness_framework.cli.main test --coverage

# Validate configuration
python -m segmentation_robustness_framework.cli.main run config.yaml --summary-only
``` 