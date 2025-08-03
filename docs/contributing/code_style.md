# Code Style Guide

This guide covers coding standards and best practices for the Segmentation Robustness Framework.

## 🎯 Code Style Philosophy

### Core Principles

- **Readability**: Code should be easy to read and understand
- **Consistency**: Follow consistent patterns throughout the codebase
- **Maintainability**: Write code that's easy to modify and extend
- **Performance**: Write efficient code without sacrificing readability
- **Safety**: Write robust code that handles edge cases gracefully

### Style Guidelines

- **Follow PEP 8**: With modifications for modern Python
- **Use type hints**: For all public functions and methods
- **Write clear docstrings**: Google-style format
- **Use meaningful names**: Variables, functions, and classes
- **Keep functions small**: Single responsibility principle

## 📏 Python Style Standards

### Code Formatting

We use **Ruff** for code formatting and linting:

```bash
# Format code
ruff format .

# Check for style issues
ruff check .

# Fix auto-fixable issues
ruff check --fix .
```

### Line Length and Formatting

- **Line length**: 88 characters (Black default)
- **Indentation**: 4 spaces (no tabs)
- **Trailing whitespace**: None
- **Blank lines**: Use sparingly but consistently

### Import Organization

```python
# Standard library imports
import os
import sys
from pathlib import Path
from typing import Any, Callable, Optional

# Third-party imports
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Local imports
from segmentation_robustness_framework.adapters import SegmentationModelProtocol
from segmentation_robustness_framework.utils import image_preprocessing
```

### Naming Conventions

#### Variables and Functions

```python
# Good - descriptive and clear
def compute_mean_iou(targets, predictions):
    return mean_iou_score

# Bad - unclear or abbreviated
def calc_miou(targs, preds):
    return miou
```

#### Classes

```python
# Good - PascalCase for classes
class SegmentationRobustnessPipeline:
    pass

class CustomAttack(AdversarialAttack):
    pass

# Bad - snake_case for classes
class segmentation_robustness_pipeline:
    pass
```

#### Constants

```python
# Good - UPPER_SNAKE_CASE for constants
DEFAULT_BATCH_SIZE = 8
MAX_IMAGE_SIZE = 1024
SUPPORTED_DATASETS = ["voc", "ade20k", "cityscapes"]

# Bad - mixed case for constants
defaultBatchSize = 8
max_image_size = 1024
```

## 🔧 Type Hints

### Function Signatures

```python
# Good - complete type hints
def evaluate_model(
    model: SegmentationModelProtocol,
    dataset: torch.utils.data.Dataset,
    metrics: list[Callable[[torch.Tensor, torch.Tensor], float]]
) -> dict[str, float]:
    pass

# Bad - missing type hints
def evaluate_model(model, dataset, metrics):
    pass
```

### Variable Annotations

```python
# Good - explicit type annotations
def process_batch(images: torch.Tensor) -> torch.Tensor:
    batch_size: int = images.shape[0]
    processed_images: torch.Tensor = preprocess(images)
    return processed_images

# Bad - implicit types
def process_batch(images):
    batch_size = images.shape[0]
    processed_images = preprocess(images)
    return processed_images
```

### Generic Types

```python
# Good - use built-in generics
from typing import Any, Callable, Optional

def create_attack(
    attack_type: str,
    model: nn.Module,
    config: dict[str, Any]
) -> Optional[AdversarialAttack]:
    pass

# Bad - use typing.Dict, typing.List
from typing import Dict, List

def create_attack(
    attack_type: str,
    model: nn.Module,
    config: Dict[str, Any]
) -> Optional[AdversarialAttack]:
    pass
```

## 📝 Docstring Standards

### Google-Style Docstrings

```python
def compute_metrics(
    targets: torch.Tensor,
    predictions: torch.Tensor,
    num_classes: int,
    ignore_index: int = 255
) -> dict[str, float]:
    """Compute segmentation metrics.
    
    Calculate various segmentation metrics including mean IoU, pixel accuracy,
    and per-class precision/recall.
    
    Args:
        targets: Ground truth labels [B, H, W] or [H, W].
        predictions: Predicted labels [B, H, W] or [H, W].
        num_classes: Number of segmentation classes.
        ignore_index: Label to ignore in computation. Defaults to 255.
        
    Returns:
        Dictionary containing metric values:
            - 'mean_iou': Mean intersection over union
            - 'pixel_accuracy': Overall pixel accuracy
            - 'precision': Per-class precision scores
            - 'recall': Per-class recall scores
            
    Raises:
        ValueError: If targets and predictions have different shapes.
        RuntimeError: If metric computation fails.
        
    Example:
        ```python
        targets = torch.randint(0, 21, (2, 224, 224))
        predictions = torch.randint(0, 21, (2, 224, 224))
        metrics = compute_metrics(targets, predictions, num_classes=21)
        print(f"IoU: {metrics['mean_iou']:.3f}")
        ```
    """
    # Implementation
    pass
```

### Class Docstrings

```python
class SegmentationRobustnessPipeline:
    """Pipeline for evaluating segmentation models under adversarial attacks.
    
    This pipeline provides a unified interface for evaluating segmentation
    models on clean and adversarial images. It supports multiple attacks,
    metrics, and output formats.
    
    The pipeline automatically handles:
    - Model evaluation on clean images
    - Attack generation and evaluation
    - Metric computation and aggregation
    - Result saving and visualization
    
    Attributes:
        model: The segmentation model to evaluate.
        dataset: Dataset for evaluation.
        attacks: List of attack instances.
        metrics: List of metric functions.
        batch_size: Batch size for evaluation.
        device: Device to use for computation.
        output_dir: Directory to save results.
        
    Example:
        ```python
        pipeline = SegmentationRobustnessPipeline(
            model=model,
            dataset=dataset,
            attacks=[FGSM(model, eps=0.02)],
            metrics=[mean_iou, pixel_accuracy],
            batch_size=4,
            device="cuda"
        )
        
        results = pipeline.run()
        pipeline.print_summary()
        ```
    """
```

### Module Docstrings

```python
"""Segmentation Robustness Framework - Attacks Module.

This module provides implementations of adversarial attacks for semantic
segmentation models. All attacks follow a common interface and can be
used interchangeably in the evaluation pipeline.

Available Attacks:
    - FGSM: Fast Gradient Sign Method
    - PGD: Projected Gradient Descent
    - RFGSM: R-FGSM with momentum
    - TPGD: Two-Phase Gradient Descent

Example:
    ```python
    from segmentation_robustness_framework.attacks import FGSM, PGD
    
    # Create attacks
    fgsm_attack = FGSM(model, eps=0.02)
    pgd_attack = PGD(model, eps=0.02, alpha=0.01, iters=10)
    
    # Use in pipeline
    pipeline = SegmentationRobustnessPipeline(
        model=model,
        dataset=dataset,
        attacks=[fgsm_attack, pgd_attack],
        metrics=metrics
    )
    ```
"""
```

## 🏗️ Code Structure

### Function Design

```python
# Good - single responsibility, clear parameters
def apply_attack(
    images: torch.Tensor,
    labels: torch.Tensor,
    attack: AdversarialAttack,
    device: str = "cpu"
) -> torch.Tensor:
    """Apply adversarial attack to images."""
    # Move to device
    images = images.to(device)
    labels = labels.to(device)
    
    # Apply attack
    adv_images = attack.apply(images, labels)
    
    return adv_images

# Bad - multiple responsibilities, unclear parameters
def process_images_and_apply_attack_and_compute_metrics(
    images, labels, attack, device="cpu", compute_metrics=True
):
    # Too many responsibilities
    pass
```

### Class Design

```python
# Good - clear interface, proper encapsulation
class AdversarialAttack:
    """Base class for adversarial attacks."""
    
    def __init__(self, model: nn.Module):
        """Initialize attack with target model."""
        self.model = model
        self.device = next(model.parameters()).device
    
    def apply(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Apply attack to images."""
        raise NotImplementedError
    
    def set_device(self, device: str) -> None:
        """Set device for attack computation."""
        self.device = device

# Bad - unclear interface, poor encapsulation
class Attack:
    def __init__(self, model, eps=0.02, alpha=0.01, iters=10, targeted=False):
        # Too many parameters, unclear purpose
        pass
```

### Error Handling

```python
# Good - specific exceptions, helpful messages
def load_model(model_config: dict[str, Any]) -> nn.Module:
    """Load model from configuration."""
    if "name" not in model_config:
        raise ValueError("Model configuration must contain 'name' field")
    
    model_name = model_config["name"]
    try:
        model = create_model(model_name, model_config)
    except ImportError as e:
        raise RuntimeError(f"Failed to import model '{model_name}': {e}")
    except Exception as e:
        raise RuntimeError(f"Failed to create model '{model_name}': {e}")
    
    return model

# Bad - generic exceptions, unhelpful messages
def load_model(model_config):
    try:
        return create_model(model_config["name"], model_config)
    except:
        raise Exception("Error loading model")
```

## 🔄 Code Patterns

### Context Managers

```python
# Good - use context managers for resource management
def evaluate_with_gpu_memory_management():
    """Evaluate with GPU memory management."""
    with torch.cuda.amp.autocast():
        with torch.no_grad():
            results = model(images)
    
    torch.cuda.empty_cache()
    return results

# Bad - manual resource management
def evaluate_without_memory_management():
    """Evaluate without proper memory management."""
    results = model(images)
    return results
```

### List Comprehensions

```python
# Good - use list comprehensions for simple transformations
def get_attack_names(attacks: list[AdversarialAttack]) -> list[str]:
    """Get names of all attacks."""
    return [attack.__class__.__name__ for attack in attacks]

# Bad - verbose loop
def get_attack_names(attacks):
    names = []
    for attack in attacks:
        names.append(attack.__class__.__name__)
    return names
```

### Dictionary Comprehensions

```python
# Good - use dict comprehensions for mappings
def create_metric_dict(metrics: list[Callable]) -> dict[str, Callable]:
    """Create dictionary mapping metric names to functions."""
    return {metric.__name__: metric for metric in metrics}

# Bad - verbose dictionary creation
def create_metric_dict(metrics):
    metric_dict = {}
    for metric in metrics:
        metric_dict[metric.__name__] = metric
    return metric_dict
```

## 🧪 Testing Patterns

### Test Structure

```python
# Good - clear test structure with descriptive names
def test_fgsm_attack_creates_valid_adversarial_images():
    """Test that FGSM creates valid adversarial images."""
    # Arrange
    model = create_mock_model()
    attack = FGSM(model, eps=0.02)
    images = torch.randn(2, 3, 224, 224)
    labels = torch.randint(0, 21, (2, 224, 224))
    
    # Act
    adv_images = attack.apply(images, labels)
    
    # Assert
    assert adv_images.shape == images.shape
    assert torch.all(adv_images >= 0) and torch.all(adv_images <= 1)
    assert torch.any(adv_images != images)

# Bad - unclear test purpose
def test_attack():
    model = create_model()
    attack = FGSM(model)
    result = attack.apply(images, labels)
    assert result is not None
```

### Fixtures

```python
# Good - reusable fixtures with clear names
@pytest.fixture
def sample_model():
    """Create a sample model for testing."""
    return create_mock_model()

@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    return create_mock_dataset()

def test_attack_with_sample_data(sample_model, sample_dataset):
    """Test attack with sample model and dataset."""
    attack = FGSM(sample_model, eps=0.02)
    # Test implementation

# Bad - inline setup, no reusability
def test_attack():
    model = create_mock_model()  # Inline setup
    dataset = create_mock_dataset()  # Inline setup
    attack = FGSM(model, eps=0.02)
    # Test implementation
```

## 🔧 Performance Considerations

### Memory Efficiency

```python
# Good - memory efficient operations
def process_large_batch(images: torch.Tensor) -> torch.Tensor:
    """Process large batch of images efficiently."""
    batch_size = images.shape[0]
    chunk_size = 4  # Process in chunks
    
    results = []
    for i in range(0, batch_size, chunk_size):
        chunk = images[i:i + chunk_size]
        processed_chunk = process_chunk(chunk)
        results.append(processed_chunk)
    
    return torch.cat(results, dim=0)

# Bad - process all at once, may cause OOM
def process_large_batch(images):
    return process_all_images(images)  # May cause OOM
```

### GPU Operations

```python
# Good - efficient GPU operations
def compute_metrics_gpu(targets: torch.Tensor, predictions: torch.Tensor) -> dict[str, float]:
    """Compute metrics on GPU efficiently."""
    # Keep on GPU for computation
    intersection = (targets == predictions).float().sum()
    union = targets.numel()
    
    # Move to CPU only for final result
    accuracy = (intersection / union).cpu().item()
    return {"accuracy": accuracy}

# Bad - unnecessary CPU-GPU transfers
def compute_metrics_gpu(targets, predictions):
    # Move to CPU for computation
    targets_cpu = targets.cpu()
    predictions_cpu = predictions.cpu()
    
    intersection = (targets_cpu == predictions_cpu).float().sum()
    union = targets_cpu.numel()
    
    accuracy = intersection / union
    return {"accuracy": accuracy}
```

## 🚨 Common Anti-Patterns

### Avoid These Patterns

```python
# ❌ Don't use global variables
global_config = {"batch_size": 8, "device": "cuda"}

def process_data():
    global global_config
    batch_size = global_config["batch_size"]

# ✅ Use dependency injection
def process_data(config: dict[str, Any]):
    batch_size = config["batch_size"]

# ❌ Don't use mutable default arguments
def create_attack(model, attacks=[]):
    attacks.append(FGSM(model))
    return attacks

# ✅ Use None as default
def create_attack(model, attacks=None):
    if attacks is None:
        attacks = []
    attacks.append(FGSM(model))
    return attacks

# ❌ Don't catch all exceptions
def load_model(model_name):
    try:
        return create_model(model_name)
    except:  # Too broad
        return None

# ✅ Catch specific exceptions
def load_model(model_name):
    try:
        return create_model(model_name)
    except ImportError as e:
        logger.error(f"Model {model_name} not found: {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected error loading {model_name}: {e}")
        raise
```

## 🔍 Code Review Checklist

### Before Submitting

- [ ] **Code formatting**: Run `ruff format .`
- [ ] **Linting**: Run `ruff check .`
- [ ] **Tests**: Run `pytest` and ensure all pass
- [ ] **Documentation**: Add/update docstrings
- [ ] **Type hints**: Add for all public functions
- [ ] **Error handling**: Add appropriate exception handling
- [ ] **Performance**: Consider efficiency for large datasets
- [ ] **Security**: Validate inputs and handle edge cases

### Review Questions

1. **Is the code readable?** Can someone else understand it easily?
2. **Is it well-documented?** Are docstrings clear and complete?
3. **Is it testable?** Can the code be easily unit tested?
4. **Is it efficient?** Are there obvious performance issues?
5. **Is it safe?** Does it handle edge cases and errors gracefully?
6. **Is it consistent?** Does it follow project conventions?

## 🚀 Next Steps

After reading this code style guide:

1. **Set up your development environment** following the [Development Setup Guide](development_setup.md)
2. **Practice the style guidelines** in your contributions
3. **Use the provided tools** to maintain quality
4. **Review existing code** to understand current patterns
5. **Ask for feedback** on your code style during reviews

Happy coding! 💻
