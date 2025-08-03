# Contributing Guide

Welcome to the **Segmentation Robustness Framework**! We're excited that you're interested in contributing to our project. This guide will help you get started and understand our development process.

## 🤝 How to Contribute

We welcome contributions from the community! Here are the main ways you can contribute:

- 🐛 **Bug Reports** - Help us identify and fix issues
- 💡 **Feature Requests** - Suggest new features or improvements
- 📝 **Documentation** - Improve our docs and examples
- 🔧 **Code Contributions** - Add new models, attacks, metrics, or datasets
- 🧪 **Testing** - Help ensure code quality and reliability

## 📋 Table of Contents

- [Getting Started](#getting-started)
- [Development Setup](#development-setup)
- [Code Style and Standards](#code-style-and-standards)
- [Adding New Components](#adding-new-components)
- [Testing Guidelines](#testing-guidelines)
- [Documentation Guidelines](#documentation-guidelines)
- [Pull Request Process](#pull-request-process)
- [Release Process](#release-process)

## 🚀 Getting Started

### Prerequisites

- Python 3.12+
- Git
- Basic knowledge of PyTorch and semantic segmentation

### Quick Setup

1. **Fork the repository**
   ```bash
   git clone https://github.com/your-username/segmentation-robustness-framework.git
   cd segmentation-robustness-framework
   ```

2. **Create a virtual environment**
   ```bash
   python -m venv .venv
   source .venv/bin/activate
   ```

3. **Install development dependencies**
   ```bash
   pip install -e ".[dev]"
   ```

4. **Run tests to verify setup**
   ```bash
   pytest tests/
   ```

## 🔧 Development Setup

### Project Structure

```
segmentation-robustness-framework/
├── segmentation_robustness_framework/    # Main package
│   ├── adapters/                        # Model adapters
│   ├── attacks/                         # Adversarial attacks
│   ├── datasets/                        # Dataset implementations
│   ├── loaders/                         # Model and dataset loaders
│   ├── metrics/                         # Evaluation metrics
│   ├── pipeline/                        # Core pipeline
│   └── utils/                          # Utility functions
├── tests/                              # Test suite
├── docs/                               # Documentation
├── examples/                           # Usage examples
├── scripts/                            # Development scripts
└── pyproject.toml                     # Project configuration
```

### Development Tools

We use several tools to maintain code quality:

- **Ruff** - Code formatting and linting
- **Pytest** - Testing framework
- **Pre-commit** - Git hooks for code quality

### Setting up Pre-commit Hooks

```bash
# Install pre-commit
pip install pre-commit

# Install git hooks
pre-commit install

# Run on all files (optional)
pre-commit run --all-files
```

## 📏 Code Style and Standards

### Python Style Guide

We follow [PEP 8](https://pep8.org/) with some modifications:

- **Line length**: 88 characters (Black default)
- **Import sorting**: Automatic with `isort`
- **Type hints**: Required for all public functions
- **Docstrings**: Google-style format

### Code Formatting

```bash
# Format code with Ruff
ruff format .

# Run all linting checks
ruff check .

# Make fixes
ruff check --fix .
```

### Type Hints

All public functions must include type hints:

```python
def evaluate_model(
    model: SegmentationModelProtocol,
    dataset: torch.utils.data.Dataset,
    metrics: list[Callable]
) -> dict[str, float]:
    """Evaluate a segmentation model.
    
    Args:
        model: The model to evaluate.
        dataset: Dataset for evaluation.
        metrics: List of metric functions.
        
    Returns:
        Dictionary of metric results.
    """
    # Implementation here
    pass
```

### Docstring Standards

We use Google-style docstrings with specific requirements:

```python
def process_images(images: torch.Tensor, target_size: tuple[int, int]) -> torch.Tensor:
    """Process images to target size.
    
    Resize and normalize images to the specified target size.
    
    Args:
        images: Input images [B, C, H, W].
        target_size: Target size as (height, width).
        
    Returns:
        Processed images [B, C, H, W].
        
    Raises:
        ValueError: If target_size is invalid.
        
    Example:
        ```python
        images = torch.randn(4, 3, 224, 224)
        processed = process_images(images, (512, 512))
        ```
    """
    # Implementation
    pass
```

## 🧩 Adding New Components

### Adding a New Attack

1. **Create the attack class** in `segmentation_robustness_framework/attacks/`:

```python
from segmentation_robustness_framework.attacks import AdversarialAttack
from segmentation_robustness_framework.attacks.registry import register_attack

@register_attack("custom_attack")
class CustomAttack(AdversarialAttack):
    """Custom adversarial attack implementation."""
    
    def __init__(self, model: nn.Module, eps: float = 0.02):
        """Initialize custom attack.
        
        Args:
            model: Model to attack.
            eps: Maximum perturbation magnitude.
        """
        super().__init__(model)
        self.eps = eps
    
    def apply(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Apply attack to images.
        
        Args:
            images: Input images [B, C, H, W].
            labels: Target labels [B, H, W].
            
        Returns:
            Adversarial images [B, C, H, W].
        """
        # Implementation here
        return adversarial_images
```

2. **Add tests** in `tests/attacks/test_custom_attack.py`:

```python
import pytest
import torch
from segmentation_robustness_framework.attacks import CustomAttack

def test_custom_attack_initialization():
    """Test custom attack initialization."""
    model = create_mock_model()
    attack = CustomAttack(model, eps=0.02)
    assert attack.eps == 0.02

def test_custom_attack_application():
    """Test custom attack application."""
    model = create_mock_model()
    attack = CustomAttack(model, eps=0.02)
    
    images = torch.randn(2, 3, 224, 224)
    labels = torch.randint(0, 21, (2, 224, 224))
    
    adv_images = attack.apply(images, labels)
    
    assert adv_images.shape == images.shape
    assert torch.all(adv_images >= 0) and torch.all(adv_images <= 1)
```

### Adding a New Metric

1. **Create the metric function** in `segmentation_robustness_framework/metrics/`:

```python
from segmentation_robustness_framework.metrics.registry import register_custom_metric

@register_custom_metric("custom_metric")
def custom_metric(targets: torch.Tensor, predictions: torch.Tensor) -> float:
    """Compute custom metric.
    
    Args:
        targets: Ground truth labels [B, H, W].
        predictions: Predicted labels [B, H, W].
        
    Returns:
        Metric value.
    """
    # Implementation here
    return metric_value
```

2. **Add tests** in `tests/metrics/test_custom_metric.py`:

```python
import pytest
import torch
from segmentation_robustness_framework.metrics import custom_metric

def test_custom_metric():
    """Test custom metric computation."""
    targets = torch.randint(0, 21, (2, 224, 224))
    predictions = torch.randint(0, 21, (2, 224, 224))
    
    result = custom_metric(targets, predictions)
    
    assert isinstance(result, float)
    assert 0 <= result <= 1
```

### Adding a New Dataset

1. **Create the dataset class** in `segmentation_robustness_framework/datasets/`:

```python
from torch.utils.data import Dataset
from segmentation_robustness_framework.datasets.registry import register_dataset

@register_dataset("custom_dataset")
class CustomDataset(Dataset):
    """Custom dataset implementation."""
    
    def __init__(self, root: str, split: str = "train", transform=None):
        """Initialize custom dataset.
        
        Args:
            root: Dataset root directory.
            split: Dataset split ('train', 'val', 'test').
            transform: Image transformations.
            target_transform: Mask transformations.
        """
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        # Implementation here
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Implementation here
        return image, mask
```

2. **Add tests** in `tests/datasets/test_custom_dataset.py`:

```python
import pytest
import torch
from segmentation_robustness_framework.datasets import CustomDataset

def test_custom_dataset():
    """Test custom dataset."""
    dataset = CustomDataset("./test_data", split="val")
    
    assert len(dataset) > 0
    
    image, mask = dataset[0]
    
    assert isinstance(image, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert image.dim() == 3  # [C, H, W]
    assert mask.dim() == 2   # [H, W]
```

### Adding a New Model Adapter

1. **Create the adapter class** in `segmentation_robustness_framework/adapters/`:

```python
from segmentation_robustness_framework.adapters.base_protocol import SegmentationModelProtocol

class CustomModelAdapter(SegmentationModelProtocol):
    """Adapter for custom model implementation."""
    
    def __init__(self, model: nn.Module, num_classes: int):
        """Initialize adapter.
        
        Args:
            model: The underlying model.
            num_classes: Number of segmentation classes.
        """
        self.model = model
        self.num_classes = num_classes
    
    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Get model logits.
        
        Args:
            x: Input images [B, C, H, W].
            
        Returns:
            Logits [B, num_classes, H, W].
        """
        return self.model(x)
    
    def predictions(self, x: torch.Tensor) -> torch.Tensor:
        """Get model predictions.
        
        Args:
            x: Input images [B, C, H, W].
            
        Returns:
            Predictions [B, H, W].
        """
        logits = self.logits(x)
        return torch.argmax(logits, dim=1)
```

2. **Add tests** in `tests/adapters/test_custom_adapter.py`:

```python
import pytest
import torch
from segmentation_robustness_framework.adapters import CustomModelAdapter

def test_custom_adapter():
    """Test custom model adapter."""
    model = create_mock_model()
    adapter = CustomModelAdapter(model, num_classes=21)
    
    x = torch.randn(2, 3, 224, 224)
    
    logits = adapter.logits(x)
    predictions = adapter.predictions(x)
    
    assert logits.shape == (2, 21, 224, 224)
    assert predictions.shape == (2, 224, 224)
```

## 🧪 Testing Guidelines

### Running Tests

```bash
# Run all tests
pytest

# Run specific test file
pytest tests/test_attacks.py

# Run with coverage
pytest --cov=segmentation_robustness_framework

# Run with verbose output
pytest -v
```

### Test Structure

```
tests/
├── adapters/               # Model adapter tests
│   ├── test_custom_adapter.py
│   ├── test_huggingface_adapter.py
│   ├── test_smp_adapter.py
│   ├── test_torchvision_adapter.py
│   └── test_registry.py
├── attacks/                # Attack implementation tests
│   ├── test_base_attack.py
│   ├── test_fgsm.py
│   ├── test_pgd.py
│   ├── test_rfgsm.py
│   ├── test_tpgd.py
│   ├── test_custom_attacks.py
│   └── test_registry.py
├── datasets/               # Dataset tests
│   ├── test_voc_dataset.py
│   ├── test_ade20k_dataset.py
│   ├── test_cityscapes_dataset.py
│   ├── test_stanford_background_dataset.py
│   └── test_registry.py
├── loaders/                # Model and dataset loader tests
│   ├── test_universal_model_loader.py
│   ├── test_torchvision_model_loader.py
│   ├── test_smp_model_loader.py
│   ├── test_huggingface_model_loader.py
│   ├── test_custom_model_loader.py
│   ├── test_dataset_loader.py
│   └── test_attack_loader.py
├── metrics/                # Metric tests
│   ├── test_base_metrics.py
│   └── test_custom_metrics.py
├── pipeline/               # Pipeline tests
│   ├── test_core.py
│   └── test_config.py
├── utils/                  # Utility function tests
│   ├── test_image_preprocessing.py
│   ├── test_model_utils.py
│   ├── test_dataset_utils.py
│   ├── test_loader_utils.py
│   └── test_visualization.py
└── data/                   # Test data files
    ├── dummy_model_weights.pth
    ├── dummy_model_checkpoint.pth
    └── dummy_encoder_weights.pth
```

### Writing Tests

1. **Use descriptive test names**:

```python
def test_fgsm_attack_creates_valid_adversarial_images():
    """Test that FGSM creates valid adversarial images."""
    # Test implementation

def test_metric_handles_empty_predictions():
    """Test that metric handles empty predictions gracefully."""
    # Test implementation
```

2. **Use fixtures for common setup**:

```python
@pytest.fixture
def sample_model():
    """Create a sample model for testing."""
    return create_mock_model()

@pytest.fixture
def sample_dataset():
    """Create a sample dataset for testing."""
    return create_mock_dataset()

def test_attack_with_sample_model(sample_model, sample_dataset):
    """Test attack with sample model and dataset."""
    # Test implementation
```

3. **Test edge cases**:

```python
def test_attack_with_zero_epsilon():
    """Test attack behavior with zero epsilon."""
    # Should return original images

def test_metric_with_single_class():
    """Test metric with single class dataset."""
    # Should handle gracefully
```

### Test Coverage

We aim for >90% test coverage. To check coverage:

```bash
# Generate coverage report
pytest --cov=segmentation_robustness_framework --cov-report=html

# View coverage report
open htmlcov/index.html
```

## 📝 Documentation Guidelines

### Code Documentation

- **All public functions** must have docstrings
- **All classes** must have docstrings
- **Complex algorithms** should include inline comments
- **Type hints** are required for all public APIs

### API Documentation

When adding new components, update the relevant documentation:

1. **Update API reference** in `docs/api_reference/`
2. **Add usage examples** in `docs/examples/`
3. **Update user guide** if needed
4. **Add to component registry** documentation

### Documentation Standards

- Use **Google-style docstrings**
- Include **type hints** in docstrings
- Provide **usage examples**
- Document **exceptions** and edge cases
- Keep documentation **up-to-date** with code changes

## 🔄 Pull Request Process

### Before Submitting

1. **Ensure code quality**:
   ```bash
   # Run all checks
   ruff check .
   ruff format .
   pytest tests/
   ```

2. **Update documentation**:
   - Add docstrings for new functions
   - Update API documentation
   - Add usage examples

3. **Add tests**:
   - Unit tests for new functionality
   - Integration tests if needed
   - Update existing tests if breaking changes

### Pull Request Template

```markdown
## Description

Brief description of changes.

## Type of Change

- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing

- [ ] Tests pass
- [ ] Documentation builds correctly

## Checklist

- [ ] Code follows style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
- [ ] Tests added/updated
- [ ] No breaking changes (or documented)
```

### Review Process

1. **Automated checks** must pass
2. **Code review** by maintainers
3. **Documentation review** if needed
4. **Final approval** and merge

## 🚀 Release Process

### Versioning

We follow [Semantic Versioning](https://semver.org/):

- **MAJOR.MINOR.PATCH**
- **MAJOR**: Breaking changes
- **MINOR**: New features (backward compatible)
- **PATCH**: Bug fixes (backward compatible)

### Release Checklist

1. **Update version** in `pyproject.toml` and `__version__.py`
2. **Update changelog** in `CHANGELOG.md`
3. **Run full test suite**
4. **Update documentation**
5. **Create release tag**
6. **Publish to PyPI**

### Creating a Release

```bash
# Update version
poetry version patch  # or minor/major

# Run tests
pytest

# Build and publish
poetry build
poetry publish

# Create git tag
git tag v1.2.3
git push origin v1.2.3
```

## 🤝 Community Guidelines

### Code of Conduct

We are committed to providing a welcoming and inclusive environment. Please:

- **Be respectful** and inclusive
- **Help others** learn and grow
- **Give constructive feedback**
- **Report issues** appropriately

### Communication

- **GitHub Issues**: Bug reports and feature requests
- **GitHub Discussions**: Questions and general discussion
- **Pull Requests**: Code contributions
- **Documentation**: Help improve docs

## 📚 Additional Resources

- **[Development Setup](development_setup.md)** - Detailed development environment setup
- **[Testing Guide](testing_guide.md)** - Comprehensive testing guidelines
- **[Documentation Guide](documentation_guide.md)** - Documentation standards
- **[Release Guide](release_guide.md)** - Release process details

## 🆘 Getting Help

If you need help contributing:

1. **Check existing issues** and discussions
2. **Read the documentation** thoroughly
3. **Ask questions** in GitHub Discussions

Thank you for contributing to the Segmentation Robustness Framework! 🎉
