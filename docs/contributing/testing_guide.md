# Testing Guide

This guide covers testing practices and standards for the Segmentation Robustness Framework.

## 🧪 Testing Philosophy

### Testing Principles

- **Comprehensive Coverage**: Aim for >90% code coverage
- **Fast Execution**: Tests should run quickly (<5 minutes for full suite)
- **Reliable Results**: Tests should be deterministic and repeatable
- **Clear Purpose**: Each test should have a specific, documented purpose
- **Maintainable**: Tests should be easy to understand and modify

## 📁 Test Structure

```
tests/
├── adapters/                     # Model adapter tests
│   ├── test_custom_adapter.py
│   ├── test_huggingface_adapter.py
│   ├── test_smp_adapter.py
│   ├── test_torchvision_adapter.py
│   └── test_registry.py
├── attacks/                      # Attack implementation tests
│   ├── test_base_attack.py
│   ├── test_fgsm.py
│   ├── test_pgd.py
│   ├── test_rfgsm.py
│   ├── test_tpgd.py
│   ├── test_custom_attacks.py
│   └── test_registry.py
├── datasets/                     # Dataset tests
│   ├── test_voc_dataset.py
│   ├── test_ade20k_dataset.py
│   ├── test_cityscapes_dataset.py
│   ├── test_stanford_background_dataset.py
│   └── test_registry.py
├── loaders/                      # Model and dataset loader tests
│   ├── test_universal_model_loader.py
│   ├── test_torchvision_model_loader.py
│   ├── test_smp_model_loader.py
│   ├── test_huggingface_model_loader.py
│   ├── test_custom_model_loader.py
│   ├── test_dataset_loader.py
│   └── test_attack_loader.py
├── metrics/                      # Metric tests
│   ├── test_base_metrics.py
│   └── test_custom_metrics.py
├── pipeline/                     # Pipeline tests
│   ├── test_core.py
│   └── test_config.py
├── utils/                        # Utility function tests
│   ├── test_image_preprocessing.py
│   ├── test_model_utils.py
│   ├── test_dataset_utils.py
│   ├── test_loader_utils.py
│   └── test_visualization.py
└── data/                         # Test data files
    ├── dummy_model_weights.pth
    ├── dummy_model_checkpoint.pth
    └── dummy_encoder_weights.pth
```

## 🚀 Running Tests

### Basic Commands

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run with coverage
pytest --cov=segmentation_robustness_framework --cov-report=html

# Run specific test file
pytest tests/attacks/test_fgsm.py

# Run specific test function
pytest tests/attacks/test_fgsm.py::test_fgsm_attack

# Run tests matching pattern
pytest -k "attack"

# Run tests in parallel
pytest -n auto
```

### Test Categories

```bash
# Adapter tests only
pytest tests/adapters/

# Attack tests only
pytest tests/attacks/

# Dataset tests only
pytest tests/datasets/

# Loader tests only
pytest tests/loaders/

# Pipeline tests only
pytest tests/pipeline/
```

## 📝 Writing Tests

### Test Function Naming

```python
def test_function_name_expected_behavior():
    """Test that function behaves as expected."""
    pass

def test_function_name_edge_case():
    """Test function behavior with edge case input."""
    pass

def test_function_name_error_condition():
    """Test function behavior with invalid input."""
    pass
```

### Test Structure

```python
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
    assert torch.any(adv_images != images)  # Images should be perturbed
```

### Using Fixtures

```python
import pytest

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
    attack = FGSM(sample_model, eps=0.02)
    # Test implementation
```

### Parameterized Tests

```python
import pytest

@pytest.mark.parametrize("eps", [0.01, 0.02, 0.05])
def test_fgsm_different_eps_values(eps):
    """Test FGSM with different epsilon values."""
    model = create_mock_model()
    attack = FGSM(model, eps=eps)
    
    images = torch.randn(2, 3, 224, 224)
    labels = torch.randint(0, 21, (2, 224, 224))
    
    adv_images = attack.apply(images, labels)
    
    # Check perturbation magnitude
    perturbation = torch.abs(adv_images - images)
    assert torch.all(perturbation <= eps + 1e-6)
```

## 🔧 Test Utilities

### Mock Models

```python
class MockModel(nn.Module):
    """Mock model for testing."""
    
    def __init__(self, num_classes=21):
        super().__init__()
        self.num_classes = num_classes
    
    def forward(self, x):
        """Forward pass returning random logits."""
        batch_size = x.shape[0]
        height, width = x.shape[2], x.shape[3]
        return torch.randn(batch_size, self.num_classes, height, width)

def create_mock_model(num_classes=21):
    """Create a mock model for testing."""
    return MockModel(num_classes)
```

### Sample Data Generators

```python
def create_sample_images(batch_size=2, channels=3, height=224, width=224):
    """Create sample images for testing."""
    return torch.randn(batch_size, channels, height, width)

def create_sample_labels(batch_size=2, height=224, width=224, num_classes=21):
    """Create sample labels for testing."""
    return torch.randint(0, num_classes, (batch_size, height, width))

def create_sample_dataset(num_samples=10):
    """Create a sample dataset for testing."""
    class SampleDataset(torch.utils.data.Dataset):
        def __init__(self, num_samples):
            self.num_samples = num_samples
        
        def __len__(self):
            return self.num_samples
        
        def __getitem__(self, idx):
            image = create_sample_images(1, 3, 224, 224).squeeze(0)
            label = create_sample_labels(1, 224, 224).squeeze(0)
            return image, label
    
    return SampleDataset(num_samples)
```

### Assertion Helpers

```python
def assert_tensor_shape(tensor, expected_shape):
    """Assert tensor has expected shape."""
    assert tensor.shape == expected_shape, f"Expected {expected_shape}, got {tensor.shape}"

def assert_tensor_range(tensor, min_val=0, max_val=1):
    """Assert tensor values are within range."""
    assert torch.all(tensor >= min_val), f"Values below {min_val}"
    assert torch.all(tensor <= max_val), f"Values above {max_val}"

def assert_metric_value(metric_value, min_val=0, max_val=1):
    """Assert metric value is within valid range."""
    assert min_val <= metric_value <= max_val, f"Metric value {metric_value} outside range [{min_val}, {max_val}]"
```

## 🏷️ Test Markers

### Built-in Markers

```python
import pytest

@pytest.mark.slow
def test_slow_operation():
    """Test that takes a long time to run."""
    pass

@pytest.mark.gpu
def test_gpu_operation():
    """Test that requires GPU."""
    pass

@pytest.mark.integration
def test_integration():
    """Integration test."""
    pass

@pytest.mark.unit
def test_unit():
    """Unit test."""
    pass
```

### Custom Markers

Register custom markers in `pytest.ini`:

```ini
[tool:pytest]
markers =
    slow: marks tests as slow (deselect with '-m "not slow"')
    gpu: marks tests as requiring GPU
    integration: marks tests as integration tests
    unit: marks tests as unit tests
    benchmark: marks tests as performance benchmarks
```

## 📊 Coverage Reporting

### Coverage Configuration

```ini
[tool:pytest]
addopts = --cov=segmentation_robustness_framework --cov-report=html --cov-report=term-missing
testpaths = tests
python_files = test_*.py
python_classes = Test*
python_functions = test_*
```

### Coverage Targets

- **Overall Coverage**: >90%
- **Critical Paths**: >95%
- **New Code**: >95%

## 🚨 Debugging Tests

### Common Issues

#### Test Isolation

```python
# Use fixtures for proper setup/teardown
@pytest.fixture(autouse=True)
def reset_state():
    """Reset global state before each test."""
    # Setup
    yield
    # Teardown
    torch.cuda.empty_cache()
```

#### Deterministic Results

```python
# Set random seeds for reproducible tests
@pytest.fixture(autouse=True)
def set_random_seed():
    """Set random seed for reproducible tests."""
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)
```

#### Memory Issues

```python
# Monitor memory usage
def test_memory_usage():
    """Test with memory monitoring."""
    import psutil
    process = psutil.Process()
    
    initial_memory = process.memory_info().rss
    # Run test
    final_memory = process.memory_info().rss
    
    print(f"Memory usage: {final_memory - initial_memory} bytes")
```

### Debugging Commands

```bash
# Run single test with debug output
pytest tests/attacks/test_fgsm.py::test_fgsm_attack -v -s

# Run with print statements
pytest tests/attacks/test_fgsm.py::test_fgsm_attack -s

# Run with pdb debugger
pytest tests/attacks/test_fgsm.py::test_fgsm_attack --pdb

# Run with detailed error information
pytest tests/attacks/test_fgsm.py::test_fgsm_attack -vvv
```

## 🎯 Best Practices

### Test Design

1. **Test One Thing**: Each test should verify one specific behavior
2. **Use Descriptive Names**: Test names should clearly describe what they test
3. **Arrange-Act-Assert**: Structure tests with clear sections
4. **Use Fixtures**: Reuse common setup code
5. **Test Edge Cases**: Include boundary conditions and error cases

### Test Maintenance

1. **Keep Tests Fast**: Avoid slow operations in unit tests
2. **Use Mocks**: Mock external dependencies
3. **Update Tests**: Update tests when code changes
4. **Review Coverage**: Regularly review and improve coverage
5. **Document Complex Tests**: Add comments for complex test logic

### Test Data

1. **Use Small Data**: Use minimal data for fast tests
2. **Create Realistic Data**: Use data that resembles real usage
3. **Avoid External Dependencies**: Don't rely on external services
4. **Version Control Test Data**: Include test data in repository
5. **Document Data Format**: Document expected data formats

## 🚀 Next Steps

After reading this testing guide:

1. **Set up your testing environment** following the [Development Setup Guide](development_setup.md)
2. **Write your first test** using the examples above
3. **Run the test suite** to ensure everything works
4. **Contribute tests** for new features or bug fixes
5. **Review existing tests** to understand patterns and conventions

Happy testing! 🧪
