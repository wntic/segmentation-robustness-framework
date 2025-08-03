# Installation

This guide will help you install the Segmentation Robustness Framework and its dependencies. ⚙️

## 📋 Prerequisites

Before installing the framework, ensure you have:

- **Python 3.12+** (3.12, 3.13)
- **PyTorch 2.6.0** with CUDA support (for GPU acceleration)

## 🚀 Installation Methods

### 📦 Method 1: Install from PyPI (Recommended)

```bash
pip install segmentation-robustness-framework
```

### 🔧 Method 2: Install from Source

```bash
# Clone the repository
git clone https://github.com/wntic/segmentation-robustness-framework.git
cd segmentation-robustness-framework

# Create and activate virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode
pip install -e .
```

### 📚 Method 3: Install with Extra Dependencies

The framework provides several optional dependency groups for different use cases:

#### 🎯 Install All Extras

For full functionality including all optional dependencies:

```bash
pip install "segmentation-robustness-framework[full]"
```

#### 🔧 Install Specific Extras

**🏗️ SMP Models Support** - For [Segmentation Models PyTorch](https://github.com/qubvel-org/segmentation_models.pytorch):

```bash
pip install "segmentation-robustness-framework[smp]"
```

**🤗 Transformers Support** - For HuggingFace models:

```bash
pip install "segmentation-robustness-framework[transformers]"
```

#### 📋 Available Extras

- **`full`** - Includes all optional dependencies (SMP + Transformers)
- **`smp`** - Segmentation Models PyTorch for additional model architectures
- **`transformers`** - HuggingFace Transformers for transformer-based models

#### 📚 What Each Extra Includes

**SMP Extra (`[smp]`)**:
- `segmentation-models-pytorch` - UNet, LinkNet, FPN, PSPNet, and more architectures
- Additional model backbones and decoder heads

**Transformers Extra (`[transformers]`)**:
- `transformers` - HuggingFace Transformers library
- `tokenizers` - Fast tokenization
- Support for SegFormer, Mask2Former, and other transformer-based models

**All Extra (`[full]`)**:
- Combines both SMP and Transformers extras
- Full model support across all available architectures

## 📦 Dependencies

### 🔧 Core Dependencies

The framework requires these core dependencies:

- **PyTorch** (2.6.0) - Deep learning framework
- **torchvision** (0.21.0) - Computer vision utilities
- **numpy** (≥2.3.0) - Numerical computing
- **matplotlib** (≥3.10.0) - Plotting and visualization
- **PyYAML** (≥6.0.2) - Configuration file parsing
- **tqdm** (≥4.67.1) - Progress bars
- **pandas** (≥2.3.1) - Data manipulation

### 📚 Optional Dependencies

These are installed automatically with the corresponding extras:

- **segmentation-models-pytorch** (≥0.5.0) - Additional segmentation models
- **transformers** (≥4.53.1) - HuggingFace model support

## GPU Support

For GPU acceleration, install PyTorch with CUDA support:

```bash
# For CUDA 11.8
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu121

# For CUDA 12.4
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cu124

# For CPU only
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
```

### Checking GPU Support

After installation, verify GPU support:

```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"Current device: {torch.cuda.current_device()}")
    print(f"Device name: {torch.cuda.get_device_name()}")
```

## Verification

After installation, verify that everything works:

```bash
# Test basic import
python -c "import segmentation_robustness_framework; print('Installation successful!')"

# Test CLI
python -m segmentation_robustness_framework.cli.main --help

# List available components
python -m segmentation_robustness_framework.cli.main list

# Run tests
python -m segmentation_robustness_framework.cli.main test
```

### Quick Test

Run a minimal test to verify the framework works:

```python
from segmentation_robustness_framework.pipeline import PipelineConfig

# Test configuration
config = {
    "model": {
        "type": "torchvision",
        "config": {"name": "deeplabv3_resnet50", "num_classes": 21}
    },
    "dataset": {
        "name": "voc",
        "split": "val",
        "root": "./data",
        "image_shape": [256, 256],
        "download": True
    },
    "attacks": [{"name": "fgsm", "eps": 0.02}],
    "pipeline": {
        "batch_size": 2,
        "device": "cpu",
        "output_dir": "./test_run",
        "auto_resize_masks": True,
        "output_formats": ["json"]
    },
    "metrics": {
        "ignore_index": 255,
        "selected_metrics": ["mean_iou", "pixel_accuracy"]
    }
}

# Create pipeline config (this tests the configuration system)
pipeline_config = PipelineConfig.from_dict(config)
print("Configuration loaded successfully!")
```

## Troubleshooting

### Common Issues

#### 1. Import Errors

If you encounter import errors:

```bash
# Check if the package is installed
pip list | grep segmentation-robustness-framework

# Reinstall if needed
pip uninstall segmentation-robustness-framework
pip install segmentation-robustness-framework
```

#### 2. CUDA Issues

If CUDA is not working:

```bash
# Check CUDA availability
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"

# Install CPU version if needed
pip install torch==2.6.0 torchvision==0.21.0 --index-url https://download.pytorch.org/whl/cpu
```

#### 3. Missing Dependencies

If you get missing dependency errors:

```bash
# Install with extras (recommended)
pip install "segmentation-robustness-framework[full]"

# Or install from source with all dependencies
git clone https://github.com/wntic/segmentation-robustness-framework.git
cd segmentation-robustness-framework
pip install -e ".[full]"
```

#### 4. Virtual Environment Issues

If you're having issues with virtual environments:

```bash
# Create a fresh virtual environment
python -m venv fresh_env
source fresh_env/bin/activate  # On Windows: fresh_env\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install the framework
pip install segmentation-robustness-framework[full]
```

#### 5. Permission Issues

If you encounter permission errors:

```bash
# Use --user flag for user installation
pip install --user segmentation-robustness-framework

# Or use virtual environment (recommended)
python -m venv .venv
source .venv/bin/activate
pip install segmentation-robustness-framework
```

### Getting Help

If you encounter issues:

1. Check the [troubleshooting section](user_guide.md#troubleshooting) in the User Guide
2. Search existing [GitHub issues](https://github.com/wntic/segmentation-robustness-framework/issues)
3. Create a new issue with detailed error information including:
   - Python version
   - PyTorch version
   - Operating system
   - Full error traceback

## Next Steps

Once installation is complete:

1. [Quick Start Guide](quick_start.md) - Get up and running in 5 minutes
2. [User Guide](user_guide.md) - Learn how to use the framework
3. [API Reference](api_reference/index.md) - Explore the complete API
4. [Basic Examples](examples/basic_examples.md) - Try out the framework with examples

## Development Setup

For contributors and developers:

```bash
# Clone the repository
git clone https://github.com/wntic/segmentation-robustness-framework.git
cd segmentation-robustness-framework

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install in development mode with all extras
pip install -e ".[full]"

# Install development dependencies
pip install -e ".[full,test,lint,docs]"

# Run tests
python -m segmentation_robustness_framework.cli.main test

# Build documentation
mkdocs build
mkdocs serve
```
