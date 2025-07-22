# Installation Guide

This guide will help you install the Segmentation Robustness Framework and all its dependencies.

## 📋 Prerequisites

### System Requirements
- **Python**: 3.10 or higher
- **CUDA**: 11.0 or higher (for GPU acceleration)
- **Memory**: At least 8GB RAM (16GB+ recommended)
- **Storage**: At least 5GB free space for datasets

### Operating Systems
- ✅ **Linux** (Ubuntu 20.04+, CentOS 7+)
- ✅ **macOS** (10.15+)
- ⚠️ **Windows** (WSL2 recommended)

## 🚀 Quick Installation

### Option 1: Install from PyPI (Recommended)

```bash
# Install the framework
pip install segmentation-robustness-framework

# Install optional dependencies for full functionality
pip install segmentation-robustness-framework[full]
```

### Option 2: Install from Source

```bash
# Clone the repository
git clone https://github.com/your-repo/segmentation-robustness-framework
cd segmentation-robustness-framework

# Install in development mode
pip install -e .

# Install development dependencies
pip install -e ".[dev]"
```

## 📦 Dependencies

### Core Dependencies
The framework automatically installs these core dependencies:

```python
# Core ML libraries
torch>=1.12.0
torchvision>=0.13.0
numpy>=1.21.0
Pillow>=8.0.0

# Data processing
pandas>=1.3.0
tqdm>=4.62.0

# Visualization
matplotlib>=3.5.0

# Configuration
pydantic>=1.9.0
PyYAML>=6.0
```

### Optional Dependencies

Install additional dependencies for extended functionality:

```bash
# For HuggingFace models
pip install transformers>=4.20.0

# For SMP models
pip install segmentation-models-pytorch>=0.3.0

# For development
pip install pytest>=7.0.0
pip install ruff>=0.5.7
```

### GPU Support

For CUDA acceleration, install PyTorch with CUDA support:

```bash
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# For CPU only
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

## 🔧 Environment Setup

### Using Conda (Recommended)

```bash
# Create a new conda environment
conda create -n seg-robust python=3.10
conda activate seg-robust

# Install PyTorch with CUDA
conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

# Install the framework
pip install segmentation-robustness-framework[full]
```

### Using Virtual Environment

```bash
# Create virtual environment
python -m venv seg-robust-env
source seg-robust-env/bin/activate  # On Windows: seg-robust-env\Scripts\activate

# Upgrade pip
pip install --upgrade pip

# Install the framework
pip install segmentation-robustness-framework[full]
```

## ✅ Verification

### Test Installation

Create a test script to verify your installation:

```python
# test_installation.py
import torch
from segmentation_robustness_framework.engine.pipeline import SegmentationRobustnessPipeline
from segmentation_robustness_framework.utils.metrics import MetricsCollection
from segmentation_robustness_framework.attacks import FGSM

print("✅ PyTorch version:", torch.__version__)
print("✅ CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("✅ CUDA version:", torch.version.cuda)

# Test framework imports
print("✅ Framework imported successfully")

# Test basic functionality
try:
    metrics = MetricsCollection(num_classes=21)
    print("✅ Metrics collection created")
    
    # Create a simple model for testing
    model = torch.nn.Conv2d(3, 21, kernel_size=3, padding=1)
    attack = FGSM(model, eps=2/255)
    print("✅ Attack created")
    
    print("🎉 Installation successful!")
except Exception as e:
    print(f"❌ Installation test failed: {e}")
```

Run the test:

```bash
python test_installation.py
```

### Test GPU Support

```python
# test_gpu.py
import torch
from segmentation_robustness_framework.utils.model_utils import get_model_output_size

# Create a simple model
model = torch.nn.Conv2d(3, 10, kernel_size=3, padding=1)

# Test on CPU
output_size_cpu = get_model_output_size(model, (3, 224, 224), "cpu")
print(f"✅ CPU test passed: {output_size_cpu}")

# Test on GPU if available
if torch.cuda.is_available():
    output_size_gpu = get_model_output_size(model, (3, 224, 224), "cuda")
    print(f"✅ GPU test passed: {output_size_gpu}")
else:
    print("⚠️  CUDA not available, skipping GPU test")
```

## 🔄 Updating

### Update the Framework

```bash
# Update to latest version
pip install --upgrade segmentation-robustness-framework

# Update from source
cd segmentation-robustness-framework
git pull
pip install -e .
```

### Update Dependencies

```bash
# Update all dependencies
pip install --upgrade -r requirements.txt

# Update specific packages
pip install --upgrade torch torchvision
```

## 📚 Next Steps

After successful installation:

1. **Quick Start**: Follow the [Quick Start Guide](quickstart.md)
2. **Basic Usage**: Read the [User Guide](user_guide.md)
3. **Examples**: Check the [Practical Example](practical_example.md) and [Custom Datasets Guide](custom_datasets_guide.md)

## 🆘 Getting Help

If you encounter issues:

1. **Search existing issues** on GitHub

2. **Create a new issue** with:

   - Your operating system and Python version
   - Complete error message
   - Steps to reproduce the issue
   - Output of `python test_installation.py`

---

**Ready to get started?** 🚀

Proceed to the [Quick Start Guide](quickstart.md) to run your first evaluation! 