# Development Setup Guide

This guide provides detailed instructions for setting up a development environment for the Segmentation Robustness Framework.

## 🖥️ System Requirements

### Operating Systems

- **Linux** (Ubuntu 20.04+, CentOS 7+)
- **macOS** (10.15+)
- **Windows** (10/11 with WSL2 recommended)

### Hardware Requirements

- **CPU**: 4+ cores recommended
- **RAM**: 8GB minimum, 16GB+ recommended
- **GPU**: NVIDIA GPU with CUDA support (optional but recommended)
- **Storage**: 15GB+ free space

### Software Requirements

- **Python**: 3.12+
- **Git**

## 🚀 Installation Methods

### Method 1: Direct Installation (Recommended)

```bash
# Clone the repository
git clone https://github.com/wntic/segmentation-robustness-framework.git
cd segmentation-robustness-framework

# Create virtual environment
python -m venv .venv

# Activate virtual environment
# On Linux/macOS:
source .venv/bin/activate

# Install in development mode
pip install -e ".[dev]"

# Verify installation
python -c "import segmentation_robustness_framework; print('Installation successful!')"
```

### Method 2: Using Poetry

```bash
# Install Poetry (if not already installed)
curl -sSL https://install.python-poetry.org | python3 -

# Clone and setup
git clone https://github.com/wntic/segmentation-robustness-framework.git
cd segmentation-robustness-framework

# Install dependencies
poetry install

# Activate virtual environment
poetry shell

# Verify installation
python -c "import segmentation_robustness_framework; print('Installation successful!')"
```

## 🔧 Development Tools Setup

### Code Quality Tools

```bash
# Install pre-commit hooks
pip install pre-commit
pre-commit install

# Install additional development tools
pip install black isort pytest-cov

# Configure git hooks
pre-commit install --hook-type pre-commit --hook-type pre-push
```

### GPU Setup (Optional)

#### NVIDIA GPU with CUDA

```bash
# Check CUDA availability
nvidia-smi

# Install PyTorch with CUDA support
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# Verify CUDA support
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### Apple Silicon (M Series)

```bash
# Install PyTorch for Apple Silicon
pip install torch torchvision

# Verify MPS support
python -c "import torch; print(f'MPS available: {torch.backends.mps.is_available()}')"
```

## 🧪 Testing Setup

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=segmentation_robustness_framework --cov-report=html

# Run specific test categories
pytest tests/adapters
pytest tests/attacks

# Run with verbose output
pytest -v
```

## 📦 Package Management

### Dependency Management

```bash
# Update dependencies
pip install --upgrade -r requirements.txt

# Check for outdated packages
pip list --outdated

# Security audit
pip-audit
```

### Virtual Environment Management

```bash
# Create new environment
python -m venv .venv-new

# Activate environment
source .venv-new/bin/activate

# Install dependencies
pip install -e ".[dev]"

# Deactivate environment
deactivate
```

## 🔧 Troubleshooting

### Common Issues

#### Import Errors

```bash
# Check Python path
python -c "import sys; print(sys.path)"

# Add project to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### CUDA Issues

```bash
# Check CUDA installation
nvcc --version

# Check PyTorch CUDA support
python -c "import torch; print(torch.version.cuda)"

# Reinstall PyTorch with correct CUDA version
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

#### Memory Issues

```bash
# Monitor memory usage
htop

# Reduce batch size in tests
export SRF_TEST_BATCH_SIZE=1

# Use CPU for testing
export SRF_TEST_DEVICE=cpu
```

### Getting Help

1. **Check existing issues** on GitHub
2. **Search documentation** thoroughly
3. **Ask in discussions** with detailed error messages
4. **Provide minimal reproduction** steps

## 🚀 Next Steps

After completing the development setup:

1. **Read the [Contributing Guide](index.md)** for contribution guidelines
2. **Explore the [Code Style Guide](code_style.md)** for coding standards
3. **Check out [Testing Guidelines](testing_guide.md)** for testing practices
4. **Review [Documentation Guidelines](documentation_guide.md)** for doc standards

Happy coding! 🎉
