[![Python](https://img.shields.io/badge/python-3.12-blue.svg?logo=python)](https://www.python.org/downloads/release/python-3120/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.6.0-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![PyPI version](https://img.shields.io/pypi/v/segmentation-robustness-framework.svg?logo=pypi)](https://pypi.org/project/segmentation-robustness-framework/)
[![Docs](https://readthedocs.org/projects/segmentation-robustness-framework/badge/?version=latest)](https://segmentation-robustness-framework.readthedocs.io/en/latest/)
[![Tests](https://img.shields.io/badge/tests-passing-brightgreen.svg?logo=pytest)](https://github.com/wntic/segmentation-robustness-framework/actions)
[![Ruff](https://img.shields.io/badge/code%20style-ruff-005A9C?logo=ruff&logoColor=white)](https://docs.astral.sh/ruff/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

# Segmentation Robustness Framework

A comprehensive framework for evaluating the robustness of semantic segmentation models against adversarial attacks. 🛡️

**Evaluate your models' security with state-of-the-art adversarial attacks and comprehensive metrics!**

## 🚀 Quick Start

### 💻 Command Line Interface

The framework provides convenient CLI commands for common tasks:

```bash
# Install the package
pip install segmentation-robustness-framework

# List available components
srf list --attacks
srf list --models
srf list --datasets

# Run pipeline from configuration
srf run config.yaml

# Run tests
srf test
```

### 🐍 Python API

Get started in minutes with our comprehensive examples:

```python
from segmentation_robustness_framework.pipeline import SegmentationRobustnessPipeline
from segmentation_robustness_framework.metrics import MetricsCollection
from segmentation_robustness_framework.attacks import FGSM
from segmentation_robustness_framework.datasets import VOCSegmentation
from segmentation_robustness_framework.loaders.models.universal_loader import UniversalModelLoader
from segmentation_robustness_framework.utils import image_preprocessing
import torch

# Load model with universal loader
loader = UniversalModelLoader()
model = loader.load_model(
    model_type="torchvision",
    model_config={"name": "deeplabv3_resnet50", "num_classes": 21}
)

# Set device and move model to it (IMPORTANT: Do this before creating attacks!)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Setup dataset with preprocessing
preprocess, target_preprocess = image_preprocessing.get_preprocessing_fn(
    [512, 512], dataset_name="voc"
)
dataset = VOCSegmentation(
    split="val", 
    root="path/to/existing/VOCdevkit/VOC2012/",
    transform=preprocess,
    target_transform=target_preprocess
)

# Setup attack and metrics (attacks will use the same device as the model)
attack = FGSM(model, eps=2/255)

# Setup metrics
metrics_collection = MetricsCollection(num_classes=21)
metrics = [metrics_collection.mean_iou, metrics_collection.pixel_accuracy]

# Create and run pipeline
pipeline = SegmentationRobustnessPipeline(
    model=model,
    dataset=dataset,
    attacks=[attack],
    metrics=metrics,
    batch_size=4,
    device=device
)

results = pipeline.run(save=True, show=True)
pipeline.print_summary()
```

## 📚 Documentation Structure

### 🚀 **Getting Started**
- **[Installation Guide](https://segmentation-robustness-framework.readthedocs.io/latest/installation)** - Setup and installation instructions
- **[Quick Start Guide](https://segmentation-robustness-framework.readthedocs.io/latest/quick_start)** - Your first evaluation in 5 minutes
- **[Core Concepts](https://segmentation-robustness-framework.readthedocs.io/latest/core_concepts)** - Understanding the framework architecture

### 📖 **User Guides**
- **[User Guide](https://segmentation-robustness-framework.readthedocs.io/latest/user_guide)** - Complete guide for using the framework
- **[Configuration Guide](https://segmentation-robustness-framework.readthedocs.io/latest/configuration_guide)** - How to write configuration files
- **[Examples](https://segmentation-robustness-framework.readthedocs.io/latest/examples/basic_examples)** - Real-world examples and use cases

### 🔧 **Technical Reference**
- **[API Reference](https://segmentation-robustness-framework.readthedocs.io/latest/api_reference)** - Complete API documentation
- **[Contributing Guide](https://segmentation-robustness-framework.readthedocs.io/latest/contributing)** - How to contribute to the project

### 🎓 **Learning Path**

1. **Start Here**: [Installation Guide](https://segmentation-robustness-framework.readthedocs.io/latest/installation) → [Quick Start](https://segmentation-robustness-framework.readthedocs.io/latest/quick_start)
2. **Basic Usage**: [User Guide](https://segmentation-robustness-framework.readthedocs.io/latest/user_guide)
3. **Configuration**: [Configuration Guide](https://segmentation-robustness-framework.readthedocs.io/latest/configuration_guide)
4. **Advanced Usage**: [Advanced Examples](https://segmentation-robustness-framework.readthedocs.io/latest/examples/advanced_examples)

## 🎯 **Key Features**

### **🔬 Comprehensive Evaluation**
- **Multiple Attacks**: FGSM, PGD, RFGSM, TPGD, and custom attacks
- **Rich Metrics**: IoU, pixel accuracy, precision, recall, dice score
- **Flexible Output**: JSON, CSV, and custom formats
- **Batch Processing**: Efficient evaluation of large datasets
- **Performance Optimization**: GPU acceleration and memory management

### **🤖 Universal Model Support**
- **Torchvision Models**: DeepLab, FCN, LRASPP architectures
- **SMP Models**: UNet, LinkNet, PSPNet, and more
- **HuggingFace Models**: Transformers-based segmentation models
- **Custom Models**: Easy integration with your own models via adapters

### **📊 Built-in Datasets**
- **VOC**: Pascal VOC 2012 (21 classes)
- **ADE20K**: Scene parsing dataset (150 classes)
- **Cityscapes**: Urban scene understanding (35 classes)
- **Stanford Background**: Natural scene dataset (9 classes)

### **⚡ Easy Integration**
- **Registry System**: Automatic discovery of custom components
- **Adapter Pattern**: Standardized model interfaces
- **Preprocessing Pipeline**: Automatic data normalization and conversion
- **Configuration System**: YAML/JSON-based pipeline configuration
- **Error Handling**: Comprehensive error messages and debugging

## 🚀 **Quick Examples**

### **⚡ Basic Evaluation**
```python
from segmentation_robustness_framework.pipeline import SegmentationRobustnessPipeline
from segmentation_robustness_framework.metrics import MetricsCollection
from segmentation_robustness_framework.attacks import FGSM

# Setup pipeline
pipeline = SegmentationRobustnessPipeline(
    model=model,
    dataset=dataset,
    attacks=[FGSM(model, eps=2/255)],
    metrics=[metrics.mean_iou, metrics.pixel_accuracy],
    batch_size=4,
    device="cuda"
)

# Run evaluation
results = pipeline.run()
print(f"Clean IoU: {results['clean']['mean_iou']:.3f}")
print(f"Attack IoU: {results['attack_fgsm']['mean_iou']:.3f}")
```

### **⚙️ Configuration-Based Evaluation**
```yaml
# config.yaml
model:
  type: torchvision
  config:
    name: deeplabv3_resnet50
    num_classes: 21

dataset:
  name: voc
  split: val
  image_shape: [512, 512]

attacks:
  - name: fgsm
    eps: 0.02
  - name: pgd
    eps: 0.02
    alpha: 0.01
    iters: 10

metrics:
  - mean_iou
  - pixel_accuracy
```

```bash
# Run from configuration
srf run config.yaml
```

### **🔧 Custom Components**
```python
# Custom Attack
@register_attack("my_attack")
class MyAttack(AdversarialAttack):
    def apply(self, images, labels):
        # Implement attack logic
        return adversarial_images

# Custom Dataset
@register_dataset("my_dataset")
class MyDataset(Dataset):
    def __init__(self, root, transform=None):
        self.num_classes = 5
        # ... implementation
    
    def __getitem__(self, idx):
        return image, mask
```

## 🛠️ **Installation**

```bash
# Install from PyPI (recommended)
pip install segmentation-robustness-framework

# Install with all optional dependencies
pip install "segmentation-robustness-framework[full]"

# Or install from source
git clone https://github.com/wntic/segmentation-robustness-framework
cd segmentation-robustness-framework
pip install -e .
```

## 🏗️ **Framework Architecture**

The framework follows a modular architecture with clear separation of concerns:

- **Pipeline**: Core orchestration component
- **Model Loaders**: Universal model loading system
- **Adapters**: Standardized model interfaces
- **Attacks**: Adversarial attack implementations
- **Datasets**: Dataset loading and preprocessing
- **Metrics**: Evaluation metrics and scoring
- **Registry**: Component discovery and registration

## 🤝 **Contributing**

We welcome contributions! Please see our [Contributing Guide](https://segmentation-robustness-framework.readthedocs.io/latest/contributing) for details on:

- 🐛 **Bug Reports** - Help us identify and fix issues
- 💡 **Feature Requests** - Suggest new features or improvements
- 📝 **Documentation** - Improve our docs and examples
- 🔧 **Code Contributions** - Add new models, attacks, metrics, or datasets
- 🧪 **Testing** - Help ensure code quality and reliability

## 📞 **Support**

- 📖 **Documentation**: Browse the guides above
- 📋 **Changelog**: See what's new in [CHANGELOG.md](CHANGELOG.md)
- 🐛 **Issues**: Report bugs and request features on [GitHub](https://github.com/wntic/segmentation-robustness-framework/issues)
- 💬 **Discussions**: Join our community discussions
- 📧 **Contact**: Reach out to the maintainers

## 📄 **License**

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file in the project root for details.

---

**Ready to evaluate your segmentation models?** 🚀

Start with our [Quick Start Guide](https://segmentation-robustness-framework.readthedocs.io/latest/quick_start) and have your first evaluation running in minutes!

**🎯 Key Benefits:**
- ⚡ **Fast Setup** - Get running in 5 minutes
- 🔧 **Easy Configuration** - YAML/JSON-based configuration
- 🤖 **Universal Support** - Works with any PyTorch segmentation model
- 📊 **Comprehensive Metrics** - Rich evaluation metrics
- 🛡️ **Security Focused** - State-of-the-art adversarial attacks
