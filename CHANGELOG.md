# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.3.0] - 2025-04-08

### 🚀 Added

#### **Configuration System**

- **New configuration system** - Pipeline can now be built using configuration files
- **Multiple format support** - JSON, YAML, and Python dictionary configurations
- **Single command execution** - Run complete evaluations with one command and config file
- **Flexible configuration** - Support for all pipeline components via config files

#### **Command Line Interface (CLI)**

- **Full-fledged CLI implementation** - Complete terminal-based framework management
- **Automatic CLI shortcuts** - `srf`, `srf-test`, and other shortcuts installed automatically
- **Component listing** - CLI function for printing available metrics, models, and attacks
- **Configuration-based execution** - Run evaluations directly from config files

#### **Metrics System**

- **Custom metrics registry** - Users can implement and register their own metrics
- **New metrics module** - Basic and custom metrics moved from `utils` to dedicated `metrics` module
- **Enhanced metric system** - Better organization and extensibility for evaluation metrics
- **CLI metric listing** - List available metrics through command line interface

#### **Performance Optimizations**

- **HuggingFace model optimization** - Device synchronization and memory cleaning for better performance
- **Attack method optimization** - Device synchronization and memory cleaning for built-in attacks
- **Memory management** - Improved memory handling across all components

#### **Documentation**

- **Complete documentation rewrite** - Comprehensive new documentation system
- **Multiple guides** - Installation, Quick Start, User Guide, Configuration Guide
- **API reference** - Complete API documentation with examples
- **Contributing guides** - Development setup, testing, code style, and release guides
- **Examples** - Basic and advanced usage examples
- **Troubleshooting guide** - Common issues and solutions

### 🔧 Changed

#### **Device Management**

- **Default device change** - Pipeline now uses `CPU` by default instead of `CUDA`
- **Improved device handling** - Better device management across all components

#### **Architecture**

- **Removed model.py config** - Simplified configuration system
- **Metrics reorganization** - Moved from `utils` to dedicated `metrics` module
- **Better module structure** - Improved code organization and maintainability

### 🐛 Fixed

#### **Testing**

- **Test fixes** - Comprehensive test suite improvements and bug fixes
- **Test coverage** - Enhanced testing across all components
- **Test reliability** - More robust and reliable test execution

### 📚 Documentation

#### **New Documentation Structure**

- **Installation Guide** - Complete setup instructions
- **Quick Start Guide** - Get running in 5 minutes
- **User Guide** - Comprehensive usage instructions
- **Configuration Guide** - Complete configuration system documentation
- **Core Concepts** - Framework architecture and design principles
- **API Reference** - Complete API documentation
- **Examples** - Basic and advanced usage examples
- **Contributing Guide** - Development setup and contribution guidelines
- **Troubleshooting Guide** - Common issues and solutions

#### **Documentation Features**

- **MkDocs integration** - Modern documentation site
- **API auto-generation** - Automatic API docs from docstrings
- **Search functionality** - Full-text search across documentation
- **Responsive design** - Mobile-friendly documentation
- **Code examples** - Comprehensive code examples throughout

### 🔄 Migration Guide

#### **For Existing Users**

1. **Configuration Changes**
   - Remove any `model.py` configurations
   - Use new YAML/JSON configuration system
   - Update pipeline initialization to use new configuration format

2. **Device Management**
   - Explicitly specify `device="cuda"` if GPU is required
   - Pipeline now defaults to CPU for better compatibility

3. **Metrics Usage**
   - Import metrics from `metrics` module instead of `utils`
   - Use new metrics registry for custom metrics
   - Update metric function calls to new API

4. **CLI Usage**
   - Use new `srf` command for framework operations
   - Use `srf list` to see available components
   - Use `srf run config.yaml` for configuration-based execution

### 🎯 Key Improvements

- **Easier Setup** - Configuration-based pipeline building
- **Better Performance** - Optimized memory and device management
- **Enhanced Extensibility** - Custom metrics and components
- **Comprehensive Documentation** - Complete guides and examples
- **Professional CLI** - Full terminal-based management
- **Improved Testing** - More reliable and comprehensive tests

---

## [0.2.0] - 2025-07-27

### 🚀 Added

#### **Custom Model Support**

- **Multiple custom segmentation models** - Support for custom models via adapter pattern
- **Custom adapter registration** - Register adapters with names starting with `custom_` (e.g., `custom_seg_adapter`)
- **Direct adapter passing** - Pass adapter class directly to `load_model` method
- **Flexible model loading** - Support for both registered and direct adapter usage

#### **Enhanced Model Loading**

- **Universal model loader improvements** - Better support for custom models
- **Adapter class specification** - Override registered adapters with custom classes
- **Model wrapping system** - Automatic model wrapping with specified adapters

#### **Transformers Integration**

- **Expanded task support** - Added `panoptic_segmentation` and `image_segmentation` to existing tasks
- **Explicit model class specification** - Specify model name and class name in config
- **Enhanced model variety** - Broader range of available transformer models
- **Encoder weight loading** - Support for loading encoder weights into transformer models

### 🔧 Changed

#### **Model Loading System**

- **Default weight specification** - Changed from uppercase `"DEFAULT"` to `"default"` for torchvision models
- **Improved adapter system** - Better adapter registration and usage patterns
- **Enhanced SMP integration** - Fixed encoder weight loading in SMP models

### 🐛 Fixed

#### **Model Loading Issues**

- **SMP encoder weights** - Fixed bug with loading encoder weights in SMP models
- **Transformers model loading** - Fixed limited model availability in transformers
- **Model class specification** - Improved explicit model class loading

### 📝 Code Examples

#### **Custom Model Implementation**

```python
import torch
import torch.nn as nn
from segmentation_robustness_framework.adapters.registry import register_adapter
from segmentation_robustness_framework.adapters import CustomAdapter

@register_adapter("custom_seg")
class CustomSegmentationAdapter(CustomAdapter):    
    def __init__(self, model: nn.Module, num_classes: int = 21):
        super().__init__(model, num_classes)
        self.model = model
        self.num_classes = num_classes
    
    def logits(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)
    
    def predictions(self, x: torch.Tensor) -> torch.Tensor:
        logits = self.logits(x)
        return torch.argmax(logits, dim=1)

# Usage with registered adapter
loader = UniversalModelLoader()
model = loader.load_model(
    model_type="custom_seg",
    model_config={
        "model_class": SimpleSegmentationModel,
        "model_kwargs": {"num_classes": 21},
    }
)

# Usage with direct adapter
model = loader.load_model(
    model_type="custom",
    model_config={
        "model_class": SimpleSegmentationModel,
        "model_kwargs": {"num_classes": 21},
    },
    adapter_cls=CustomSegmentationAdapter
)
```

#### **Transformers Model Loading**

```python
# Standard loading with task specification
model_config = {
    "model_name": "nvidia/segformer-b0-finetuned-ade-512-512",
    "task": "semantic_segmentation"
}

# Explicit model class specification
model_config = {
    "model_name": "nvidia/segformer-b0-finetuned-ade-512-512",
    "model_class": "SegformerForSemanticSegmentation",
    "task": "semantic_segmentation"
}

# Encoder weight loading
model_config = {
    "model_name": "nvidia/segformer-b0-finetuned-ade-512-512",
    "weight_type": "encoder"
}
```

---

## [0.1.0] - 2025-07-20

### 🚀 Added

- Initial release
- Basic segmentation model evaluation
- Simple adversarial attack support
- Core pipeline functionality

---

*For more detailed information about each version, please refer to the [GitHub releases](https://github.com/wntic/segmentation-robustness-framework/releases).*
