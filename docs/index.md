# Segmentation Robustness Framework

Welcome to the **Segmentation Robustness Framework** - a comprehensive toolkit for evaluating and improving the robustness of semantic segmentation models against adversarial attacks.

## 🎯 What is this framework?

The Segmentation Robustness Framework provides a unified, extensible platform for:

- **Evaluating model robustness** against various adversarial attacks
- **Comparing different segmentation models** across multiple datasets
- **Standardized benchmarking** with reproducible results
- **Easy integration** of custom models, attacks, and metrics
- **Comprehensive reporting** with detailed analysis

## 🚀 Quick Start

Get started in minutes with our comprehensive quick start guide:

```bash
# Install the framework
pip install segmentation-robustness-framework

# Run your first evaluation
python -m segmentation_robustness_framework.cli.main run config.yaml
```

[Get Started →](quick_start.md){ .md-button .md-button--primary }

## 📚 Documentation

- 🚀 **[Quick Start](quick_start.md)** - Get up and running in 5 minutes
- 📖 **[User Guide](user_guide.md)** - Comprehensive usage guide
- 🔧 **[API Reference](api_reference/index.md)** - Complete API documentation
- 🧠 **[Core Concepts](core_concepts.md)** - Understanding the framework architecture
- ⚙️ **[Configuration Guide](configuration_guide.md)** - How to write configuration files
- 🤝 **[Contributing Guide](contributing/index.md)** - How to contribute to the project

## 🔧 Key Features

### 🎯 **Unified Pipeline**

- Single configuration file for complete experiments
- Automatic model loading and preprocessing
- Built-in attack generation and evaluation

### 🛡️ **Comprehensive Attacks**

- **FGSM** - Fast Gradient Sign Method
- **PGD** - Projected Gradient Descent
- **RFGSM** - R-FGSM with momentum
- **TPGD** - Two-Phase Gradient Descent
- Easy integration of custom attacks

### 📊 **Rich Metrics**

- **Mean IoU** - Intersection over Union
- **Pixel Accuracy** - Overall accuracy
- **Precision & Recall** - Per-class metrics
- **Dice Score** - F1-score for segmentation
- Custom metric support

### 🏗️ **Extensible Architecture**

- **Adapter Pattern** - Easy model integration
- **Registry System** - Plugin-based components
- **Universal Loader** - Support for any model type
- **Custom Components** - Add your own models, datasets, attacks, metrics

### 🎨 **Multiple Model Support**

- **Torchvision Models** - FCN, DeepLabV3, LRASPP
- **SMP Models** - [Segmentation Models PyTorch](https://github.com/qubvel-org/segmentation_models.pytorch)
- **HuggingFace Models** - Transformers-based models
- **Custom Models** - Your own implementations

### 📁 **Dataset Support**

- **VOC** - PASCAL VOC 2012
- **ADE20K** - MIT Scene Parsing
- **Cityscapes** - Urban scene understanding
- **Stanford Background** - Natural scene parsing

## 🏆 Why Choose This Framework?

### ✅ **Production Ready**

- Comprehensive error handling
- Memory-efficient processing
- GPU acceleration support
- Reproducible results

### ✅ **Research Friendly**

- Easy experiment configuration
- Detailed logging and reporting
- Custom component integration
- Open-source and extensible

### ✅ **Developer Friendly**

- Clean, well-documented API
- Type hints throughout
- Comprehensive test suite
- Active development and support

## 📈 Example Results

Here's what you can achieve with the framework:

| Model | Dataset | Clean IoU | FGSM IoU | PGD IoU |
|-------|---------|-----------|----------|---------|
| DeepLabV3+ | VOC | 82.3% | 45.2% | 23.1% |
| UNet | Cityscapes | 78.9% | 41.7% | 19.8% |
| SegFormer | ADE20K | 75.6% | 38.9% | 17.2% |

## 🤝 Contributing

We welcome contributions! Check out our comprehensive [Contributing Guide](contributing/index.md) to get started.

- 🐛 **Bug Reports** - Help us identify and fix issues
- 💡 **Feature Requests** - Suggest new features or improvements
- 📝 **Documentation** - Improve our docs and examples
- 🔧 **Code Contributions** - Add new models, attacks, metrics, or datasets
- 🧪 **Testing** - Help ensure code quality and reliability

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](https://github.com/wntic/segmentation-robustness-framework/blob/main/LICENSE) file for details.

---

<div align="center">
  <strong>Ready to evaluate your segmentation models?</strong>
  <br><br>
  <a href="quick_start" class="md-button md-button--primary">Get Started Now &rarr;</a>
</div>
