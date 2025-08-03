# Quick Start

Get your first segmentation robustness evaluation running in 5 minutes!

## 🚀 Install

```bash
# Install the framework
pip install segmentation-robustness-framework

# Install with all optional dependencies for full functionality
pip install segmentation-robustness-framework[full]
```

## 🎯 Your First Example

Here's a complete working example that evaluates a segmentation model against adversarial attacks:

```python
from segmentation_robustness_framework.pipeline import SegmentationRobustnessPipeline
from segmentation_robustness_framework.metrics import MetricsCollection
from segmentation_robustness_framework.attacks import FGSM
from segmentation_robustness_framework.datasets import StanfordBackground
from segmentation_robustness_framework.loaders.models.universal_loader import UniversalModelLoader
from segmentation_robustness_framework.utils import image_preprocessing
import torch

# Step 1: Load a model
loader = UniversalModelLoader()
model = loader.load_model(
    model_type="torchvision",
    model_config={"name": "deeplabv3_resnet50", "num_classes": 21}
)

# Step 2: Set up device (IMPORTANT: Do this before creating attacks!)
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

# Step 3: Prepare dataset with preprocessing
preprocess, target_preprocess = image_preprocessing.get_preprocessing_fn(
    [512, 512], dataset_name="stanford_background"
)
dataset = StanfordBackground(
    root="./data",
    transform=preprocess,
    target_transform=target_preprocess,
    download=True  # This will download the dataset automatically
)

# Step 4: Set up attack and metrics
attack = FGSM(model, eps=2/255)  # Fast Gradient Sign Method attack
metrics_collection = MetricsCollection(num_classes=21)
metrics = [metrics_collection.mean_iou, metrics_collection.pixel_accuracy]

# Step 5: Create and run the evaluation pipeline
pipeline = SegmentationRobustnessPipeline(
    model=model,
    dataset=dataset,
    attacks=[attack],
    metrics=metrics,
    batch_size=4,
    device=device
)

# Step 6: Run the evaluation
results = pipeline.run(save=True, show=True)

# Step 7: View your results
print("🎉 Evaluation complete!")
print(f"Clean IoU: {results['clean']['mean_iou']:.3f}")
# Get the actual attack name from results (attack names include parameters like "attack_FGSM_eps_0p008")
attack_key = [key for key in results.keys() if key.startswith('attack_')][0]
print(f"Attack IoU: {results[attack_key]['mean_iou']:.3f}")
print(f"Robustness drop: {results['clean']['mean_iou'] - results[attack_key]['mean_iou']:.3f}")
```

## ✅ What Just Happened?

1. **We loaded a model**: Used the universal loader to load a DeepLabV3 model from torchvision
2. **We set up the device**: Moved the model to GPU/CPU (this is crucial for attacks to work properly)
3. **We prepared the dataset**: Downloaded and set up the Stanford Background dataset with proper preprocessing
4. **We configured the attack**: Created an FGSM attack with a small perturbation (2/255)
5. **We set up metrics**: Chose IoU and pixel accuracy to measure performance
6. **We ran the evaluation**: The pipeline tested both clean and adversarial performance
7. **We got results**: Clean performance vs. attack performance, showing how robust your model is

## 📊 Understanding Your Results

The evaluation gives you two key metrics:

- **Clean IoU**: How well your model performs on normal images
- **Attack IoU**: How well your model performs on adversarial images
- **Robustness drop**: The difference shows how vulnerable your model is to attacks

A smaller drop means your model is more robust!

## 🔧 What You Can Customize

### Different Models
```python
# Try different torchvision models
model = loader.load_model(
    model_type="torchvision",
    model_config={"name": "fcn_resnet50", "num_classes": 21}
)

# Or use SMP models
model = loader.load_model(
    model_type="smp",
    model_config={"architecture": "unet", "encoder_name": "resnet50", "classes": 21}
)
```

### Different Attacks
```python
from segmentation_robustness_framework.attacks import PGD, RFGSM

# Try PGD (more powerful attack)
attack = PGD(model, eps=8/255, alpha=2/255, iters=10)

# Or R+FGSM (FGSM with random initialization)
attack = RFGSM(model, eps=8/255, alpha=2/255, iters=10)
```

### Different Datasets
```python
from segmentation_robustness_framework.datasets import VOCSegmentation

# Use VOC dataset (21 classes)
dataset = VOCSegmentation(
    split="val",
    root="./data",
    transform=preprocess,
    target_transform=target_preprocess,
    download=True
)
```

## 🚨 Common Issues and Solutions

### Issue: "CUDA out of memory"
**Solution**: Reduce batch size or use CPU
```python
pipeline = SegmentationRobustnessPipeline(
    model=model,
    dataset=dataset,
    attacks=[attack],
    metrics=metrics,
    batch_size=1,  # Reduce from 4 to 1
    device="cpu"   # Or use CPU instead of GPU
)
```

### Issue: "Model not found"
**Solution**: Check model name and install dependencies
```python
# Make sure you have the right dependencies
pip install torch torchvision

# Use a valid model name
model = loader.load_model(
    model_type="torchvision",
    model_config={"name": "deeplabv3_resnet50", "num_classes": 21}
)
```

### Issue: "Dataset download failed"
**Solution**: Check internet connection and disk space
```python
# Try downloading to a different location
dataset = StanfordBackground(
    root="/path/to/your/data",  # Use absolute path
    download=True
)
```

## 🎯 Next Steps

Now that you've run your first evaluation, here's what to explore next:

- **[User Guide](user_guide.md)** - Learn more features and customization options
- **[API Reference](api_reference/index.md)** - Complete technical documentation
- **[Examples](examples/basic_examples.md)** - Real-world use cases and advanced scenarios
- **[Advanced Examples](examples/advanced_examples.md)** - Add your own datasets, models, and attacks

## 💡 Pro Tips

1. **Start small**: Use smaller datasets and batch sizes for testing
2. **Check your device**: Make sure GPU is available if you want faster evaluation
3. **Save results**: The pipeline automatically saves detailed results to files
4. **Try different attacks**: Different attacks test different types of vulnerabilities
5. **Monitor memory**: Large models and datasets need more memory

---

**🎉 Congratulations! You've successfully evaluated your first segmentation model for robustness against adversarial attacks.**
