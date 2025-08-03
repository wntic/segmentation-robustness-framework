# Custom Components Guide

This guide explains how to create and integrate custom datasets, attacks, models, and metrics into the Segmentation Robustness Framework.

## 📚 Table of Contents

1. [Custom Datasets](#custom-datasets)
2. [Custom Attacks](#custom-attacks)
3. [Custom Models](#custom-models)
4. [Custom Metrics](#custom-metrics)
5. [Custom Adapters](#custom-adapters)
6. [Integration Examples](#integration-examples)
7. [Best Practices](#best-practices)
8. [Troubleshooting](#troubleshooting)

## 📊 Custom Datasets

### Dataset Requirements

Your custom dataset must inherit from `torch.utils.data.Dataset` and implement the required interface:

```python
from torch.utils.data import Dataset
from PIL import Image
import torch

class MyCustomDataset(Dataset):
    def __init__(self, root: str, transform=None, target_transform=None):
        # Initialize your dataset
        self.num_classes = 10  # Set your number of classes
        pass
    
    def __len__(self) -> int:
        # Return total number of samples
        pass
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Return (image, mask) pair
        pass
```

### Required Data Format

#### Images
- **Format**: PIL Image in RGB mode
- **Processing**: Will be automatically converted to tensor `[C, H, W]` and normalized
- **Normalization**: Uses ImageNet stats `mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]`

#### Masks
- **Format**: PIL Image or numpy array
- **Values**: Class indices (0 to num_classes-1)
- **Shape**: `[H, W]` or `[C, H, W]` (will be squeezed to 2D)
- **Data Type**: Will be converted to `torch.long`

#### Color-Coded Masks
If your masks are RGB images with color-coded classes:

```python
# Register your color palette
from segmentation_robustness_framework.utils.image_preprocessing import register_dataset_colors

MY_DATASET_COLORS = [
    (0, 0, 0),      # Class 0: background
    (255, 0, 0),    # Class 1: object1
    (0, 255, 0),    # Class 2: object2
    (0, 0, 255),    # Class 3: object3
    # ... more colors
]

register_dataset_colors("my_dataset", MY_DATASET_COLORS)
```

### Dataset Registration

Register your dataset to make it available in the framework:

```python
from segmentation_robustness_framework.datasets.registry import register_dataset

@register_dataset("my_dataset")
class MyCustomDataset(Dataset):
    def __init__(self, root: str, transform=None, target_transform=None):
        # Your initialization code
        self.num_classes = 10  # Set your number of classes
        # ... rest of initialization
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Load image and mask
        image = Image.open(image_path).convert("RGB")
        mask = Image.open(mask_path)  # or load as numpy array
        
        # Apply transforms if provided
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        
        return image, mask
```

### Complete Dataset Example

```python
import os
from pathlib import Path
from typing import Callable, Optional, Union
from PIL import Image
import torch
from torch.utils.data import Dataset
from segmentation_robustness_framework.datasets.registry import register_dataset

@register_dataset("my_custom")
class MyCustomDataset(Dataset):
    """Custom dataset for semantic segmentation.
    
    Attributes:
        root (str): Path to dataset directory
        transform (callable, optional): Image transformations
        target_transform (callable, optional): Mask transformations
        num_classes (int): Number of semantic classes
    """
    
    def __init__(
        self,
        root: Union[str, Path],
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.num_classes = 5  # Set your number of classes
        
        # Load file paths
        self.images_dir = self.root / "images"
        self.masks_dir = self.root / "masks"
        
        # Validate directories exist
        if not self.images_dir.exists():
            raise ValueError(f"Images directory not found: {self.images_dir}")
        if not self.masks_dir.exists():
            raise ValueError(f"Masks directory not found: {self.masks_dir}")
        
        # Get sorted list of image files
        self.images = sorted([
            f for f in os.listdir(self.images_dir) 
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        
        if len(self.images) == 0:
            raise ValueError(f"No image files found in {self.images_dir}")
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Load image
        img_path = self.images_dir / self.images[idx]
        image = Image.open(img_path).convert("RGB")
        
        # Load mask (assuming same name with .png extension)
        mask_name = self.images[idx].rsplit('.', 1)[0] + '.png'
        mask_path = self.masks_dir / mask_name
        
        if not mask_path.exists():
            raise FileNotFoundError(f"Mask not found: {mask_path}")
        
        mask = Image.open(mask_path)
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        
        return image, mask
    
    def __repr__(self) -> str:
        return f"MyCustomDataset(root={self.root}, num_samples={len(self)})"
```

### Dataset Directory Structure

Your dataset should follow this structure:

```
my_dataset/
├── images/
│   ├── image_001.jpg
│   ├── image_002.jpg
│   └── ...
├── masks/
│   ├── image_001.png
│   ├── image_002.png
│   └── ...
└── README.md  # Optional: dataset documentation
```

## ⚔️ Custom Attacks

### Attack Requirements

Your custom attack must inherit from `AdversarialAttack` and implement the attack method:

```python
from segmentation_robustness_framework.attacks.attack import AdversarialAttack
import torch

class MyCustomAttack(AdversarialAttack):
    def __init__(self, model, **kwargs):
        super().__init__(model)
        # Initialize your attack parameters
        pass
    
    def apply(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Implement your attack logic
        pass
```

### Required Data Format

#### Input Images
- **Format**: `torch.Tensor`
- **Shape**: `[B, C, H, W]` where B=batch_size, C=3 (RGB), H=height, W=width
- **Values**: Normalized to [0, 1] range
- **Device**: Will be moved to the same device as the model

#### Input Labels
- **Format**: `torch.Tensor`
- **Shape**: `[B, H, W]` or `[B, 1, H, W]` (will be squeezed to 3D)
- **Values**: Class indices (0 to num_classes-1)
- **Data Type**: `torch.long`
- **Special Values**: -1 for ignore_index (pixels to ignore)

#### Output Adversarial Images
- **Format**: `torch.Tensor`
- **Shape**: `[B, C, H, W]` (same as input images)
- **Values**: Must be in [0, 1] range (will be clamped)
- **Device**: Same device as input images

### Attack Registration

Register your attack to make it available:

```python
from segmentation_robustness_framework.attacks.registry import register_attack

@register_attack("my_attack")
class MyCustomAttack(AdversarialAttack):
    def __init__(self, model, eps: float = 0.1):
        super().__init__(model)
        self.eps = eps
    
    def apply(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        # Your attack implementation
        pass
```

### Complete Attack Example

```python
import torch
import torch.nn as nn
from segmentation_robustness_framework.attacks.attack import AdversarialAttack
from segmentation_robustness_framework.attacks.registry import register_attack

@register_attack("simple_fgsm")
class SimpleFGSM(AdversarialAttack):
    """Simple Fast Gradient Sign Method attack.
    
    Attributes:
        model (nn.Module): Model to attack
        eps (float): Maximum perturbation magnitude
    """
    
    def __init__(self, model: nn.Module, eps: float = 2/255):
        super().__init__(model)
        self.eps = eps
    
    def __repr__(self) -> str:
        return f"SimpleFGSM: eps={self.eps}"
    
    def apply(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Apply FGSM attack.
        
        Args:
            images (torch.Tensor): Input images [B, C, H, W]
            labels (torch.Tensor): Target labels [B, H, W]
        
        Returns:
            torch.Tensor: Adversarial images [B, C, H, W]
        """
        self.model.eval()
        
        # Move to device
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # Handle label shape
        if labels.dim() == 4 and labels.shape[1] == 1:
            labels = labels.squeeze(1)  # [B, H, W]
        
        # Filter valid pixels (ignore -1 values)
        valid_mask = labels >= 0
        
        if not torch.any(valid_mask):
            return images.detach()
        
        # Enable gradients
        images.requires_grad = True
        
        # Forward pass
        outputs = self.model(images)  # [B, num_classes, H, W]
        
        # Reshape for loss computation
        B, C, H, W = outputs.shape
        outputs_flat = outputs.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
        labels_flat = labels.reshape(-1)  # [B*H*W]
        
        # Only compute loss on valid pixels
        valid_indices = valid_mask.reshape(-1)  # [B*H*W]
        valid_outputs = outputs_flat[valid_indices]  # [N_valid, C]
        valid_labels = labels_flat[valid_indices]  # [N_valid]
        
        # Compute loss
        loss_fn = nn.CrossEntropyLoss()
        cost = loss_fn(valid_outputs, valid_labels)
        
        # Backward pass
        self.model.zero_grad()
        cost.backward()
        
        # Generate adversarial images
        adv_images = images + self.eps * images.grad.sign()
        adv_images = torch.clamp(adv_images, 0, 1)
        
        return adv_images.detach()
```

### Advanced Attack Example

```python
@register_attack("adaptive_pgd")
class AdaptivePGD(AdversarialAttack):
    """Adaptive PGD attack that adjusts parameters based on model performance."""
    
    def __init__(self, model, eps=8/255, alpha=2/255, max_iters=20, targeted=False):
        super().__init__(model)
        self.eps = eps
        self.alpha = alpha
        self.max_iters = max_iters
        self.targeted = targeted
    
    def apply(self, images, labels):
        self.model.eval()
        
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        if labels.dim() == 4 and labels.shape[1] == 1:
            labels = labels.squeeze(1)
        
        valid_mask = labels >= 0
        if not torch.any(valid_mask):
            return images.detach()
        
        # Initialize adversarial images
        adv_images = images.clone().detach()
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-self.eps, self.eps)
        adv_images = torch.clamp(adv_images, 0, 1).detach()
        
        loss_fn = nn.CrossEntropyLoss()
        
        for iteration in range(self.max_iters):
            adv_images.requires_grad = True
            
            outputs = self.model(adv_images)
            
            # Reshape for loss computation
            B, C, H, W = outputs.shape
            outputs_flat = outputs.permute(0, 2, 3, 1).reshape(-1, C)
            labels_flat = labels.reshape(-1)
            
            valid_indices = valid_mask.reshape(-1)
            valid_outputs = outputs_flat[valid_indices]
            valid_labels = labels_flat[valid_indices]
            
            if self.targeted:
                cost = -loss_fn(valid_outputs, valid_labels)
            else:
                cost = loss_fn(valid_outputs, valid_labels)
            
            # Adaptive step size based on loss
            adaptive_alpha = self.alpha * (1 + cost.item())
            
            self.model.zero_grad()
            grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
            
            adv_images = adv_images.detach() + adaptive_alpha * grad.sign()
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, 0, 1).detach()
        
        return adv_images
```

## 🏗️ Custom Models

### Model Requirements

Your custom model should follow PyTorch conventions and be compatible with the framework's adapter system:

```python
import torch
import torch.nn as nn

class MyCustomModel(nn.Module):
    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.num_classes = num_classes
        # Define your model architecture
        self.backbone = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, num_classes, 1)
        )
    
    def forward(self, x):
        return self.backbone(x)
```

### Model Loading

Use the custom model loader to load your model:

```python
from segmentation_robustness_framework.loaders.models.universal_loader import UniversalModelLoader

loader = UniversalModelLoader()

# Load custom model
model = loader.load_model(
    model_type="custom",
    model_config={
        "model_class": "path.to.MyCustomModel",
        "model_args": [21],  # num_classes
        "model_kwargs": {"pretrained": True}
    }
)
```

### Adapter Declaration (Required)

To use your custom model with the framework, you must wrap it with an adapter that implements the required interface (`logits`, `predictions`, and `num_classes`). The framework provides a `CustomAdapter` template, but you need to register your own adapter for use with the universal loader:

```python
from segmentation_robustness_framework.adapters import CustomAdapter
from segmentation_robustness_framework.adapters.registry import register_adapter

@register_adapter("my_custom_adapter")
class MyCustomAdapter(CustomAdapter):
    pass  # You can override methods if your model's output format is different
```

Then, specify the model type (name of registered adapter) in your loader:

```python
loader = UniversalModelLoader()

model = loader.load_model(
    model_type="my_custom_adapter",
    model_config={
        "model_class": "path.to.MyCustomModel",
        "model_args": [21],
        "model_kwargs": {"pretrained": True}
    },
)
```

If you do not register an adapter, the universal loader will not be able to wrap your model correctly.

### Complete Model Example

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SimpleSegmentationModel(nn.Module):
    """Simple segmentation model for demonstration."""
    
    def __init__(self, num_classes: int = 21, input_channels: int = 3):
        super().__init__()
        self.num_classes = num_classes
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Conv2d(input_channels, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2)
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(64, 32, 2, stride=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, num_classes, 1)
        )
    
    def forward(self, x):
        # Encoder
        features = self.encoder(x)
        
        # Decoder
        output = self.decoder(features)
        
        return output
    
    def get_num_classes(self):
        return self.num_classes
```

## 📈 Custom Metrics

### Metric Requirements

Custom metrics should follow this signature:

```python
def custom_metric(targets: torch.Tensor, predictions: torch.Tensor) -> float:
    """Custom evaluation metric.
    
    Args:
        targets: Ground truth labels [B, H, W]
        predictions: Predicted labels [B, H, W]
    
    Returns:
        float: Metric value
    """
    # Your metric implementation
    return score
```

### Metric Registration

```python
from segmentation_robustness_framework.utils.metrics import MetricsCollection

# Extend MetricsCollection
class ExtendedMetricsCollection(MetricsCollection):
    def custom_metric(self, targets: torch.Tensor, predictions: torch.Tensor) -> float:
        """Custom metric implementation."""
        # Ensure inputs are on CPU and converted to numpy
        targets, predictions = self._preprocess_input_data(targets, predictions)
        
        # Your metric computation
        # ...
        
        return score

# Use with pipeline
metrics_collection = ExtendedMetricsCollection(num_classes=21)
metrics = [metrics_collection.mean_iou, metrics_collection.custom_metric]
```

### Complete Metric Example

```python
import torch
import numpy as np
from segmentation_robustness_framework.utils.metrics import MetricsCollection

class ExtendedMetricsCollection(MetricsCollection):
    def boundary_accuracy(self, targets: torch.Tensor, predictions: torch.Tensor) -> float:
        """Compute boundary accuracy metric.
        
        This metric measures how well the model predicts object boundaries.
        """
        targets, predictions = self._preprocess_input_data(targets, predictions)
        
        # Convert to numpy for easier processing
        targets = targets.numpy()
        predictions = predictions.numpy()
        
        total_boundary_pixels = 0
        correct_boundary_pixels = 0
        
        for target, pred in zip(targets, predictions):
            # Find boundaries (simple edge detection)
            from scipy import ndimage
            
            # Create binary masks for each class
            for class_id in range(self.num_classes):
                target_mask = (target == class_id).astype(np.uint8)
                pred_mask = (pred == class_id).astype(np.uint8)
                
                # Find boundaries using morphological operations
                target_boundary = ndimage.binary_erosion(target_mask) != target_mask
                pred_boundary = ndimage.binary_erosion(pred_mask) != pred_mask
                
                # Count boundary pixels
                total_boundary_pixels += target_boundary.sum()
                correct_boundary_pixels += (target_boundary & pred_boundary).sum()
        
        if total_boundary_pixels == 0:
            return 1.0  # No boundaries to predict
        
        return correct_boundary_pixels / total_boundary_pixels
    
    def class_balanced_accuracy(self, targets: torch.Tensor, predictions: torch.Tensor) -> float:
        """Compute class-balanced accuracy.
        
        This metric gives equal weight to each class regardless of frequency.
        """
        targets, predictions = self._preprocess_input_data(targets, predictions)
        
        # Convert to numpy
        targets = targets.numpy()
        predictions = predictions.numpy()
        
        class_accuracies = []
        
        for class_id in range(self.num_classes):
            # Find pixels belonging to this class
            class_mask = targets == class_id
            
            if class_mask.sum() == 0:
                # Class not present in this batch
                continue
            
            # Compute accuracy for this class
            class_correct = (predictions[class_mask] == class_id).sum()
            class_accuracy = class_correct / class_mask.sum()
            class_accuracies.append(class_accuracy)
        
        if not class_accuracies:
            return 0.0
        
        return np.mean(class_accuracies)
```

## 🔧 Custom Adapters

### Adapter Requirements

Custom adapters must implement the `SegmentationModelProtocol`:

```python
import torch
from segmentation_robustness_framework.adapters.base_protocol import SegmentationModelProtocol
from segmentation_robustness_framework.adapters.registry import register_adapter

@register_adapter("my_adapter")
class MyCustomAdapter(torch.nn.Module, SegmentationModelProtocol):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.num_classes = self._get_num_classes()
    
    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits."""
        return self.model(x)
    
    def predictions(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted labels."""
        logits = self.logits(x)
        return torch.argmax(logits, dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
        return self.logits(x)
    
    def _get_num_classes(self) -> int:
        """Extract number of classes from model."""
        # Implementation depends on your model structure
        pass
```

### Complete Adapter Example

```python
import torch
from segmentation_robustness_framework.adapters.base_protocol import SegmentationModelProtocol
from segmentation_robustness_framework.adapters.registry import register_adapter

@register_adapter("custom_segmentation")
class CustomSegmentationAdapter(torch.nn.Module, SegmentationModelProtocol):
    """Adapter for custom segmentation models.
    
    This adapter handles models that return logits directly.
    """
    
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model
        self.num_classes = self._extract_num_classes()
    
    def _extract_num_classes(self) -> int:
        """Extract number of classes from model."""
        # Try different common patterns
        if hasattr(self.model, 'num_classes'):
            return self.model.num_classes
        elif hasattr(self.model, 'classifier'):
            if hasattr(self.model.classifier, 'out_channels'):
                return self.model.classifier.out_channels
        elif hasattr(self.model, 'last_layer'):
            if hasattr(self.model.last_layer, 'out_channels'):
                return self.model.last_layer.out_channels
        
        # Default fallback
        return 1
    
    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits for input images.
        
        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
        
        Returns:
            torch.Tensor: Logits tensor of shape (B, num_classes, H, W).
        """
        return self.model(x)
    
    def predictions(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted class labels for input images.
        
        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
        
        Returns:
            torch.Tensor: Predicted label tensor of shape (B, H, W).
        """
        logits = self.logits(x)
        return torch.argmax(logits, dim=1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x (torch.Tensor): Input image tensor of shape (B, C, H, W).
        
        Returns:
            torch.Tensor: Logits tensor of shape (B, num_classes, H, W).
        """
        return self.logits(x)
```

## Custom Adapters: Multiple and Direct Passing

### Registering Multiple Custom Adapters

You can register multiple custom adapters with names like `custom_myadapter`:

```python
from segmentation_robustness_framework.adapters import CustomAdapter
from segmentation_robustness_framework.adapters.registry import register_adapter

@register_adapter("custom_myadapter")
class MyCustomAdapter(CustomAdapter):
    pass

@register_adapter("custom_other")
class OtherCustomAdapter(CustomAdapter):
    pass
```

To use a specific adapter, set `model_type` to the registered name:

```python
loader = UniversalModelLoader()
model = loader.load_model(
    model_type="custom_myadapter",
    model_config={...}
)
```

### Passing Adapter Class Directly

You can also pass an adapter class directly to `UniversalModelLoader.load_model`:

```python
model = loader.load_model(
    model_type="custom",
    model_config={...},
    adapter_cls=MyCustomAdapter
)
```

If both a registered adapter and an `adapter_cls` are provided, the `adapter_cls` takes precedence.

## Loader Weights: Torchvision, SMP, HuggingFace

- **Torchvision**: Use `weights="default"` or `weights="DEFAULT"` (case-insensitive) to load default pretrained weights.
- **SMP**: Use `weight_type="encoder"` in `load_weights` to load only encoder weights.
- **HuggingFace**: Use `weight_type="encoder"` in `load_weights` to load only encoder weights.

## HuggingFace Loader: Expanded Model/Task Support

- You can specify a custom model class in the config with `model_cls` (string or class).
- Supported tasks: `semantic_segmentation`, `instance_segmentation`, `panoptic_segmentation`, `image_segmentation`.

## 🔗 Integration Examples

### Complete Custom Component Integration

```python
# 1. Define custom dataset
@register_dataset("medical_lungs")
class MedicalLungsDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.num_classes = 2  # Background and lungs
        
        # Load file paths
        self.images = [...]  # List of image paths
        self.masks = [...]   # List of mask paths
    
    def __getitem__(self, idx):
        image = Image.open(self.images[idx]).convert("RGB")
        mask = Image.open(self.masks[idx])
        
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        
        return image, mask

# 2. Define custom attack
@register_attack("medical_fgsm")
class MedicalFGSM(AdversarialAttack):
    def __init__(self, model, eps=2/255):
        super().__init__(model)
        self.eps = eps
    
    def apply(self, images, labels):
        # Medical-specific attack implementation
        # ...
        return adversarial_images

# 3. Define custom metric
def medical_dice_score(targets, predictions):
    """Medical-specific Dice score."""
    # Implementation
    return score

# 4. Use in pipeline
from segmentation_robustness_framework.engine.pipeline import SegmentationRobustnessPipeline

# Load components
dataset = MedicalLungsDataset(root="./data/medical")
attack = MedicalFGSM(model, eps=2/255)
metrics = [medical_dice_score]

# Create pipeline
pipeline = SegmentationRobustnessPipeline(
    model=model,
    dataset=dataset,
    attacks=[attack],
    metrics=metrics,
    batch_size=4
)

# Run evaluation
results = pipeline.run()
```

### Multi-Component Example

```python
# Multiple custom components working together
@register_dataset("satellite_imagery")
class SatelliteDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.num_classes = 5  # Different land cover types
        # Implementation...

@register_attack("satellite_robust")
class SatelliteRobustAttack(AdversarialAttack):
    def __init__(self, model, eps=4/255, weather_condition="cloudy"):
        super().__init__(model)
        self.eps = eps
        self.weather_condition = weather_condition
    
    def apply(self, images, labels):
        # Weather-aware attack implementation
        # ...
        return adversarial_images

@register_adapter("satellite_model")
class SatelliteModelAdapter(torch.nn.Module, SegmentationModelProtocol):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.num_classes = 5
    
    def logits(self, x):
        return self.model(x)
    
    def predictions(self, x):
        return torch.argmax(self.logits(x), dim=1)

# Integration
dataset = SatelliteDataset(root="./data/satellite")
model = SatelliteModelAdapter(load_satellite_model())
attacks = [
    SatelliteRobustAttack(model, eps=2/255, weather_condition="clear"),
    SatelliteRobustAttack(model, eps=4/255, weather_condition="cloudy"),
    SatelliteRobustAttack(model, eps=6/255, weather_condition="stormy"),
]

pipeline = SegmentationRobustnessPipeline(
    model=model,
    dataset=dataset,
    attacks=attacks,
    metrics=metrics
)
```

## ✅ Best Practices

### 1. Component Design

```python
# Good: Clear interface and documentation
@register_dataset("well_designed")
class WellDesignedDataset(Dataset):
    """Well-designed dataset with clear interface.
    
    This dataset loads satellite imagery for land cover classification.
    """
    
    def __init__(self, root, transform=None, target_transform=None):
        """Initialize the dataset.
        
        Args:
            root: Path to dataset directory
            transform: Image transformations
            target_transform: Mask transformations
        """
        super().__init__()
        self.root = Path(root)
        self.transform = transform
        self.target_transform = target_transform
        self.num_classes = 5
        
        # Validate dataset structure
        self._validate_dataset()
    
    def _validate_dataset(self):
        """Validate dataset structure and files."""
        if not self.root.exists():
            raise ValueError(f"Dataset root not found: {self.root}")
        
        # Check for required directories
        required_dirs = ["images", "masks"]
        for dir_name in required_dirs:
            dir_path = self.root / dir_name
            if not dir_path.exists():
                raise ValueError(f"Required directory not found: {dir_path}")
    
    def __getitem__(self, idx):
        """Get a sample from the dataset."""
        # Implementation with proper error handling
        try:
            image = self._load_image(idx)
            mask = self._load_mask(idx)
            
            if self.transform:
                image = self.transform(image)
            if self.target_transform:
                mask = self.target_transform(mask)
            
            return image, mask
        except Exception as e:
            raise RuntimeError(f"Error loading sample {idx}: {e}")
    
    def _load_image(self, idx):
        """Load image with error handling."""
        # Implementation
        pass
    
    def _load_mask(self, idx):
        """Load mask with error handling."""
        # Implementation
        pass
```

### 2. Error Handling

```python
# Good: Comprehensive error handling
class RobustAttack(AdversarialAttack):
    def apply(self, images, labels):
        try:
            # Validate inputs
            if images.dim() != 4:
                raise ValueError(f"Images must be 4D, got {images.dim()}D")
            
            if labels.dim() not in [3, 4]:
                raise ValueError(f"Labels must be 3D or 4D, got {labels.dim()}D")
            
            # Check value ranges
            if torch.any(images < 0) or torch.any(images > 1):
                raise ValueError("Images must be in [0, 1] range")
            
            # Implementation
            return adversarial_images
            
        except Exception as e:
            logger.error(f"Attack failed: {e}")
            # Return original images as fallback
            return images.detach()
```

### 3. Performance Optimization

```python
# Good: Optimized implementation
class OptimizedDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        super().__init__()
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        
        # Pre-load file paths for faster access
        self._load_file_paths()
        
        # Optional: Cache frequently accessed data
        self.cache = {}
        self.cache_size = 100
    
    def _load_file_paths(self):
        """Pre-load all file paths."""
        self.images = []
        self.masks = []
        
        # Load paths efficiently
        for img_path in self.root.glob("images/*.jpg"):
            mask_path = img_path.parent.parent / "masks" / f"{img_path.stem}.png"
            if mask_path.exists():
                self.images.append(img_path)
                self.masks.append(mask_path)
    
    def __getitem__(self, idx):
        # Use caching for frequently accessed samples
        if idx in self.cache:
            return self.cache[idx]
        
        # Load sample
        sample = self._load_sample(idx)
        
        # Cache if cache not full
        if len(self.cache) < self.cache_size:
            self.cache[idx] = sample
        
        return sample
```

### 4. Testing

```python
# Good: Comprehensive testing
import pytest
import torch

def test_custom_dataset():
    """Test custom dataset functionality."""
    dataset = MyCustomDataset(root="./test_data")
    
    # Test length
    assert len(dataset) > 0
    
    # Test sample loading
    image, mask = dataset[0]
    assert isinstance(image, torch.Tensor)
    assert isinstance(mask, torch.Tensor)
    assert image.shape[0] == 3  # RGB channels
    assert mask.shape == image.shape[1:]  # Same spatial dimensions

def test_custom_attack():
    """Test custom attack functionality."""
    model = SimpleModel(num_classes=2)
    attack = MyCustomAttack(model, eps=2/255)
    
    # Test input validation
    images = torch.randn(2, 3, 224, 224)
    labels = torch.randint(0, 2, (2, 224, 224))
    
    # Test attack application
    adv_images = attack.apply(images, labels)
    assert adv_images.shape == images.shape
    assert torch.all(adv_images >= 0) and torch.all(adv_images <= 1)

def test_custom_metric():
    """Test custom metric functionality."""
    targets = torch.randint(0, 2, (2, 224, 224))
    predictions = torch.randint(0, 2, (2, 224, 224))
    
    score = custom_metric(targets, predictions)
    assert isinstance(score, float)
    assert 0 <= score <= 1
```

## 🐛 Troubleshooting

### Common Issues

#### 1. Registration Errors

```python
# Problem: Component not found after registration
# Solution: Ensure imports are correct
from segmentation_robustness_framework.datasets.registry import register_dataset
from segmentation_robustness_framework.attacks.registry import register_attack

# Make sure decorators are applied correctly
@register_dataset("my_dataset")  # Correct
class MyDataset(Dataset):
    pass

# Not like this:
class MyDataset(Dataset):
    pass
register_dataset("my_dataset")(MyDataset)  # Incorrect
```

#### 2. Shape Mismatches

```python
# Problem: Tensor shape errors
# Solution: Validate shapes in your components

def validate_shapes(tensor, expected_shape, name):
    if tensor.shape != expected_shape:
        raise ValueError(f"{name} shape mismatch: {tensor.shape} vs {expected_shape}")

# Use in your components
def __getitem__(self, idx):
    image, mask = self._load_sample(idx)
    
    validate_shapes(image, (3, 224, 224), "image")
    validate_shapes(mask, (224, 224), "mask")
    
    return image, mask
```

#### 3. Device Issues

```python
# Problem: Tensors on different devices
# Solution: Ensure consistent device placement

class DeviceAwareAttack(AdversarialAttack):
    def apply(self, images, labels):
        # Ensure all tensors are on the same device
        device = next(self.model.parameters()).device
        
        images = images.to(device)
        labels = labels.to(device)
        
        # Implementation...
        return adversarial_images.to(images.device)
```

#### 4. Memory Issues

```python
# Problem: Out of memory errors
# Solution: Implement memory-efficient loading

class MemoryEfficientDataset(Dataset):
    def __init__(self, root, cache_size=50):
        self.root = root
        self.cache = {}
        self.cache_size = cache_size
    
    def __getitem__(self, idx):
        # Use LRU cache
        if idx in self.cache:
            return self.cache[idx]
        
        # Load sample
        sample = self._load_sample(idx)
        
        # Manage cache size
        if len(self.cache) >= self.cache_size:
            # Remove oldest entry
            oldest_key = next(iter(self.cache))
            del self.cache[oldest_key]
        
        self.cache[idx] = sample
        return sample
```

### Debugging Tips

```python
# 1. Enable debug logging
import logging
logging.basicConfig(level=logging.DEBUG)

# 2. Add debug prints
class DebugDataset(Dataset):
    def __getitem__(self, idx):
        print(f"Loading sample {idx}")
        image, mask = self._load_sample(idx)
        print(f"Sample {idx} shapes: image={image.shape}, mask={mask.shape}")
        return image, mask

# 3. Validate data at each step
def validate_data(image, mask, idx):
    """Validate data integrity."""
    if torch.isnan(image).any():
        raise ValueError(f"NaN values in image {idx}")
    if torch.isnan(mask).any():
        raise ValueError(f"NaN values in mask {idx}")
    
    if image.min() < 0 or image.max() > 1:
        print(f"Warning: Image {idx} values outside [0, 1] range")

# 4. Use small test dataset
test_dataset = MyCustomDataset(root="./test_data")
print(f"Test dataset size: {len(test_dataset)}")
for i in range(min(5, len(test_dataset))):
    image, mask = test_dataset[i]
    print(f"Sample {i}: image={image.shape}, mask={mask.shape}")
```

---

**This guide covers all aspects of creating custom components. For more information:**

- 📖 [User Guide](user_guide.md) - General framework usage
<!-- - 🧪 [Advanced Usage](advanced_usage.md) - Advanced features
- 📋 [API Reference](api_reference.md) - Complete API documentation  -->