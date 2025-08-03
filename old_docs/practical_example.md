# Practical Example: Custom Dataset and Attack

This example demonstrates how to integrate a custom dataset and attack into the segmentation robustness framework.

## Example: Medical Image Segmentation Dataset

### 1. Custom Dataset Implementation

```python
import os
import numpy as np
from pathlib import Path
from typing import Callable, Optional, Union
from PIL import Image
import torch
from torch.utils.data import Dataset
from segmentation_robustness_framework.datasets.registry import register_dataset

@register_dataset("medical_lungs")
class MedicalLungsDataset(Dataset):
    """Medical lung segmentation dataset.
    
    This dataset contains CT scans with lung segmentation masks.
    Classes: 0=background, 1=left_lung, 2=right_lung
    """
    
    def __init__(
        self,
        root: Union[str, Path],
        split: str = "train",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        self.root = Path(root)
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.num_classes = 3  # background, left_lung, right_lung
        
        # Setup paths
        self.images_dir = self.root / "images" / split
        self.masks_dir = self.root / "masks" / split
        
        # Get file list
        self.images = sorted([f for f in os.listdir(self.images_dir) 
                            if f.endswith('.png')])
    
    def __len__(self) -> int:
        return len(self.images)
    
    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Load image (CT scan)
        img_path = self.images_dir / self.images[idx]
        image = Image.open(img_path).convert("RGB")
        
        # Load mask (grayscale with class indices)
        mask_path = self.masks_dir / self.images[idx]
        mask = Image.open(mask_path).convert("L")  # Grayscale
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        
        return image, mask
```

### 2. Register Color Palette (if using RGB masks)

```python
from segmentation_robustness_framework.utils.image_preprocessing import register_dataset_colors

# Define colors for visualization (if masks are RGB)
MEDICAL_LUNGS_COLORS = [
    (0, 0, 0),      # Class 0: background (black)
    (255, 0, 0),    # Class 1: left lung (red)
    (0, 255, 0),    # Class 2: right lung (green)
]

register_dataset_colors("medical_lungs", MEDICAL_LUNGS_COLORS)
```

### 3. Custom Attack Implementation

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from segmentation_robustness_framework.attacks.attack import AdversarialAttack
from segmentation_robustness_framework.attacks.registry import register_attack

@register_attack("medical_fgsm")
class MedicalFGSM(AdversarialAttack):
    """FGSM attack adapted for medical images.
    
    This attack is specifically designed for medical images with:
    - Smaller perturbation magnitudes (medical images are sensitive)
    - Focus on clinically relevant regions
    """
    
    def __init__(self, model: nn.Module, eps: float = 1/255, focus_region: bool = True):
        super().__init__(model)
        self.eps = eps
        self.focus_region = focus_region
    
    def __repr__(self) -> str:
        return f"MedicalFGSM: eps={self.eps}, focus_region={self.focus_region}"
    
    def apply(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Apply medical-adapted FGSM attack.
        
        Args:
            images (torch.Tensor): Input CT images [B, C, H, W]
            labels (torch.Tensor): Lung segmentation labels [B, H, W]
            
        Returns:
            torch.Tensor: Adversarial CT images [B, C, H, W]
        """
        self.model.eval()
        
        # Move to device
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        # Handle label shape
        if labels.dim() == 4 and labels.shape[1] == 1:
            labels = labels.squeeze(1)  # [B, H, W]
        
        # Create focus mask for lung regions (classes 1 and 2)
        if self.focus_region:
            focus_mask = (labels == 1) | (labels == 2)  # Lung regions
        else:
            focus_mask = torch.ones_like(labels, dtype=torch.bool)
        
        # Filter valid pixels
        valid_mask = labels >= 0
        combined_mask = valid_mask & focus_mask
        
        if not torch.any(combined_mask):
            return images.detach()
        
        # Enable gradients
        images.requires_grad = True
        
        # Forward pass
        outputs = self.model(images)  # [B, num_classes, H, W]
        
        # Reshape for loss computation
        B, C, H, W = outputs.shape
        outputs_flat = outputs.permute(0, 2, 3, 1).reshape(-1, C)  # [B*H*W, C]
        labels_flat = labels.reshape(-1)  # [B*H*W]
        
        # Only compute loss on valid lung pixels
        valid_indices = combined_mask.reshape(-1)  # [B*H*W]
        valid_outputs = outputs_flat[valid_indices]  # [N_valid, C]
        valid_labels = labels_flat[valid_indices]  # [N_valid]
        
        # Compute loss with higher weight for lung regions
        loss_fn = nn.CrossEntropyLoss()
        cost = loss_fn(valid_outputs, valid_labels)
        
        # Backward pass
        self.model.zero_grad()
        cost.backward()
        
        # Generate adversarial images with smaller perturbation
        adv_images = images + self.eps * images.grad.sign()
        adv_images = torch.clamp(adv_images, 0, 1)
        
        return adv_images.detach()
```

### 4. Complete Integration Example

```python
import torch
from segmentation_robustness_framework.utils.image_preprocessing import get_preprocessing_fn
from segmentation_robustness_framework.loaders.models.universal_loader import UniversalModelLoader
from segmentation_robustness_framework.engine.pipeline import SegmentationRobustnessPipeline
from segmentation_robustness_framework.utils.metrics import MetricsCollection

def run_medical_robustness_evaluation():
    """Run robustness evaluation on medical lung segmentation dataset."""
    
    # 1. Load model (pretrained on medical data)
    model_loader = UniversalModelLoader()
    model_config = {
        "name": "fcn_resnet50",
        "num_classes": 3,  # background, left_lung, right_lung
        "weights": "DEFAULT"
    }
    model = model_loader.load_model("torchvision", model_config)
    
    # 2. Get preprocessing functions
    preprocess, target_preprocess = get_preprocessing_fn([256, 256], "medical_lungs")
    
    # 3. Load dataset
    dataset = MedicalLungsDataset(
        root="data/medical_lungs",  # Exact path to dataset directory
        split="val",
        transform=preprocess,
        target_transform=target_preprocess
    )
    
    # 4. Create attack
    attack = MedicalFGSM(model, eps=1/255, focus_region=True)
    
    # 5. Setup metrics
    metrics_collection = MetricsCollection(num_classes=3)
    metrics = [
        metrics_collection.mean_iou,
        metrics_collection.pixel_accuracy,
        metrics_collection.dice_score
    ]
    
    # 6. Create and run pipeline
    pipeline = SegmentationRobustnessPipeline(
        model=model,
        dataset=dataset,
        attacks=[attack],
        metrics=metrics,
        batch_size=2,  # Smaller batch for medical images
        device="cuda",
        output_dir="./results/medical_robustness"
    )
    
    # 7. Run evaluation
    pipeline.run(save=True, show=False)
    
    print("Medical robustness evaluation completed!")

if __name__ == "__main__":
    run_medical_robustness_evaluation()
```

## Example: Custom Loss Function Attack

### Advanced Attack with Custom Loss

```python
@register_attack("dice_attack")
class DiceLossAttack(AdversarialAttack):
    """Attack using Dice loss instead of CrossEntropy.
    
    This attack is particularly effective for segmentation tasks
    as it directly optimizes the Dice coefficient.
    """
    
    def __init__(self, model: nn.Module, eps: float = 2/255, alpha: float = 0.5):
        super().__init__(model)
        self.eps = eps
        self.alpha = alpha  # Weight for Dice loss
    
    def dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Compute Dice loss for segmentation."""
        smooth = 1e-6
        
        # Convert to one-hot
        pred_soft = F.softmax(pred, dim=1)
        target_onehot = F.one_hot(target, num_classes=pred.shape[1]).float()
        
        # Compute Dice coefficient
        intersection = (pred_soft * target_onehot).sum(dim=1)
        union = pred_soft.sum(dim=1) + target_onehot.sum(dim=1)
        
        dice = (2 * intersection + smooth) / (union + smooth)
        return 1 - dice.mean()
    
    def apply(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Apply Dice loss-based attack."""
        self.model.eval()
        
        images = images.to(self.device)
        labels = labels.to(self.device)
        
        if labels.dim() == 4 and labels.shape[1] == 1:
            labels = labels.squeeze(1)
        
        valid_mask = labels >= 0
        if not torch.any(valid_mask):
            return images.detach()
        
        images.requires_grad = True
        
        # Forward pass
        outputs = self.model(images)
        
        # Reshape for loss computation
        B, C, H, W = outputs.shape
        outputs_flat = outputs.permute(0, 2, 3, 1).reshape(-1, C)
        labels_flat = labels.reshape(-1)
        
        valid_indices = valid_mask.reshape(-1)
        valid_outputs = outputs_flat[valid_indices]
        valid_labels = labels_flat[valid_indices]
        
        # Combined loss: CrossEntropy + Dice
        ce_loss = nn.CrossEntropyLoss()(valid_outputs, valid_labels)
        dice_loss = self.dice_loss(valid_outputs, valid_labels)
        
        total_loss = ce_loss + self.alpha * dice_loss
        
        self.model.zero_grad()
        total_loss.backward()
        
        adv_images = images + self.eps * images.grad.sign()
        adv_images = torch.clamp(adv_images, 0, 1)
        
        return adv_images.detach()
```

## Testing Your Implementation

### Validation Script

```python
def validate_custom_implementation():
    """Validate that custom dataset and attack work correctly."""
    
    # Test dataset
    dataset = MedicalLungsDataset(root="data/medical_lungs", split="val")  # Exact path
    print(f"Dataset loaded: {len(dataset)} samples")
    
    # Test single sample
    image, mask = dataset[0]
    print(f"Image shape: {image.shape}, dtype: {image.dtype}")
    print(f"Mask shape: {mask.shape}, dtype: {mask.dtype}")
    print(f"Mask unique values: {torch.unique(mask)}")
    
    # Test model compatibility
    model = load_model()
    model.eval()
    
    # Test forward pass
    with torch.no_grad():
        output = model(image.unsqueeze(0))
        print(f"Model output shape: {output.shape}")
    
    # Test attack
    attack = MedicalFGSM(model, eps=1/255)
    adv_image = attack(image.unsqueeze(0), mask.unsqueeze(0))
    print(f"Adversarial image shape: {adv_image.shape}")
    print(f"Perturbation range: {torch.min(adv_image - image.unsqueeze(0)):.4f} to {torch.max(adv_image - image.unsqueeze(0)):.4f}")
    
    print("Validation completed successfully!")

if __name__ == "__main__":
    validate_custom_implementation()
```

## Key Points for Medical Applications

1. **Smaller Perturbations**: Medical images are sensitive, use smaller eps values
2. **Focus on Regions**: Target clinically relevant areas
3. **Validation**: Ensure attacks don't create unrealistic artifacts
4. **Ethical Considerations**: Be careful with medical data
5. **Performance**: Medical models often need higher accuracy

This example shows how to adapt the framework for domain-specific applications while maintaining compatibility with the existing infrastructure. 
