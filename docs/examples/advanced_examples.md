# Advanced Examples

This page provides advanced examples and use cases for the Segmentation Robustness Framework. 🚀

## 🤖 Multi-Model Evaluation

Compare multiple models on the same dataset and attacks:

```python
from segmentation_robustness_framework.pipeline import SegmentationRobustnessPipeline
from segmentation_robustness_framework.loaders import UniversalModelLoader, DatasetLoader
from segmentation_robustness_framework.metrics import MetricsCollection
from segmentation_robustness_framework.attacks import FGSM, PGD
import torch

# Initialize components
model_loader = UniversalModelLoader()
dataset_loader = DatasetLoader({
    "name": "voc",
    "split": "val",
    "root": "./data",
    "image_shape": [512, 512],
    "download": True
})
dataset = dataset_loader.load_dataset()

# Define models to compare
models = {
    "deeplabv3_resnet50": {
        "type": "torchvision",
        "config": {"name": "deeplabv3_resnet50", "num_classes": 21}
    },
    "deeplabv3_resnet101": {
        "type": "torchvision", 
        "config": {"name": "deeplabv3_resnet101", "num_classes": 21}
    },
    "fcn_resnet50": {
        "type": "torchvision",
        "config": {"name": "fcn_resnet50", "num_classes": 21}
    }
}

# Define attacks
attacks = [
    FGSM(None, eps=0.02),  # Will be set per model
    FGSM(None, eps=0.05),
    PGD(None, eps=0.02, alpha=0.02, iters=10, targeted=False),
    PGD(None, eps=0.05, alpha=0.02, iters=20, targeted=False)
]

# Define metrics
metrics = MetricsCollection(num_classes=21, ignore_index=255)
metric_functions = [metrics.mean_iou, metrics.pixel_accuracy, metrics.dice_score]

# Evaluate each model
results = {}
for model_name, model_config in models.items():
    print(f"\nEvaluating {model_name}...")
    
    # Load model
    model = model_loader.load_model(
        model_type=model_config["type"],
        model_config=model_config["config"]
    )
    model = model.to("cuda")
    
    # Set model for attacks
    for attack in attacks:
        attack.model = model
        attack.set_device("cuda")
    
    # Create pipeline
    pipeline = SegmentationRobustnessPipeline(
        model=model,
        dataset=dataset,
        attacks=attacks,
        metrics=metric_functions,
        batch_size=4,
        device="cuda",
        output_dir=f"./results/{model_name}",
        auto_resize_masks=True,
        output_formats=["json", "csv"]
    )
    
    # Run evaluation
    model_results = pipeline.run()
    results[model_name] = model_results

# Compare results
print("\n" + "="*60)
print("MODEL COMPARISON RESULTS")
print("="*60)

for model_name, model_results in results.items():
    print(f"\n{model_name}:")
    print(f"  Clean IoU: {model_results['clean']['mean_iou']:.3f}")
    print(f"  Clean Pixel Accuracy: {model_results['clean']['pixel_accuracy']:.3f}")
    
    for attack_name, attack_results in model_results.items():
        if attack_name.startswith('attack_'):
            print(f"  {attack_name}:")
            print(f"    IoU: {attack_results['mean_iou']:.3f}")
            print(f"    Pixel Accuracy: {attack_results['pixel_accuracy']:.3f}")
```

## ⚔️ Custom Attack Implementation

Create your own adversarial attack:

```python
import torch
import torch.nn as nn
from segmentation_robustness_framework.attacks import AdversarialAttack
from segmentation_robustness_framework.attacks.registry import register_attack

@register_attack("custom_attack")
class CustomAttack(AdversarialAttack):
    """Custom adversarial attack implementation.
    
    This is an example of how to implement your own attack.
    """
    
    def __init__(self, model: nn.Module, eps: float = 0.02, alpha: float = 0.01, iters: int = 10):
        """Initialize custom attack.
        
        Args:
            model (nn.Module): Model to attack.
            eps (float): Maximum perturbation magnitude.
            alpha (float): Step size for each iteration.
            iters (int): Number of iterations.
        """
        super().__init__(model)
        self.eps = eps
        self.alpha = alpha
        self.iters = iters
    
    def apply(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Apply custom attack to images.
        
        Args:
            images (torch.Tensor): Input images [B, C, H, W].
            labels (torch.Tensor): Target labels [B, H, W].
            
        Returns:
            torch.Tensor: Adversarial images [B, C, H, W].
        """
        self.model.eval()
        
        images = images.to(self.device, non_blocking=True)
        labels = labels.to(self.device, non_blocking=True)
        
        # Initialize adversarial images
        adv_images = images.clone().detach()
        
        for _ in range(self.iters):
            adv_images.requires_grad = True
            
            # Forward pass
            outputs = self.model(adv_images)
            
            # Compute loss
            loss = torch.nn.CrossEntropyLoss()(outputs, labels)
            
            # Compute gradients
            grad = torch.autograd.grad(loss, adv_images, retain_graph=False)[0]
            
            # Update adversarial images
            adv_images = adv_images.detach() + self.alpha * grad.sign()
            
            # Project to epsilon ball
            delta = torch.clamp(adv_images - images, min=-self.eps, max=self.eps)
            adv_images = torch.clamp(images + delta, min=0, max=1).detach()
        
        return adv_images

# Use custom attack
custom_attack = CustomAttack(model, eps=0.03, alpha=0.01, iters=15)

pipeline = SegmentationRobustnessPipeline(
    model=model,
    dataset=dataset,
    attacks=[custom_attack],
    metrics=[metrics.mean_iou],
    batch_size=4,
    device="cuda"
)
```

## 📊 Advanced Metric Configuration

Create complex metric configurations with custom averaging:

```python
from segmentation_robustness_framework.metrics import MetricsCollection, register_custom_metric
import numpy as np

# Custom weighted IoU metric
@register_custom_metric("weighted_iou")
def weighted_iou(targets, predictions):
    """Compute weighted IoU with class-specific weights."""
    # Convert to numpy
    if isinstance(targets, torch.Tensor):
        targets = targets.cpu().numpy()
    if isinstance(predictions, torch.Tensor):
        predictions = predictions.cpu().numpy()
    
    # Define class weights (e.g., more weight for foreground classes)
    class_weights = np.ones(21)  # VOC has 21 classes
    class_weights[1:] = 2.0  # Give more weight to foreground classes
    
    # Compute IoU for each class
    ious = []
    for cls in range(21):
        if cls == 255:  # ignore index
            continue
            
        pred = (predictions == cls).astype(np.int32)
        true = (targets == cls).astype(np.int32)
        
        intersection = np.sum(pred * true)
        union = np.sum(pred) + np.sum(true) - intersection
        
        if union > 0:
            iou = intersection / union
        else:
            iou = 0.0
        ious.append(iou)
    
    # Compute weighted average
    valid_ious = [iou for iou in ious if not np.isnan(iou)]
    if valid_ious:
        return np.average(valid_ious, weights=class_weights[:len(valid_ious)])
    return 0.0

# Create metrics collection with custom metrics
metrics = MetricsCollection(num_classes=21, ignore_index=255)

# Define metric configuration with different averaging strategies
metric_config = {
    "ignore_index": 255,
    "selected_metrics": [
        "mean_iou",  # Default macro averaging
        "pixel_accuracy",
        {"name": "precision", "average": "macro"},
        {"name": "precision", "average": "micro"},
        {"name": "recall", "average": "macro"},
        {"name": "recall", "average": "micro"},
        {"name": "dice_score", "average": "macro"},
        {"name": "dice_score", "average": "micro"},
        "weighted_iou"  # Custom metric
    ]
}

# Use in pipeline
pipeline = SegmentationRobustnessPipeline(
    model=model,
    dataset=dataset,
    attacks=[FGSM(model, eps=0.02)],
    metrics=[metrics.mean_iou, metrics.pixel_accuracy, weighted_iou],
    batch_size=4,
    device="cuda"
)
```

## 📁 Custom Dataset Integration

Integrate your own dataset:

```python
import torch
from torch.utils.data import Dataset
from PIL import Image
import os
from segmentation_robustness_framework.datasets.registry import register_dataset

@register_dataset("custom_dataset")
class CustomDataset(Dataset):
    """Custom dataset implementation."""
    
    def __init__(self, root: str, split: str = "train", transform=None, target_transform=None):
        """Initialize custom dataset.
        
        Args:
            root (str): Dataset root directory.
            split (str): Dataset split ('train', 'val', 'test').
            transform: Image transformations.
            target_transform: Target transformations.
        """
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        
        # Load image and mask paths
        self.images_dir = os.path.join(root, "images", split)
        self.masks_dir = os.path.join(root, "masks", split)
        
        self.images = [f for f in os.listdir(self.images_dir) if f.endswith('.jpg')]
        self.num_classes = 21  # Adjust based on your dataset
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        # Load image
        img_path = os.path.join(self.images_dir, self.images[idx])
        image = Image.open(img_path).convert('RGB')
        
        # Load mask
        mask_path = os.path.join(self.masks_dir, self.images[idx].replace('.jpg', '.png'))
        mask = Image.open(mask_path)
        
        # Apply transformations
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            mask = self.target_transform(mask)
        
        return image, mask

# Use custom dataset
dataset_config = {
    "name": "custom_dataset",
    "root": "./path/to/your/dataset",
    "split": "val",
    "image_shape": [512, 512],
}

dataset_loader = DatasetLoader(dataset_config)
dataset = dataset_loader.load_dataset()
```

## 🤖 Advanced Model Integration

Integrate complex models with custom preprocessing:

```python
import torch
import torch.nn as nn
from segmentation_robustness_framework.adapters import CustomAdapter
from transformers import AutoImageProcessor, AutoModelForSemanticSegmentation

class CustomHuggingFaceAdapter(CustomAdapter):
    """Custom adapter for HuggingFace models with preprocessing."""
    
    def __init__(self, model, processor, num_classes=21):
        super().__init__(model, num_classes)
        self.processor = processor
    
    def logits(self, x):
        """Get logits with custom preprocessing."""
        # Apply processor transformations
        inputs = self.processor(x, return_tensors="pt")
        
        # Move to model device
        for key, value in inputs.items():
            if isinstance(value, torch.Tensor):
                inputs[key] = value.to(self.model.device)
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        return outputs.logits
    
    def predictions(self, x):
        """Get predictions with custom preprocessing."""
        logits = self.logits(x)
        return torch.argmax(logits, dim=1)

# Load custom HuggingFace model
model_name = "nvidia/segformer-b0-finetuned-ade-512-512"
processor = AutoImageProcessor.from_pretrained(model_name)
model = AutoModelForSemanticSegmentation.from_pretrained(model_name)

# Create custom adapter
adapter = CustomHuggingFaceAdapter(model, processor, num_classes=150)

# Use in pipeline
pipeline = SegmentationRobustnessPipeline(
    model=adapter,
    dataset=dataset,
    attacks=[FGSM(adapter, eps=0.02)],
    metrics=[metrics.mean_iou],
    batch_size=2,  # Smaller batch size for large models
    device="cuda"
)
```

## ⚡ Performance Optimization

Optimize for large-scale evaluation:

```python
import torch
from segmentation_robustness_framework.pipeline import SegmentationRobustnessPipeline

# Optimize for performance
def create_optimized_pipeline(model, dataset, attacks, metrics):
    """Create pipeline with performance optimizations."""
    
    # Use mixed precision for faster computation
    if torch.cuda.is_available():
        model = model.half()  # Use FP16
    
    # Optimize DataLoader
    pipeline = SegmentationRobustnessPipeline(
        model=model,
        dataset=dataset,
        attacks=attacks,
        metrics=metrics,
        batch_size=8,  # Larger batch size
        device="cuda",
        num_workers=4,  # Parallel data loading
        pin_memory=True,  # Faster data transfer to GPU
        persistent_workers=True,  # Keep workers alive
        auto_resize_masks=True,
        output_formats=["json"],  # Faster than CSV for large results
        metric_precision=3  # Reduce precision for speed
    )
    
    return pipeline

# Memory optimization
def optimize_memory():
    """Optimize memory usage."""
    if torch.cuda.is_available():
        # Clear cache before large operations
        torch.cuda.empty_cache()
        
        # Set memory fraction
        torch.cuda.set_per_process_memory_fraction(0.8)
        
        # Enable memory efficient attention if available
        if hasattr(torch, 'backends') and hasattr(torch.backends, 'cuda'):
            torch.backends.cuda.enable_flash_sdp(True)

# Use optimizations
optimize_memory()
pipeline = create_optimized_pipeline(model, dataset, attacks, metrics)
results = pipeline.run()
```

## 🛡️ Error Handling and Logging

Implement robust error handling:

```python
import logging
from segmentation_robustness_framework.pipeline import SegmentationRobustnessPipeline

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)

class RobustPipeline:
    """Pipeline with robust error handling."""
    
    def __init__(self, model, dataset, attacks, metrics):
        self.model = model
        self.dataset = dataset
        self.attacks = attacks
        self.metrics = metrics
    
    def run_with_error_handling(self):
        """Run pipeline with comprehensive error handling."""
        try:
            # Create pipeline
            pipeline = SegmentationRobustnessPipeline(
                model=self.model,
                dataset=self.dataset,
                attacks=self.attacks,
                metrics=self.metrics,
                batch_size=4,
                device="cuda",
                output_dir="./robust_results"
            )
            
            logger.info("Starting evaluation...")
            results = pipeline.run()
            logger.info("Evaluation completed successfully!")
            
            return results
            
        except torch.cuda.OutOfMemoryError:
            logger.error("GPU out of memory. Trying with smaller batch size...")
            return self.run_with_smaller_batch()
            
        except Exception as e:
            logger.error(f"Evaluation failed: {e}")
            logger.error("Attempting recovery...")
            return self.run_with_recovery()
    
    def run_with_smaller_batch(self):
        """Run with reduced batch size."""
        try:
            pipeline = SegmentationRobustnessPipeline(
                model=self.model,
                dataset=self.dataset,
                attacks=self.attacks,
                metrics=self.metrics,
                batch_size=1,  # Minimal batch size
                device="cuda",
                output_dir="./robust_results_small_batch"
            )
            
            logger.info("Running with batch size 1...")
            return pipeline.run()
            
        except Exception as e:
            logger.error(f"Even small batch failed: {e}")
            return None
    
    def run_with_recovery(self):
        """Run with recovery mechanisms."""
        try:
            # Clear GPU memory
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            # Try CPU fallback
            logger.info("Trying CPU fallback...")
            pipeline = SegmentationRobustnessPipeline(
                model=self.model,
                dataset=self.dataset,
                attacks=self.attacks,
                metrics=self.metrics,
                batch_size=1,
                device="cpu",
                output_dir="./robust_results_cpu"
            )
            
            return pipeline.run()
            
        except Exception as e:
            logger.error(f"Recovery failed: {e}")
            return None

# Use robust pipeline
robust_pipeline = RobustPipeline(model, dataset, attacks, metrics)
results = robust_pipeline.run_with_error_handling()
```

## 🔗 Integration with External Tools

Integrate with popular ML tools:

```python
# Integration with Weights & Biases
import wandb
from segmentation_robustness_framework.pipeline import SegmentationRobustnessPipeline

def run_with_wandb():
    """Run evaluation with Weights & Biases logging."""
    
    # Initialize wandb
    wandb.init(project="segmentation-robustness", name="advanced-evaluation")
    
    # Log configuration
    config = {
        "model": "deeplabv3_resnet50",
        "dataset": "voc",
        "attacks": ["fgsm", "pgd"],
        "batch_size": 4,
        "device": "cuda"
    }
    wandb.config.update(config)
    
    # Run evaluation
    pipeline = SegmentationRobustnessPipeline(
        model=model,
        dataset=dataset,
        attacks=attacks,
        metrics=metrics,
        batch_size=4,
        device="cuda"
    )
    
    results = pipeline.run()
    
    # Log results - handle missing metrics gracefully
    for attack_name, attack_results in results.items():
        if attack_name.startswith('attack_'):
            # Log each metric if it exists
            for metric_name, metric_value in attack_results.items():
                if metric_value is not None:  # Handle None values
                    wandb.log({f"{attack_name}/{metric_name}": metric_value})
    
    # Log clean performance
    if "clean" in results:
        for metric_name, metric_value in results["clean"].items():
            if metric_value is not None:
                wandb.log({f"clean/{metric_name}": metric_value})
    
    # Log summary statistics
    summary = pipeline.get_summary()
    if "robustness_analysis" in summary:
        for attack_name, robustness in summary["robustness_analysis"].items():
            for metric_name, degradation in robustness.items():
                if degradation is not None:
                    wandb.log({f"robustness/{attack_name}/{metric_name}": degradation})
    
    wandb.finish()
    return results

# Integration with MLflow
import mlflow
import mlflow.pytorch

def run_with_mlflow():
    """Run evaluation with MLflow tracking."""
    
    mlflow.set_experiment("segmentation-robustness")
    
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params({
            "model": "deeplabv3_resnet50",
            "dataset": "voc",
            "batch_size": 4
        })
        
        # Run evaluation
        pipeline = SegmentationRobustnessPipeline(
            model=model,
            dataset=dataset,
            attacks=attacks,
            metrics=metrics,
            batch_size=4,
            device="cuda"
        )
        
        results = pipeline.run()
        
        # Log metrics - handle missing metrics gracefully
        for attack_name, attack_results in results.items():
            if attack_name.startswith('attack_'):
                for metric_name, metric_value in attack_results.items():
                    if metric_value is not None:  # Handle None values
                        mlflow.log_metric(f"{attack_name}_{metric_name}", metric_value)
        
        # Log clean performance
        if "clean" in results:
            for metric_name, metric_value in results["clean"].items():
                if metric_value is not None:
                    mlflow.log_metric(f"clean_{metric_name}", metric_value)
        
        # Log model (only if it's a standard PyTorch model)
        try:
            if hasattr(model, 'model'):  # If it's an adapter
                mlflow.pytorch.log_model(model.model, "segmentation_model")
            else:
                mlflow.pytorch.log_model(model, "segmentation_model")
        except Exception as e:
            print(f"Could not log model to MLflow: {e}")
        
        return results

# Integration with TensorBoard
import torch
from torch.utils.tensorboard import SummaryWriter
from segmentation_robustness_framework.pipeline import SegmentationRobustnessPipeline

def run_with_tensorboard():
    """Run evaluation with TensorBoard logging."""
    
    # Initialize TensorBoard writer
    writer = SummaryWriter('runs/segmentation_robustness')
    
    # Run evaluation
    pipeline = SegmentationRobustnessPipeline(
        model=model,
        dataset=dataset,
        attacks=attacks,
        metrics=metrics,
        batch_size=4,
        device="cuda"
    )
    
    results = pipeline.run()
    
    # Log results
    for attack_name, attack_results in results.items():
        if attack_name.startswith('attack_'):
            for metric_name, metric_value in attack_results.items():
                if metric_value is not None:
                    writer.add_scalar(f'{attack_name}/{metric_name}', metric_value, 0)
    
    # Log clean performance
    if "clean" in results:
        for metric_name, metric_value in results["clean"].items():
            if metric_value is not None:
                writer.add_scalar(f'clean/{metric_name}', metric_value, 0)
    
    writer.close()
    return results

# Integration with Neptune.ai
import neptune
from segmentation_robustness_framework.pipeline import SegmentationRobustnessPipeline

def run_with_neptune():
    """Run evaluation with Neptune.ai logging."""
    
    # Initialize Neptune
    run = neptune.init_run(project="your-workspace/segmentation-robustness")
    
    # Log configuration
    run["config/model"] = "deeplabv3_resnet50"
    run["config/dataset"] = "voc"
    run["config/batch_size"] = 4
    
    # Run evaluation
    pipeline = SegmentationRobustnessPipeline(
        model=model,
        dataset=dataset,
        attacks=attacks,
        metrics=metrics,
        batch_size=4,
        device="cuda"
    )
    
    results = pipeline.run()
    
    # Log results
    for attack_name, attack_results in results.items():
        if attack_name.startswith('attack_'):
            for metric_name, metric_value in attack_results.items():
                if metric_value is not None:
                    run[f"results/{attack_name}/{metric_name}"] = metric_value
    
    # Log clean performance
    if "clean" in results:
        for metric_name, metric_value in results["clean"].items():
            if metric_value is not None:
                run[f"results/clean/{metric_name}"] = metric_value
    
    run.stop()
    return results
```

## 🚀 Next Steps

- 📖 [User Guide](../user_guide.md) - Comprehensive usage guide
- 🔧 [API Reference](../api_reference/index.md) - Complete API documentation

