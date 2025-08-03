# API Reference

Complete API documentation for the Segmentation Robustness Framework.

## 📚 Table of Contents

1. [Core Pipeline](#core-pipeline)
2. [Model Loaders](#model-loaders)
3. [Datasets](#datasets)
4. [Attacks](#attacks)
5. [Metrics](#metrics)
6. [Adapters](#adapters)
7. [Utilities](#utilities)
8. [Configuration](#configuration)

## 🔧 Core Pipeline

### SegmentationRobustnessPipeline

The main pipeline class for running segmentation robustness evaluations.

```python
class SegmentationRobustnessPipeline:
    """Main pipeline for segmentation robustness evaluation."""
    
    def __init__(
        self,
        model: SegmentationModelProtocol,
        dataset: torch.utils.data.Dataset,
        attacks: list[AdversarialAttack],
        metrics: list[Callable],
        batch_size: int = 8,
        device: str = "cuda",
        output_dir: Optional[str] = None,
        auto_resize_masks: bool = True,
        metric_names: Optional[list[str]] = None,
        output_formats: list[str] = ["json", "csv"],
    ) -> None:
        """Initialize the pipeline.
        
        Args:
            model: Segmentation model with adapter
            dataset: Dataset instance
            attacks: List of attack instances
            metrics: List of metric functions
            batch_size: Batch size for evaluation
            device: Device for computation
            output_dir: Directory for saving results
            auto_resize_masks: Auto-resize masks to model output
            metric_names: Names for metrics
            output_formats: Output formats
        """
```

#### Methods

##### run()

```python
def run(self, save: bool = True, show: bool = False) -> dict[str, Any]:
    """Run the evaluation pipeline.
    
    Args:
        save: Whether to save results to files
        show: Whether to show visualizations
    
    Returns:
        Dictionary containing evaluation results
    """
```

##### get_summary()

```python
def get_summary(self) -> dict[str, Any]:
    """Get summary of evaluation results.
    
    Returns:
        Dictionary containing summary statistics
    """
```

##### print_summary()

```python
def print_summary(self) -> None:
    """Print summary of evaluation results to console."""
```

##### get_run_info()

```python
def get_run_info(self) -> dict[str, Any]:
    """Get information about the current run.
    
    Returns:
        Dictionary containing run information
    """
```

## 🏗️ Model Loaders

### UniversalModelLoader

Universal loader for different model types.

```python
class UniversalModelLoader:
    """Universal loader for different model types."""
    
    def __init__(self) -> None:
        """Initialize the universal loader."""
    
    def load_model(
        self,
        model_type: str,
        model_config: dict[str, Any],
        weights_path: Optional[str] = None,
        weight_type: str = "full",
    ) -> nn.Module:
        """Load a model.
        
        Args:
            model_type: Type of model (torchvision, smp, huggingface, custom)
            model_config: Model configuration dictionary
            weights_path: Path to weights file
            weight_type: Type of weights to load
        
        Returns:
            Loaded model
        """
```

### TorchvisionModelLoader

Loader for torchvision models.

```python
class TorchvisionModelLoader(BaseModelLoader):
    """Loader for torchvision segmentation models."""
    
    SUPPORTED_MODELS = {
        "deeplabv3_resnet50": tv_segmentation.deeplabv3_resnet50,
        "deeplabv3_resnet101": tv_segmentation.deeplabv3_resnet101,
        "deeplabv3_mobilenetv3_large": tv_segmentation.deeplabv3_mobilenet_v3_large,
        "fcn_resnet50": tv_segmentation.fcn_resnet50,
        "fcn_resnet101": tv_segmentation.fcn_resnet101,
        "lraspp_mobilenet_v3_large": tv_segmentation.lraspp_mobilenet_v3_large,
    }
    
    def load_model(self, model_config: dict[str, Any]) -> nn.Module:
        """Load a torchvision model.
        
        Args:
            model_config: Model configuration
        
        Returns:
            Loaded model
        """
```

### SMPModelLoader

Loader for SMP (Segmentation Models PyTorch) models.

```python
class SMPModelLoader(BaseModelLoader):
    """Loader for segmentation_models_pytorch models."""
    
    def load_model(self, model_config: dict[str, Any]) -> nn.Module:
        """Load an SMP model.
        
        Args:
            model_config: Model configuration
        
        Returns:
            Loaded model
        """
```

### HuggingFaceModelLoader

Loader for HuggingFace models.

```python
class HuggingFaceModelLoader(BaseModelLoader):
    """Loader for HuggingFace models."""
    
    def load_model(self, model_config: dict[str, Any]) -> "HFSegmentationBundle | nn.Module":
        """Load a HuggingFace model.
        
        Args:
            model_config: Model configuration
        
        Returns:
            Model bundle or model
        """
```

### CustomModelLoader

Loader for custom models.

```python
class CustomModelLoader(BaseModelLoader):
    """Loader for custom models."""
    
    def load_model(self, model_config: dict[str, Any]) -> nn.Module:
        """Load a custom model.
        
        Args:
            model_config: Model configuration
        
        Returns:
            Loaded model
        """
```

## 📊 Datasets

### VOCSegmentation

Pascal VOC 2012 dataset.

```python
class VOCSegmentation(Dataset):
    """Pascal VOC 2012 dataset for semantic segmentation."""
    
    def __init__(
        self,
        split: str,
        root: Optional[Union[Path, str]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
    ) -> None:
        """Initialize VOC dataset.
        
        Args:
            split: Dataset split (train, val, trainval)
            root: Dataset root directory
            transform: Image transformations
            target_transform: Mask transformations
            download: Whether to download dataset
        """
```

### ADE20K

ADE20K dataset for scene parsing.

```python
class ADE20K(Dataset):
    """ADE20K dataset for semantic segmentation."""
    
    def __init__(
        self,
        split: str,
        root: Optional[Union[Path, str]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
    ) -> None:
        """Initialize ADE20K dataset.
        
        Args:
            split: Dataset split (train, val)
            root: Dataset root directory
            transform: Image transformations
            target_transform: Mask transformations
            download: Whether to download dataset
        """
```

### Cityscapes

Cityscapes dataset for urban scene understanding.

```python
class Cityscapes(Dataset):
    """Cityscapes dataset for semantic segmentation.

    Note:
        The Cityscapes dataset cannot be downloaded automatically because it requires user authorization on the official website.
        You must register and download the dataset manually from https://www.cityscapes-dataset.com/ and place it in the specified root directory.
    """
    
    def __init__(
        self,
        root: Union[Path, str],
        split: str = "train",
        mode: str = "fine",
        target_type: str = "semantic",
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        """Initialize Cityscapes dataset.
        
        Args:
            root: Dataset root directory
            split: Dataset split (train, val, test, train_extra)
            mode: Annotation mode (fine, coarse)
            target_type: Target type (semantic, instance, color, polygon)
            transform: Image transformations
            target_transform: Target transformations
        """
```

### StanfordBackground

Stanford Background dataset.

```python
class StanfordBackground(Dataset):
    """Stanford Background dataset for semantic segmentation."""
    
    def __init__(
        self,
        root: Optional[Union[Path, str]] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        download: bool = True,
    ) -> None:
        """Initialize Stanford Background dataset.
        
        Args:
            root: Dataset root directory
            transform: Image transformations
            target_transform: Mask transformations
            download: Whether to download dataset
        """
```

## ⚔️ Attacks

### AdversarialAttack

Base class for adversarial attacks.

```python
class AdversarialAttack(ABC):
    """Base class for adversarial attacks."""
    
    def __init__(self, model: nn.Module) -> None:
        """Initialize attack.
        
        Args:
            model: Model to attack
        """
    
    @abstractmethod
    def apply(self, image: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Apply the attack.
        
        Args:
            image: Input images [B, C, H, W]
            labels: Target labels [B, H, W]
        
        Returns:
            Adversarial images [B, C, H, W]
        """
```

### FGSM

Fast Gradient Sign Method attack.

```python
class FGSM(AdversarialAttack):
    """Fast Gradient Sign Method attack."""
    
    def __init__(self, model: nn.Module, eps: float = 2/255) -> None:
        """Initialize FGSM attack.
        
        Args:
            model: Model to attack
            eps: Maximum perturbation magnitude
        """
    
    def apply(self, image: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Apply FGSM attack."""
```

### PGD

Projected Gradient Descent attack.

```python
class PGD(AdversarialAttack):
    """Projected Gradient Descent attack."""
    
    def __init__(
        self,
        model: nn.Module,
        eps: float = 2/255,
        alpha: float = 2/255,
        iters: int = 10,
        targeted: bool = False,
    ) -> None:
        """Initialize PGD attack.
        
        Args:
            model: Model to attack
            eps: Maximum perturbation magnitude
            alpha: Step size
            iters: Number of iterations
            targeted: Whether to perform targeted attack
        """
    
    def apply(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Apply PGD attack."""
```

### RFGSM

Random + Fast Gradient Sign Method attack.

```python
class RFGSM(AdversarialAttack):
    """Random + Fast Gradient Sign Method attack."""
    
    def __init__(
        self,
        model: nn.Module,
        eps: float = 8/255,
        alpha: float = 2/255,
        iters: int = 10,
        targeted: bool = False,
    ) -> None:
        """Initialize RFGSM attack.
        
        Args:
            model: Model to attack
            eps: Maximum perturbation magnitude
            alpha: Step size
            iters: Number of iterations
            targeted: Whether to perform targeted attack
        """
    
    def apply(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Apply RFGSM attack."""
```

### TPGD

Theoretically Principled PGD attack.

```python
class TPGD(AdversarialAttack):
    """Theoretically Principled PGD attack."""
    
    def __init__(
        self,
        model: nn.Module,
        eps: float = 8/255,
        alpha: float = 2/255,
        iters: int = 10,
    ) -> None:
        """Initialize TPGD attack.
        
        Args:
            model: Model to attack
            eps: Maximum perturbation magnitude
            alpha: Step size
            iters: Number of iterations
        """
    
    def apply(self, images: torch.Tensor, labels: torch.Tensor = None) -> torch.Tensor:
        """Apply TPGD attack."""
```

## 📈 Metrics

### MetricsCollection

Collection of evaluation metrics.

```python
class MetricsCollection:
    """Collection of evaluation metrics."""
    
    def __init__(self, num_classes: int, ignore_index: int = 255) -> None:
        """Initialize metrics collection.
        
        Args:
            num_classes: Number of classes
            ignore_index: Index to ignore in evaluation
        """
    
    def mean_iou(self, targets: torch.Tensor, preds: torch.Tensor, average: str = "macro") -> float:
        """Compute mean Intersection over Union.
        
        Args:
            targets: Ground truth labels
            preds: Predicted labels
            average: Averaging strategy (macro, micro)
        
        Returns:
            Mean IoU score
        """
    
    def pixel_accuracy(self, targets: torch.Tensor, preds: torch.Tensor) -> float:
        """Compute pixel accuracy.
        
        Args:
            targets: Ground truth labels
            preds: Predicted labels
        
        Returns:
            Pixel accuracy score
        """
    
    def precision(self, targets: torch.Tensor, preds: torch.Tensor, average: str = "macro") -> float:
        """Compute precision.
        
        Args:
            targets: Ground truth labels
            preds: Predicted labels
            average: Averaging strategy (macro, micro)
        
        Returns:
            Precision score
        """
    
    def recall(self, targets: torch.Tensor, preds: torch.Tensor, average: str = "macro") -> float:
        """Compute recall.
        
        Args:
            targets: Ground truth labels
            preds: Predicted labels
            average: Averaging strategy (macro, micro)
        
        Returns:
            Recall score
        """
    
    def dice_score(self, targets: torch.Tensor, preds: torch.Tensor, average: str = "macro") -> float:
        """Compute Dice score.
        
        Args:
            targets: Ground truth labels
            preds: Predicted labels
            average: Averaging strategy (macro, micro)
        
        Returns:
            Dice score
        """
```

## 🔧 Adapters

### SegmentationModelProtocol

Protocol for segmentation model adapters.

```python
class SegmentationModelProtocol(Protocol):
    """Protocol for segmentation model adapters."""
    
    num_classes: int
    
    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logits.
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Logits tensor [B, num_classes, H, W]
        """
    
    def predictions(self, x: torch.Tensor) -> torch.Tensor:
        """Return predicted labels.
        
        Args:
            x: Input tensor [B, C, H, W]
        
        Returns:
            Predicted labels [B, H, W]
        """
```

### TorchvisionAdapter

Adapter for torchvision models.

```python
class TorchvisionAdapter(torch.nn.Module, SegmentationModelProtocol):
    """Adapter for torchvision segmentation models."""
    
    def __init__(self, model: torch.nn.Module) -> None:
        """Initialize adapter.
        
        Args:
            model: Torchvision model
        """
    
    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits."""
    
    def predictions(self, x: torch.Tensor) -> torch.Tensor:
        """Return predictions."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
```

### SMPAdapter

Adapter for SMP models.

```python
class SMPAdapter(torch.nn.Module, SegmentationModelProtocol):
    """Adapter for SMP models."""
    
    def __init__(self, model: torch.nn.Module) -> None:
        """Initialize adapter.
        
        Args:
            model: SMP model
        """
    
    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits."""
    
    def predictions(self, x: torch.Tensor) -> torch.Tensor:
        """Return predictions."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
```

### HuggingFaceAdapter

Adapter for HuggingFace models.

```python
class HuggingFaceAdapter(torch.nn.Module, SegmentationModelProtocol):
    """Adapter for HuggingFace models."""
    
    def __init__(self, model: torch.nn.Module) -> None:
        """Initialize adapter.
        
        Args:
            model: HuggingFace model
        """
    
    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits."""
    
    def predictions(self, x: torch.Tensor) -> torch.Tensor:
        """Return predictions."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
```

### CustomAdapter

Template adapter for custom models.

```python
class CustomAdapter(torch.nn.Module, SegmentationModelProtocol):
    """Template adapter for custom models."""
    
    def __init__(self, model: torch.nn.Module, num_classes: int = 1) -> None:
        """Initialize adapter.
        
        Args:
            model: Custom model
            num_classes: Number of classes
        """
    
    def logits(self, x: torch.Tensor) -> torch.Tensor:
        """Return logits."""
    
    def predictions(self, x: torch.Tensor) -> torch.Tensor:
        """Return predictions."""
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass."""
```

## 🛠️ Utilities

### Model Utils

```python
def get_model_output_size(model, input_shape: tuple[int, int, int], device: str = "cpu") -> tuple[int, int]:
    """Get model output size.
    
    Args:
        model: Model to analyze
        input_shape: Input shape (C, H, W)
        device: Device to use
    
    Returns:
        Output size (H, W)
    """

def get_huggingface_output_size(model, input_shape: tuple[int, int, int], device: str = "cpu") -> tuple[int, int]:
    """Get HuggingFace model output size.
    
    Args:
        model: HuggingFace model
        input_shape: Input shape (C, H, W)
        device: Device to use
    
    Returns:
        Output size (H, W)
    """
```

### Image Preprocessing

```python
def register_dataset_colors(dataset_name: str, colors: list[tuple[int, int, int]]) -> None:
    """Register color palette for dataset.
    
    Args:
        dataset_name: Name of the dataset
        colors: List of RGB colors
    """

def prepare_inputs(sample, maybe_bundle, device="cuda"):
    """Prepare inputs for model inference.
    
    Args:
        sample: Input sample
        maybe_bundle: Optional model bundle
        device: Device to use
    
    Returns:
        Prepared inputs
    """

def get_preprocessing_fn(image_shape: list[int], dataset_name: Optional[str] = None) -> Callable:
    """Get preprocessing function.
    
    Args:
        image_shape: Target image shape
        dataset_name: Name of the dataset
    
    Returns:
        Preprocessing function
    """
```

### Dataset Utils

```python
def download(url: str, dest_dir: str, md5: str | None = None) -> str:
    """Download a file.
    
    Args:
        url: URL to download
        dest_dir: Destination directory
        md5: Expected MD5 hash
    
    Returns:
        Path to downloaded file
    """

def extract(file_path: str, dest_dir: str) -> None:
    """Extract an archive.
    
    Args:
        file_path: Path to archive
        dest_dir: Destination directory
    """

def get_cache_dir(dataset_name: str) -> Path:
    """Get cache directory for dataset.
    
    Args:
        dataset_name: Name of the dataset
    
    Returns:
        Cache directory path
    """
```

### Visualization

```python
def visualize_images(
    image: Tensor,
    ground_truth: Tensor,
    mask: Tensor,
    adv_mask: Tensor,
    dataset_name: str,
    denormalize_image: bool = True,
    title: str = None,
    show: bool = False,
    save: bool = False,
    save_dir: str = None,
) -> None:
    """Visualize segmentation results.
    
    Args:
        image: Input image
        ground_truth: Ground truth mask
        mask: Predicted mask
        adv_mask: Adversarial mask
        dataset_name: Name of the dataset
        denormalize_image: Whether to denormalize image
        title: Plot title
        show: Whether to show plot
        save: Whether to save plot
        save_dir: Directory to save plot
    """

def visualize_metrics(
    json_data: Union[Path, str, dict[str, Any]],
    attack_name: str,
    attack_param: str,
    metric_names: Union[str, list[str]],
) -> None:
    """Visualize metrics.
    
    Args:
        json_data: JSON data or path
        attack_name: Name of the attack
        attack_param: Attack parameter
        metric_names: Names of metrics to visualize
    """

def print_clean_metrics(
    json_data: Union[Path, str, dict[str, Any]],
    metric_names: Union[str, list[str]],
) -> None:
    """Print clean metrics.
    
    Args:
        json_data: JSON data or path
        metric_names: Names of metrics to print
    """
```

## ⚙️ Configuration

### Configuration Classes

```python
class TorchvisionConfig(BaseModel):
    """Configuration for torchvision models."""
    type: Literal["torchvision"]
    name: str
    num_classes: Optional[int] = 21
    weights: Optional[Any] = None
    device: Optional[str] = "cpu"

class SMPConfig(BaseModel):
    """Configuration for SMP models."""
    type: Literal["smp"]
    architecture: str
    encoder_name: Optional[str] = "resnet50"
    encoder_weights: Optional[str] = "imagenet"
    classes: Optional[int] = 1
    activation: Optional[str] = None
    weights_path: Optional[Union[str, Path]] = None
    checkpoint: Optional[str] = None
    device: Optional[str] = "cpu"

class HuggingFaceConfig(BaseModel):
    """Configuration for HuggingFace models."""
    type: Literal["huggingface"]
    model_name: str
    num_labels: Optional[int] = None
    trust_remote_code: Optional[bool] = False
    weights_path: Optional[Union[str, Path]] = None
    device: Optional[str] = "cpu"
    task: Optional[str] = "semantic_segmentation"
    return_processor: Optional[bool] = True
    config_overrides: Optional[dict[str, Any]] = Field(default_factory=dict)
    processor_overrides: Optional[dict[str, Any]] = Field(default_factory=dict)

class CustomConfig(BaseModel):
    """Configuration for custom models."""
    type: Literal["custom"]
    model_class: Union[str, Callable[..., Any]]
    model_args: Optional[list[Any]] = Field(default_factory=list)
    model_kwargs: Optional[dict[str, Any]] = Field(default_factory=dict)
    weights_path: Optional[Union[str, Path]] = None
    device: Optional[str] = "cpu"

ModelConfig = Union[TorchvisionConfig, SMPConfig, HuggingFaceConfig, CustomConfig]
```

### Configuration Functions

```python
def load_config(config_path: str) -> dict[str, Any]:
    """Load configuration from file.
    
    Args:
        config_path: Path to configuration file
    
    Returns:
        Configuration dictionary
    """

def validate_config(config: dict[str, Any]) -> ValidationResult:
    """Validate configuration.
    
    Args:
        config: Configuration dictionary
    
    Returns:
        Validation result
    """

def save_config(config: dict[str, Any], config_path: str) -> None:
    """Save configuration to file.
    
    Args:
        config: Configuration dictionary
        config_path: Path to save configuration
    """
```

## 🔗 Registry System

### Dataset Registry

```python
DATASET_REGISTRY: dict[str, Callable] = {}

def register_dataset(name: str) -> Callable:
    """Register a dataset class.
    
    Args:
        name: Name to register the dataset under
    
    Returns:
        Decorator function
    """

def get_dataset(name: str) -> Callable:
    """Get a registered dataset class.
    
    Args:
        name: Name of the dataset
    
    Returns:
        Dataset class
    """
```

### Attack Registry

```python
ATTACK_REGISTRY: dict[str, Callable] = {}

def register_attack(name: str) -> Callable:
    """Register an attack class.
    
    Args:
        name: Name to register the attack under
    
    Returns:
        Decorator function
    """

def get_attack(name: str) -> Callable:
    """Get a registered attack class.
    
    Args:
        name: Name of the attack
    
    Returns:
        Attack class
    """
```

### Adapter Registry

```python
ADAPTER_REGISTRY: dict[str, type] = {}

def register_adapter(name: str) -> Callable:
    """Register an adapter class.
    
    Args:
        name: Name to register the adapter under
    
    Returns:
        Decorator function
    """

def get_adapter(name: str) -> type:
    """Get a registered adapter class.
    
    Args:
        name: Name of the adapter
    
    Returns:
        Adapter class
    """
```

## 📋 CLI Commands

### List Available Components

```bash
# List available attacks
python -m segmentation_robustness_framework list_attacks

# List available datasets
python -m segmentation_robustness_framework list_datasets
```

### Run Evaluation

```bash
# Run evaluation with configuration file
python -m segmentation_robustness_framework run --config config.yaml

# Run evaluation with command line arguments
python -m segmentation_robustness_framework run \
    --model torchvision \
    --model-name deeplabv3_resnet50 \
    --dataset voc \
    --attack fgsm \
    --eps 0.008
```

---

**This API reference covers all public interfaces of the framework. For more information:**

- 📖 [User Guide](user_guide.md) - General framework usage
- 🔧 [Custom Components](custom_components.md) - Creating custom components
- 🧪 [Advanced Usage](advanced_usage.md) - Advanced features 