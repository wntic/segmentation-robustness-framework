# Documentation Guide

This guide covers documentation standards and practices for the Segmentation Robustness Framework.

## 📝 Documentation Philosophy

### Documentation Principles

- **Clear and Concise**: Write for clarity, not verbosity
- **User-Focused**: Write from the user's perspective
- **Comprehensive**: Cover all public APIs and features
- **Up-to-Date**: Keep documentation synchronized with code
- **Accessible**: Use clear language and proper formatting

### Documentation Types

1. **API Documentation**: Function and class documentation
2. **User Guides**: How-to guides and tutorials
3. **Conceptual Documentation**: Architecture and design explanations
4. **Examples**: Code examples and use cases
5. **Reference Documentation**: Complete API reference

## 📚 Documentation Structure

```
docs/
├── index.md                    # Main documentation page
├── installation.md             # Installation guide
├── quick_start.md             # Quick start tutorial
├── user_guide.md              # Comprehensive user guide
├── core_concepts.md           # Framework concepts
├── api_reference/             # API documentation
│   ├── index.md              # API overview
│   ├── pipeline.md           # Pipeline API
│   ├── attacks.md            # Attacks API
│   ├── metrics.md            # Metrics API
│   └── adapters.md           # Adapters API
├── examples/                  # Code examples
│   ├── basic_examples.md     # Basic usage examples
│   └── advanced_examples.md  # Advanced examples
├── contributing/              # Contributing documentation
│   ├── index.md              # Contributing guide
│   ├── development_setup.md  # Development setup
│   ├── testing_guide.md      # Testing guidelines
│   └── documentation_guide.md # This file
└── img/                      # Documentation images
    ├── architecture.png      # Framework architecture
    └── workflow.png          # Usage workflow
```

## ✍️ Writing Documentation

### Documentation Standards

#### Markdown Formatting

```markdown
# Main Heading (H1)

## Section Heading (H2)

### Subsection Heading (H3)

#### Sub-subsection Heading (H4)

**Bold text** for emphasis

*Italic text* for secondary emphasis

`code` for inline code

``​`python
# Code blocks with syntax highlighting
def example_function():
    return "Hello, World!"
``​`

> Blockquotes for important notes or warnings
```

#### Code Examples

```python
# Good example - complete and runnable
from segmentation_robustness_framework.pipeline import SegmentationRobustnessPipeline
from segmentation_robustness_framework.attacks import FGSM

# Load model
model = load_model("deeplabv3_resnet50")

# Create attack
attack = FGSM(model, eps=0.02)

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

#### API Documentation

```python
def evaluate_model(
    model: SegmentationModelProtocol,
    dataset: torch.utils.data.Dataset,
    metrics: list[Callable]
) -> dict[str, float]:
    """Evaluate a segmentation model.
    
    This function evaluates a segmentation model on a dataset using
    specified metrics. The evaluation includes both clean and adversarial
    performance.
    
    Args:
        model: The segmentation model to evaluate. Must implement the
            SegmentationModelProtocol interface.
        dataset: Dataset for evaluation. Should return (image, mask) pairs.
        metrics: List of metric functions. Each function should accept
            (targets, predictions) and return a float.
            
    Returns:
        Dictionary containing evaluation results with keys:
            - 'clean': Clean performance metrics
            - 'attack_<name>': Attack-specific metrics
            
    Raises:
        ValueError: If model or dataset is invalid.
        RuntimeError: If evaluation fails.
        
    Example:
        ```python
        model = load_model("deeplabv3_resnet50")
        dataset = VOCSegmentation(split="val")
        metrics = [mean_iou, pixel_accuracy]
        
        results = evaluate_model(model, dataset, metrics)
        print(f"Clean IoU: {results['clean']['mean_iou']:.3f}")
        ```
    """
    # Implementation here
    pass
```

### Documentation Style Guide

#### Tone and Voice

- **Use active voice**: "The function returns..." not "The function is returned by..."
- **Be direct and clear**: Avoid unnecessary words
- **Use present tense**: "The function processes..." not "The function will process..."
- **Be consistent**: Use consistent terminology throughout

#### Language Guidelines

- **Use US English**: color, center, analyze
- **Avoid jargon**: Explain technical terms
- **Be inclusive**: Use inclusive language
- **Be precise**: Use exact terms and measurements

#### Formatting Guidelines

- **Use proper headings**: H1 for page title, H2 for sections, etc.
- **Use lists appropriately**: Bullet points for unordered lists, numbers for steps
- **Use code blocks**: For all code examples
- **Use emphasis sparingly**: Bold for important terms, italic for emphasis

## 📖 API Documentation

### Function Documentation

```python
def process_images(
    images: torch.Tensor,
    target_size: tuple[int, int],
    normalize: bool = True
) -> torch.Tensor:
    """Process images to target size.
    
    Resize and optionally normalize images to the specified target size.
    The function supports both CPU and GPU tensors.
    
    Args:
        images: Input images [B, C, H, W]. Can be on CPU or GPU.
        target_size: Target size as (height, width) in pixels.
        normalize: Whether to normalize images to [0, 1] range.
            Defaults to True.
            
    Returns:
        Processed images [B, C, H, W] with same device as input.
        
    Raises:
        ValueError: If target_size contains non-positive values.
        RuntimeError: If image processing fails.
        
    Example:
        ```python
        images = torch.randn(4, 3, 224, 224)
        processed = process_images(images, (512, 512))
        assert processed.shape == (4, 3, 512, 512)
        ```
        
    Note:
        This function modifies images in-place for efficiency.
    """
    # Implementation
    pass
```

### Class Documentation

```python
class SegmentationRobustnessPipeline:
    """Pipeline for evaluating segmentation models under adversarial attacks.
    
    This pipeline provides a unified interface for evaluating segmentation
    models on clean and adversarial images. It supports multiple attacks,
    metrics, and output formats.
    
    The pipeline automatically handles:
    - Model evaluation on clean images
    - Attack generation and evaluation
    - Metric computation and aggregation
    - Result saving and visualization
    
    Attributes:
        model: The segmentation model to evaluate.
        dataset: Dataset for evaluation.
        attacks: List of attack instances.
        metrics: List of metric functions.
        batch_size: Batch size for evaluation.
        device: Device to use for computation.
        output_dir: Directory to save results.
        
    Example:
        ```python
        pipeline = SegmentationRobustnessPipeline(
            model=model,
            dataset=dataset,
            attacks=[FGSM(model, eps=0.02)],
            metrics=[mean_iou, pixel_accuracy],
            batch_size=4,
            device="cuda"
        )
        
        results = pipeline.run()
        pipeline.print_summary()
        ```
    """
    
    def __init__(
        self,
        model: SegmentationModelProtocol,
        dataset: torch.utils.data.Dataset,
        attacks: list,
        metrics: list[Callable],
        batch_size: int = 8,
        device: str = "cpu",
        output_dir: Optional[str] = None
    ):
        """Initialize the segmentation robustness pipeline.
        
        Args:
            model: Segmentation model (adapter-wrapped).
            dataset: Dataset object for evaluation.
            attacks: List of attack instances.
            metrics: List of metric functions or classes.
            batch_size: Batch size for evaluation. Defaults to 8.
            device: Device to use for computation. Defaults to "cpu".
            output_dir: Directory to save results. If None, uses "./runs".
        """
        # Implementation
        pass
```

### Module Documentation

```python
"""Segmentation Robustness Framework - Attacks Module.

This module provides implementations of adversarial attacks for semantic
segmentation models. All attacks follow a common interface and can be
used interchangeably in the evaluation pipeline.

Available Attacks:
    - FGSM: Fast Gradient Sign Method
    - PGD: Projected Gradient Descent
    - RFGSM: R-FGSM with momentum
    - TPGD: Two-Phase Gradient Descent

Example:
    ```python
    from segmentation_robustness_framework.attacks import FGSM, PGD
    
    # Create attacks
    fgsm_attack = FGSM(model, eps=0.02)
    pgd_attack = PGD(model, eps=0.02, alpha=0.01, iters=10)
    
    # Use in pipeline
    pipeline = SegmentationRobustnessPipeline(
        model=model,
        dataset=dataset,
        attacks=[fgsm_attack, pgd_attack],
        metrics=metrics
    )
    ```
"""

# Module implementation
```

## 📋 User Guide Documentation

### Structure

1. **Introduction**: What the feature does and why it's useful
2. **Prerequisites**: What users need to know or have
3. **Step-by-step instructions**: Clear, numbered steps
4. **Examples**: Real-world usage examples
5. **Troubleshooting**: Common issues and solutions

### Example User Guide Section

```markdown
## Adding Custom Attacks

This guide shows you how to create and use custom adversarial attacks
with the Segmentation Robustness Framework.

### Prerequisites

- Basic knowledge of PyTorch
- Understanding of adversarial attacks
- Familiarity with the framework's attack interface

### Step 1: Create Your Attack Class

First, create a new attack class that inherits from `AdversarialAttack`:

``​`python
from segmentation_robustness_framework.attacks import AdversarialAttack
from segmentation_robustness_framework.attacks.registry import register_attack

@register_attack("custom_attack")
class CustomAttack(AdversarialAttack):
    """Custom adversarial attack implementation."""
    
    def __init__(self, model: nn.Module, eps: float = 0.02):
        super().__init__(model)
        self.eps = eps
    
    def apply(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Apply custom attack to images."""
        # Your attack implementation here
        return adversarial_images
``​`

### Step 2: Implement the Attack Logic

The `apply` method should:
- Take input images and labels
- Generate adversarial perturbations
- Return adversarial images

### Step 3: Use Your Attack

Register and use your attack in the pipeline:

``​`python
# Create attack instance
custom_attack = CustomAttack(model, eps=0.03)

# Use in pipeline
pipeline = SegmentationRobustnessPipeline(
    model=model,
    dataset=dataset,
    attacks=[custom_attack],
    metrics=metrics
)

results = pipeline.run()
``​`

### Examples

#### Simple Random Attack

``​`python
@register_attack("random_attack")
class RandomAttack(AdversarialAttack):
    """Simple random perturbation attack."""
    
    def __init__(self, model: nn.Module, eps: float = 0.02):
        super().__init__(model)
        self.eps = eps
    
    def apply(self, images: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Apply random perturbation."""
        perturbation = torch.randn_like(images) * self.eps
        adv_images = torch.clamp(images + perturbation, 0, 1)
        return adv_images
``​`

### Troubleshooting

**Issue**: Attack not registered
- **Solution**: Make sure to use the `@register_attack` decorator

**Issue**: Attack produces invalid images
- **Solution**: Ensure images are in [0, 1] range using `torch.clamp`

**Issue**: Attack is too slow
- **Solution**: Use vectorized operations and avoid loops
```

## 🖼️ Visual Documentation

### Screenshots and Diagrams

- **Use high-quality images**: PNG format, appropriate resolution
- **Include alt text**: For accessibility
- **Keep images up-to-date**: Update when UI changes
- **Use consistent styling**: Same color scheme and fonts

## 🔄 Documentation Maintenance

### Keeping Documentation Current

1. **Update with code changes**: Modify docs when code changes
2. **Review regularly**: Schedule regular documentation reviews
3. **Test examples**: Ensure all code examples work
4. **Check links**: Verify all internal and external links
5. **Update version numbers**: Keep version info current

### Documentation Review Process

1. **Self-review**: Review your own documentation
2. **Peer review**: Have others review your documentation
3. **User testing**: Test with actual users
4. **Continuous improvement**: Gather feedback and improve

### Documentation Tools

```bash
# Build documentation locally
mkdocs build

# Serve documentation locally
mkdocs serve

# Check for broken links
linkchecker docs/

# Validate markdown
markdownlint docs/
```

### API Reference Generation

The API reference documentation is automatically generated from docstrings using MkDocs and the `mkdocstrings` plugin. This ensures that the API documentation stays synchronized with the code.

#### Setup for API Documentation

```bash
# Install mkdocstrings plugin
pip install mkdocstrings[python]

# Install additional plugins for better documentation
pip install mkdocs-material
pip install mkdocs-git-revision-date-localized-plugin
```

#### Configuration

The API documentation is configured in `mkdocs.yml`:

```yaml
plugins:
  - search
  - mkdocstrings:
      default_handler: python
      handlers:
        python:
          paths: [segmentation_robustness_framework]
          options:
            show_source: true
            show_root_heading: true
            show_category_heading: true
            show_signature_annotations: true
            show_bases: true
            heading_level: 1
            members_order: source
            docstring_style: google
            preload_modules: true
```

#### Generating API Reference

```bash
# Generate API documentation from docstrings
mkdocs build

# The API reference will be automatically generated in:
# - docs/api_reference/pipeline.md
# - docs/api_reference/attacks.md
# - docs/api_reference/metrics.md
# - docs/api_reference/adapters.md
```

#### Docstring Requirements for API Generation

To ensure proper API documentation generation, follow these docstring standards:

1. **Use Google-style docstrings** (required by mkdocstrings)
2. **Include type hints** in function signatures
3. **Document all parameters** with types and descriptions
4. **Document return values** with types
5. **Document exceptions** that may be raised
6. **Include usage examples** in docstrings

Example of a well-documented function for API generation:

```python
def evaluate_model(
    model: SegmentationModelProtocol,
    dataset: torch.utils.data.Dataset,
    metrics: list[Callable[[torch.Tensor, torch.Tensor], float]]
) -> dict[str, float]:
    """Evaluate a segmentation model on a dataset.
    
    This function evaluates a segmentation model on a dataset using
    specified metrics. The evaluation includes both clean and adversarial
    performance.
    
    Args:
        model: The segmentation model to evaluate. Must implement the
            SegmentationModelProtocol interface.
        dataset: Dataset for evaluation. Should return (image, mask) pairs.
        metrics: List of metric functions. Each function should accept
            (targets, predictions) and return a float.
            
    Returns:
        Dictionary containing evaluation results with keys:
            - 'clean': Clean performance metrics
            - 'attack_<name>': Attack-specific metrics
            
    Raises:
        ValueError: If model or dataset is invalid.
        RuntimeError: If evaluation fails.
        
    Example:
        ```python
        model = load_model("deeplabv3_resnet50")
        dataset = VOCSegmentation(split="val")
        metrics = [mean_iou, pixel_accuracy]
        
        results = evaluate_model(model, dataset, metrics)
        print(f"Clean IoU: {results['clean']['mean_iou']:.3f}")
        ```
    """
    # Implementation
    pass
```

#### API Documentation Structure

The generated API documentation follows this structure:

```
docs/api_reference/
├── index.md              # API overview and navigation
├── pipeline.md           # Pipeline classes and functions
├── attacks.md            # Attack implementations
├── metrics.md            # Metric functions and classes
├── adapters.md           # Model adapters
├── datasets.md           # Dataset classes
└── loaders.md            # Model and dataset loaders
```

#### Customizing API Documentation

You can customize the API documentation generation by:

1. **Adding custom handlers** in `mkdocs.yml`
2. **Filtering modules** to include/exclude specific components
3. **Customizing templates** for different documentation styles
4. **Adding cross-references** between related functions

Example customization:

```yaml
plugins:
  - mkdocstrings:
      default_handler: python
      handlers:
        python:
          options:
            show_source: true
            show_root_heading: true
            show_category_heading: true
            show_signature_annotations: true
            show_bases: true
            heading_level: 1
            members_order: source
            docstring_style: google
            preload_modules: true
            filters: ["!^_"]  # Exclude private members
            members: ["public", "special"]  # Include public and special methods
```

#### Maintaining API Documentation

To keep API documentation up-to-date:

1. **Update docstrings** when changing function signatures
2. **Add new functions** with proper docstrings
3. **Run documentation build** to regenerate API reference
4. **Review generated documentation** for accuracy
5. **Test documentation locally** before committing

```bash
# Regenerate API documentation
mkdocs build

# Check for documentation errors
mkdocs build --strict

# Serve documentation locally for review
mkdocs serve
```

## 🚀 Best Practices

### Writing Guidelines

1. **Start with the user**: Write from the user's perspective
2. **Be specific**: Use concrete examples and numbers
3. **Use active voice**: "The function returns..." not "The function is returned by..."
4. **Be consistent**: Use consistent terminology and formatting
5. **Test your examples**: Ensure all code examples work

### Organization Guidelines

1. **Logical structure**: Organize content logically
2. **Progressive disclosure**: Start simple, add complexity
3. **Cross-references**: Link related content
4. **Searchable**: Use clear headings and keywords
5. **Scannable**: Use lists, tables, and formatting

### Maintenance Guidelines

1. **Version control**: Track documentation changes
2. **Regular reviews**: Schedule documentation reviews
3. **User feedback**: Gather and incorporate feedback
4. **Automation**: Use tools to check documentation
5. **Continuous improvement**: Always look for ways to improve

## 🎯 Next Steps

After reading this documentation guide:

1. **Review existing documentation** to understand current standards
2. **Practice writing documentation** using the templates provided
3. **Contribute documentation** for new features or improvements
4. **Help maintain documentation** by reporting issues or suggesting improvements
5. **Share knowledge** with other contributors

Happy documenting! 📚
