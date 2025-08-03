# Attacks API

This page documents the adversarial attack components of the Segmentation Robustness Framework.

## Attack Classes

::: segmentation_robustness_framework.attacks.attack
    options:
        show_signature_annotations: true

::: segmentation_robustness_framework.attacks.fgsm
    options:
        show_signature_annotations: true

::: segmentation_robustness_framework.attacks.pgd
    options:
        show_signature_annotations: true

::: segmentation_robustness_framework.attacks.rfgsm
    options:
        show_signature_annotations: true

::: segmentation_robustness_framework.attacks.tpgd
    options:
        show_signature_annotations: true

## Attack Overview

The framework provides a comprehensive suite of adversarial attacks designed to test the robustness of segmentation models. All attacks inherit from the `AdversarialAttack` base class.

### AdversarialAttack Base Class

The base class that all attacks must implement:

```python
from abc import ABC, abstractmethod
import torch

class AdversarialAttack(ABC):
    """Base class for adversarial attacks."""
    
    def __init__(self, model, eps=0.1, device="cuda"):
        self.model = model
        self.eps = eps
        self.device = device
    
    @abstractmethod
    def apply(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Apply the attack to input x with target y."""
        pass
```

### Available Attacks

#### FGSM (Fast Gradient Sign Method)

A simple but effective first-order attack:

```python
from segmentation_robustness_framework.attacks import FGSM

# Create FGSM attack
attack = FGSM(model, eps=0.02)

# Apply attack
adversarial_x = attack.apply(x, y)
```

**Parameters:**
- `eps`: Maximum perturbation magnitude (default: 2/255 ≈ 0.008)

#### PGD (Projected Gradient Descent)

A more powerful iterative attack:

```python
from segmentation_robustness_framework.attacks import PGD

# Create PGD attack
attack = PGD(
    model=model,
    eps=0.02,
    alpha=0.02,
    iters=10,
    targeted=False
)

# Apply attack
adversarial_x = attack.apply(x, y)
```

**Parameters:**
- `eps`: Maximum perturbation magnitude (default: 2/255 ≈ 0.008)
- `alpha`: Step size for each iteration (default: 2/255 ≈ 0.008)
- `iters`: Number of iterations (default: 10)
- `targeted`: Whether to perform targeted attack (default: False)

#### RFGSM (R-FGSM with Momentum)

FGSM with momentum for better convergence:

```python
from segmentation_robustness_framework.attacks import RFGSM

# Create RFGSM attack
attack = RFGSM(
    model=model,
    eps=0.1,
    alpha=0.01,
    iters=10,
    targeted=False
)

# Apply attack
adversarial_x = attack.apply(x, y)
```

**Parameters:**
- `eps`: Maximum perturbation magnitude (default: 0.1)
- `alpha`: Step size for each iteration (default: 0.01)
- `iters`: Number of iterations (default: 10)
- `targeted`: Whether to perform targeted attack (default: False)

#### TPGD (Targeted Projected Gradient Descent)

```python
from segmentation_robustness_framework.attacks import TPGD

# Create TPGD attack
attack = TPGD(
    model=model,
    eps=0.1,
    alpha=0.01,
    iters=10
)

# Apply attack
adversarial_x = attack.apply(x, y)
```

**Parameters:**
- `eps`: Maximum perturbation magnitude (default: 0.1)
- `alpha`: Step size for each iteration (default: 0.01)
- `iters`: Number of iterations (default: 10)

### Attack Configuration

Configure attacks in YAML configuration files:

```yaml
attacks:
  - name: fgsm
    eps: 0.02
  - name: pgd
    eps: 0.02
    alpha: 0.02
    iters: 10
    targeted: false
  - name: rfgsm
    eps: 0.02
    alpha: 0.02
    iters: 10
    targeted: false
  - name: tpgd
    eps: 0.02
    alpha: 0.02
    iters: 10
```

### Custom Attacks

Create custom attacks by inheriting from `AdversarialAttack`:

```python
from segmentation_robustness_framework.attacks import AdversarialAttack
import torch

class MyCustomAttack(AdversarialAttack):
    def __init__(self, model, eps=0.1, custom_param=1.0):
        super().__init__(model, eps)
        self.custom_param = custom_param
    
    def apply(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        """Apply your custom attack logic here."""
        # Your attack implementation
        x.requires_grad_(True)
        
        # Forward pass
        logits = self.model.logits(x)
        loss = torch.nn.functional.cross_entropy(logits, y)
        
        # Backward pass
        loss.backward()
        
        # Create perturbation
        perturbation = self.custom_param * x.grad.sign()
        
        # Apply perturbation with clipping
        adversarial_x = x + perturbation
        adversarial_x = torch.clamp(adversarial_x, 0, 1)
        
        return adversarial_x.detach()

# Use custom attack
attack = MyCustomAttack(model, eps=0.1, custom_param=0.5)
adversarial_x = attack.apply(x, y)
```

### Attack Registration

Register custom attacks for automatic discovery:

```python
from segmentation_robustness_framework.attacks import register_attack

@register_attack("my_custom_attack")
class MyCustomAttack(AdversarialAttack):
    def __init__(self, model, eps=0.1, custom_param=1.0):
        super().__init__(model, eps)
        self.custom_param = custom_param
    
    def apply(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        # Your attack implementation
        pass

# Now you can use it in configuration
# attacks:
#   - name: my_custom_attack
#     eps: 0.1
#     custom_param: 0.5
```

### Attack Usage in Pipeline

Attacks are automatically used by the pipeline:

```python
from segmentation_robustness_framework.pipeline import SegmentationRobustnessPipeline
from segmentation_robustness_framework.attacks import FGSM, PGD

# Create attacks
attacks = [
    FGSM(model, eps=0.1),
    PGD(model, eps=0.1, alpha=0.01, iters=10)
]

# Use in pipeline
pipeline = SegmentationRobustnessPipeline(
    model=model,
    dataset=dataset,
    attacks=attacks,
    metrics=[metrics.mean_iou],
    batch_size=4,
    device="cuda"
)

results = pipeline.run()
```

### Attack Evaluation

Evaluate attack effectiveness:

```python
# Compare clean vs adversarial performance
clean_iou = results['clean']['mean_iou']
fgsm_iou = results['attack_fgsm']['mean_iou']
pgd_iou = results['attack_pgd']['mean_iou']

print(f"Clean IoU: {clean_iou:.3f}")
print(f"FGSM IoU: {fgsm_iou:.3f}")
print(f"PGD IoU: {pgd_iou:.3f}")

# Calculate robustness
fgsm_robustness = fgsm_iou / clean_iou
pgd_robustness = pgd_iou / clean_iou

print(f"FGSM Robustness: {fgsm_robustness:.3f}")
print(f"PGD Robustness: {pgd_robustness:.3f}")
```

### Performance Considerations

- **GPU Acceleration**: All attacks support GPU acceleration
- **Memory Efficiency**: Optimized for batch processing
- **Gradient Computation**: Efficient gradient computation for iterative attacks
- **Convergence**: Automatic convergence detection for iterative attacks
