from dataclasses import dataclass

from torch import nn
from transformers import AutoImageProcessor


@dataclass
class HFSegmentationBundle:
    """Container for a HuggingFace segmentation model and its paired processor.

    Attributes:
        model (nn.Module): The loaded HuggingFace model.
        processor (AutoImageProcessor): The corresponding image processor.

    Example:
        ```python
        bundle = HFSegmentationBundle(model, processor)
        inputs = bundle.processor(image, return_tensors="pt")
        logits = bundle.model(**inputs)
        ```
    """

    model: nn.Module
    processor: AutoImageProcessor
