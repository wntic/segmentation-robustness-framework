from dataclasses import dataclass

from torch import nn
from transformers import AutoImageProcessor


@dataclass
class HFSegmentationBundle:
    """
    Container for a HuggingFace segmentation model and its paired processor.

    Attributes:
        model (nn.Module): The loaded HuggingFace model.
        processor (AutoImageProcessor): The corresponding image processor.

    Example usage:
        bundle = HFSegmentationBundle(model, processor)
        logits = bundle.model(**bundle.processor(image, return_tensors="pt"))
    """

    model: nn.Module
    processor: AutoImageProcessor
