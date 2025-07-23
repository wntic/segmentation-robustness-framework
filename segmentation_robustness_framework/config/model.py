from pathlib import Path
from typing import Any, Callable, Literal, Optional, Union

from pydantic import BaseModel, Field


class TorchvisionConfig(BaseModel):
    """Configuration for Torchvision segmentation models.

    Attributes:
        type (Literal["torchvision"]): Model type identifier.
        name (str): Model name (e.g., 'deeplabv3_resnet50').
        num_classes (int, optional): Number of output classes. Defaults to 21.
        weights (Any, optional): Pretrained weights identifier or enum. Defaults to None.
        device (str, optional): Device to use. Defaults to 'cpu'.
    """

    type: Literal["torchvision"]
    name: str
    num_classes: Optional[int] = 21
    weights: Optional[Any] = None
    device: Optional[str] = "cpu"


class SMPConfig(BaseModel):
    """Configuration for segmentation_models_pytorch (SMP) models.

    Attributes:
        type (Literal["smp"]): Model type identifier.
        architecture (str): Model architecture (e.g., 'unet').
        encoder_name (str, optional): Encoder name. Defaults to 'resnet50'.
        encoder_weights (str, optional): Encoder weights. Defaults to 'imagenet'.
        classes (int, optional): Number of output classes. Defaults to 1.
        activation (str, optional): Activation function. Defaults to None.
        weights_path (str | Path, optional): Path to weights file. Defaults to None.
        checkpoint (str, optional): Pretrained checkpoint identifier. Defaults to None.
        device (str, optional): Device to use. Defaults to 'cpu'.
    """

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
    """Configuration for HuggingFace segmentation models.

    Attributes:
        type (Literal["huggingface"]): Model type identifier.
        model_name (str): HuggingFace model name or path.
        num_labels (int, optional): Number of output classes. Defaults to None.
        trust_remote_code (bool, optional): Allow remote code. Defaults to False.
        weights_path (str | Path, optional): Path to weights file. Defaults to None.
        device (str, optional): Device to use. Defaults to 'cpu'.
        task (str, optional): Model task. Defaults to 'semantic_segmentation'.
        return_processor (bool, optional): Return processor with model. Defaults to True.
        config_overrides (dict, optional): Config attribute overrides. Defaults to empty dict.
        processor_overrides (dict, optional): Processor attribute overrides. Defaults to empty dict.
    """

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

    model_config = {"protected_namespaces": ()}


class CustomConfig(BaseModel):
    """Configuration for custom user-defined segmentation models.

    Attributes:
        type (Literal["custom"]): Model type identifier.
        model_class (str | Callable[..., Any]): Model class or factory function.
        model_args (list, optional): Positional arguments for model initialization. Defaults to empty list.
        model_kwargs (dict, optional): Keyword arguments for model initialization. Defaults to empty dict.
        weights_path (str | Path, optional): Path to weights file. Defaults to None.
        device (str, optional): Device to use. Defaults to 'cpu'.
    """

    type: Literal["custom"]
    model_class: Union[str, Callable[..., Any]]
    model_args: Optional[list[Any]] = Field(default_factory=list)
    model_kwargs: Optional[dict[str, Any]] = Field(default_factory=dict)
    weights_path: Optional[Union[str, Path]] = None
    device: Optional[str] = "cpu"

    model_config = {"protected_namespaces": ()}


ModelConfig = Union[TorchvisionConfig, SMPConfig, HuggingFaceConfig, CustomConfig]
