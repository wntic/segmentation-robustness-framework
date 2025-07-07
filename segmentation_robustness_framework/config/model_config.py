from pathlib import Path
from typing import Any, Callable, Literal, Optional, Union

from pydantic import BaseModel, Field


class TorchvisionConfig(BaseModel):
    type: Literal["torchvision"]
    name: str
    num_classes: Optional[int] = 21
    weights: Optional[Any] = None
    device: Optional[str] = "cpu"


class SMPConfig(BaseModel):
    type: Literal["smp"]
    architecture: str
    encoder_name: Optional[str] = "resnet50"
    encoder_weights: Optional[str] = "imagenet"
    classes: Optional[int] = 1
    activation: Optional[str] = None
    weights_path: Optional[Union[str, Path]] = None
    device: Optional[str] = "cpu"


class HuggingFaceConfig(BaseModel):
    type: Literal["huggingface"]
    model_name: str
    num_labels: Optional[int] = None
    trust_remote_code: Optional[bool] = False
    weights_path: Optional[Union[str, Path]] = None
    device: Optional[str] = "cpu"

    model_config = {"protected_namespaces": ()}


class CustomConfig(BaseModel):
    type: Literal["custom"]
    model_class: Callable[..., Any]
    model_args: Optional[list[Any]] = Field(default_factory=list)
    model_kwargs: Optional[dict[str, Any]] = Field(default_factory=dict)
    weights_path: Optional[Union[str, Path]] = None
    device: Optional[str] = "cpu"

    model_config = {"protected_namespaces": ()}


ModelConfig = Union[TorchvisionConfig, SMPConfig, HuggingFaceConfig, CustomConfig]
