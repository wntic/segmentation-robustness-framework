import os
from typing import List, Literal, Optional

from pydantic import BaseModel, Field, ValidationInfo, conlist, field_validator, model_validator
from typing_extensions import Annotated


class ModelConfig(BaseModel):
    name: Literal["FCN", "DeepLabV3"]
    encoder: Literal["resnet50", "resnet101", "mobilenet_v3_large"]
    weights: str
    num_classes: int
    device: Literal["cpu", "cuda"]

    @field_validator("encoder")
    @classmethod
    def validate_encoder(cls, v: str, info: ValidationInfo) -> str:
        name = info.data.get("name")
        if name == "FCN" and v not in {"resnet50", "resnet101"}:
            raise ValueError(f"Encoder {v} is not supported for FCN. Choose either 'resnet50' or 'resnet101'")
        elif name == "DeepLabV3" and v not in {"resnet50", "resnet101", "mobilenet_v3_large"}:
            raise ValueError(
                f"Encoder {v} is not supported for DeepLabV3. Choose 'resnet50', 'resnet101', or 'mobilenet_v3_large'"
            )
        return v


class AttackConfig(BaseModel):
    name: Literal["FGSM", "PGD"]
    epsilon: conlist(Annotated[float, Field(strict=True, gt=0, le=1)], min_length=1)  # type: ignore
    alpha: conlist(Annotated[float, Field(strict=True, gt=0, le=1)], min_length=1) = None  # type: ignore
    steps: Optional[Annotated[int, Field(strict=True, gt=0)]] = None
    targeted: bool

    @model_validator(mode="after")
    def validate_attack_parameters(self):
        if self.name == "FGSM":
            if self.alpha is not None:
                raise ValueError(
                    "FGSM attack does not support 'alpha' parameter. Valid parameters are 'epsilon' and 'targeted'"
                )
            if self.steps is not None:
                raise ValueError(
                    "FGSM attack does not support 'steps' parameter. Valid parameters are 'epsilon' and 'targeted'"
                )
        if self.name == "PGD":
            if self.alpha is None:
                raise ValueError("PGD ​​attack requires 'alpha' parameter")
            if self.steps is None:
                raise ValueError("PGD ​​attack requires 'steps' parameter")
        return self


class DatasetConfig(BaseModel):
    name: Literal["StanfordBackground", "Cityscapes", "ADE20K", "PascalVOC"]
    root: str
    split: Optional[Literal["train", "val", "test", "train_extra", "trainval"]] = None
    mode: Optional[Literal["fine", "coarse"]] = None

    @field_validator("root")
    @classmethod
    def validate_root(cls, v: str) -> str:
        if not os.path.exists(v):
            raise ValueError(f"The specified root directory does not exist: {v}")
        return v

    @field_validator("split")
    @classmethod
    def validate_split(cls, v: str, info: ValidationInfo) -> str:
        name = info.data.get("name")
        if name == "StanfordBackground" and v is not None:
            raise ValueError("The 'split' field should be None for StanfordBackground")
        if name == "Cityscapes" and v not in {"train", "val", "test", "train_extra"}:
            raise ValueError("For Cityscapes, 'split' must be one of 'train', 'val', 'test', or 'train_extra'")
        if name == "ADE20K" and v not in {"train", "val"}:
            raise ValueError("For ADE20K, 'split' must be one of 'train', or 'val'")
        if name == "PascalVOC" and v not in {"train", "val", "trainval"}:
            raise ValueError("For Pascal VOC, 'split' must be one of 'train', 'val', or 'trainval'")
        return v

    @field_validator("mode")
    @classmethod
    def validate_mode(cls, v: str, info: ValidationInfo) -> str:
        name = info.data.get("name")
        if name != "Cityscapes" and v is not None:
            raise ValueError(f"The 'mode' field should only be used with Cityscapes and should be None for {name}")
        if name == "Cityscapes" and v not in {"fine", "coarse"}:
            raise ValueError("For Cityscapes, 'mode' must be either 'fine' or 'coarse'")
        return v


class OutputConfig(BaseModel):
    save_dir: str
    save_images: bool
    save_log: bool

    @field_validator("save_dir")
    @classmethod
    def validate_save_dir(cls, v: str) -> str:
        if not os.path.exists(v):
            raise ValueError(f"The specified save directory does not exist: {v}")
        return v


class Config(BaseModel):
    model: ModelConfig
    attacks: List[AttackConfig]
    dataset: DatasetConfig
    output: OutputConfig
