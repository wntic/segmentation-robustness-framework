import os
from typing import Literal, Optional, Union

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
            raise ValueError(f'Encoder {v} is not supported for FCN. Choose either "resnet50" or "resnet101"')
        elif name == "DeepLabV3" and v not in {"resnet50", "resnet101", "mobilenet_v3_large"}:
            raise ValueError(
                f'Encoder {v} is not supported for DeepLabV3. Choose "resnet50", "resnet101", or "mobilenet_v3_large"'
            )
        return v


class AttackConfig(BaseModel):
    name: Literal["FGSM", "PGD"]
    epsilon: conlist(Annotated[float, Field(strict=True, gt=0, le=1)], min_length=1) = None  # type: ignore
    alpha: conlist(Annotated[float, Field(strict=True, gt=0, le=1)], min_length=1) = None  # type: ignore
    steps: Optional[Annotated[int, Field(strict=True, gt=0)]] = None
    targeted: bool = None
    target_label: int = None

    @model_validator(mode="after")
    def validate_attack_parameters(self):
        # FGSM attack params validation
        if self.name == "FGSM":
            if (
                self.alpha is not None
                or self.steps is not None
                or self.targeted is not None
                or self.target_label is not None
            ):
                raise ValueError("FGSM attack got unexpected parameter. Valid parameter is 'epsilon' only")
            if self.epsilon is None:
                raise ValueError("For FGSM, parameter 'epsilon'  should not be None")

        # PGD attack params validation
        if self.name == "PGD":
            if self.epsilon is None or self.alpha is None or self.steps is None or self.targeted is None:
                raise ValueError("For PGD, parameters 'epsilon', 'alpha', 'steps' and 'targeted' should not be None")
            if self.targeted and self.target_label is None:
                raise ValueError("For a targeted attack, 'target_label' must not be None")
            if not self.targeted and self.target_label is not None:
                raise ValueError("For a untargeted attack, 'target_label' must be None")
        return self


class DatasetConfig(BaseModel):
    name: Literal["StanfordBackground", "Cityscapes", "ADE20K", "VOC"]
    root: str
    split: Optional[Literal["train", "val", "test", "train_extra", "trainval"]] = None
    mode: Optional[Literal["fine", "coarse"]] = None
    target_type: Optional[
        Union[
            conlist(Literal["semantic", "instance", "color", "polygon"], min_length=1, max_length=4),  # type: ignore
            Literal["semantic", "instance", "color", "polygon"],
        ]
    ] = None  # type: ignore
    image_shape: Optional[conlist(Annotated[int, Field(strict=True, gt=0)], min_length=2, max_length=2)]  # type: ignore
    max_images: int = None

    @model_validator(mode="after")
    def validate_dataset_parameters(self):
        # VOC dataset params validation
        if self.name == "VOC":
            if self.mode is not None or self.target_type is not None:
                raise ValueError("Invalid configuration: 'mode' and 'target_type' should be None for Pascal VOC")
            if self.split is None:
                raise ValueError("For Pascal VOC, 'split' must be 'train', 'val' or 'trainval'")

        # StanfordBackground dataset params validation
        if self.name == "StanfordBackground":
            if self.split is not None or self.mode is not None or self.target_type is not None:
                raise ValueError(
                    "Invalid configuration: 'split', 'mode', and 'target_type' should be None for StanfordBackground"
                )

        # ADE20K dataset params validation
        if self.name == "ADE20K":
            if self.mode is not None or self.target_type is not None:
                raise ValueError("Invalid configuration: 'mode' and 'target_type' should be None for ADE20K")
            if self.split is None:
                raise ValueError("For ADE20K, 'split' must be either 'train' or 'val'")

        # Cityscapes dataset params validation
        if self.name == "Cityscapes":
            if self.split is None:
                raise ValueError("For Cityscapes, 'split' must be 'train', 'train_extra', 'val' or 'test'")
            if self.mode is None:
                raise ValueError("For Cityscapes, 'mode' must be either 'fine' or 'coarse'")
            if self.target_type is None:
                raise ValueError(
                    "For Cityscapes, 'target_type' must be one of ['semantic', 'instance', 'color', 'polygon'] or multiple as a list"
                )
            if self.split == "train_extra" and self.mode == "fine":
                raise ValueError("'train_extra' split is not available for 'fine' mode. Use 'coarse' mode instead")
        return self

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
            raise ValueError("The 'split' field should be None for StanfordBackground.")
        if name == "Cityscapes" and v not in {"train", "val", "test", "train_extra"}:
            raise ValueError("For Cityscapes, 'split' must be one of 'train', 'val', 'test', or 'train_extra'")
        if name == "ADE20K" and v not in {"train", "val"}:
            raise ValueError("For ADE20K, 'split' must be one of 'train', or 'val'")
        if name == "VOC" and v not in {"train", "val", "trainval"}:
            raise ValueError("For Pascal VOC, 'split' must be one of 'train', 'val', or 'trainval'")
        return v

    @field_validator("target_type")
    @classmethod
    def validate_target_type(cls, v: str, info: ValidationInfo) -> str:
        name = info.data.get("name")
        if name != "Cityscapes" and v is not None:
            raise ValueError(
                f"The 'target_type' field should only be used with Cityscapes and should be None for {name}."
            )
        return v


class Config(BaseModel):
    model: ModelConfig
    attacks: list[AttackConfig]
    dataset: DatasetConfig
