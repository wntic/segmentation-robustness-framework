from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import torch.nn as nn


class BaseModelLoader(ABC):
    """Abstract base class for all model loaders.

    Defines the interface for loading models and weights.

    Example:
        ```python
        class MyLoader(BaseModelLoader):
            def load_model(self, model_config: dict[str, Any]) -> nn.Module: ...
            def load_weights(
                self,
                model: nn.Module,
                weights_path: str | Path,
                weight_type: str = "full",
            ) -> nn.Module: ...
        ```
    """

    @abstractmethod
    def load_model(self, model_config: dict[str, Any]) -> Any:
        """Load a model based on the provided configuration.

        Args:
            model_config (dict[str, Any]): Dictionary with model-specific parameters.

        Returns:
            `nn.Module` | `HFSegmentationBundle`: Instantiated model.
        """
        pass

    @abstractmethod
    def load_weights(self, model: nn.Module, weights_path: str | Path, weight_type: str = "full") -> nn.Module:
        """Load weights into a model.

        Args:
            model (nn.Module): Model instance.
            weights_path (str | Path): Path to weights file.
            weight_type (str): `'full'` for entire model, `'encoder'` for backbone only.

        Returns:
            `nn.Module`: Model with loaded weights.
        """
        pass
