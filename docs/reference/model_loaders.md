# Model Loaders

This page provides the API reference documentation for the model loaders available in the Segmentation Robustness Framework.

The framework includes the following model loader classes:

- `TorchvisionModelLoader`: Loads segmentation models from the Torchvision library.
- `SMPModelLoader`: Loads models from the `segmentation_models_pytorch` (SMP) library.
- `HuggingFaceModelLoader`: Loads segmentation models from HuggingFace Transformers.
- `CustomModelLoader`: Allows loading of user-defined or custom models.
- `UniversalModelLoader`: A unified interface that automatically selects the appropriate loader based on the model type.

Each loader implements a consistent interface with the following main methods:

- `load_model(model_config: dict) -> nn.Module`: Instantiates a model according to the provided configuration dictionary.
- `load_weights(model: nn.Module, weights_path: str, weight_type: str = "full") -> nn.Module`: Loads weights into the model. The `weight_type` argument can be `"full"` (entire model) or `"encoder"` (encoder-only weights, if supported).

Refer to the API documentation below for detailed class and method signatures, configuration options, and usage examples for each loader.

::: segmentation_robustness_framework.loaders.models
