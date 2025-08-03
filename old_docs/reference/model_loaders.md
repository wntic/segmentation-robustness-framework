# Model Loaders

This page provides the API reference documentation for the model loaders available in the Segmentation Robustness Framework.

The framework includes the following model loader classes:

- `TorchvisionModelLoader`: Loads segmentation models from the Torchvision library.
- `SMPModelLoader`: Loads models from the `segmentation_models_pytorch` (SMP) library.
- `HuggingFaceModelLoader`: Loads segmentation models from HuggingFace Transformers.
- `CustomModelLoader`: Allows loading of user-defined or custom models.
- `UniversalModelLoader`: A unified interface that automatically selects the appropriate loader based on the model type.

Each loader implements a consistent interface with the following main methods:

- `load_model(model_config: dict, adapter_cls: Optional[type] = None, ...) -> nn.Module`: Instantiates a model according to the provided configuration dictionary. For the universal loader, you can now pass an adapter class directly via `adapter_cls` (takes precedence over registry).
- `load_weights(model: nn.Module, weights_path: str, weight_type: str = "full") -> nn.Module`: Loads weights into the model. The `weight_type` argument can be `"full"` (entire model) or `"encoder"` (encoder-only weights, if supported).

## UniversalModelLoader: Custom Adapters and Direct Adapter Passing

- You can register multiple custom adapters with names like `custom_myadapter` and use them by specifying `model_type="custom_myadapter"`.
- You can also pass an adapter class directly to `load_model` via the `adapter_cls` argument, e.g.:

```python
model = loader.load_model("custom", config, adapter_cls=MyCustomAdapter)
```

## TorchvisionModelLoader: Weights Handling

- The `weights` argument now accepts both `"default"` and `"DEFAULT"` (case-insensitive) to use the default pretrained weights.

## SMPModelLoader: Encoder Weights

- You can load only encoder weights by passing `weight_type="encoder"` to `load_weights`.

## HuggingFaceModelLoader: Expanded Support

- Supports tasks: `semantic_segmentation`, `instance_segmentation`, `panoptic_segmentation`, `image_segmentation`.
- You can specify a custom model class in the config with `model_cls` (string or class).
- You can load only encoder weights by passing `weight_type="encoder"` to `load_weights`.

Refer to the API documentation below for detailed class and method signatures, configuration options, and usage examples for each loader.

::: segmentation_robustness_framework.loaders.models
    options:
        show_signature_annotations: true
