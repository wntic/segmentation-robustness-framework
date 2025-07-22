import logging

import torch

logger = logging.getLogger(__name__)


def get_huggingface_output_size(model, input_shape: tuple[int, int, int], device: str = "cpu") -> tuple[int, int]:
    """Get the output spatial size of a HuggingFace segmentation model by running a dummy forward pass.

    Args:
        model: HuggingFace segmentation model (adapter-wrapped).
        input_shape: Input image shape as (C, H, W).
        device: Device to run the dummy pass on.

    Returns:
        tuple[int, int]: Output spatial size (H_out, W_out).

    Raises:
        RuntimeError: If the model fails to produce valid output.
    """
    model = model.to(device)
    model.eval()

    try:
        with torch.no_grad():
            dummy = torch.zeros((1, *input_shape), device=device)  # [1, C, H, W]

            if hasattr(model, "logits"):
                out = model.logits(dummy)
            else:
                out = model(dummy)

            # out shape: [1, num_classes, H_out, W_out]
            if out.dim() != 4:
                raise RuntimeError(f"Expected 4D output tensor, got {out.dim()}D")

            _, _, H_out, W_out = out.shape
            logger.info(f"Detected model output size: {H_out}x{W_out} for input {input_shape[1]}x{input_shape[2]}")
            return H_out, W_out

    except Exception as e:
        logger.error(f"Failed to detect model output size: {e}")
        raise RuntimeError(f"Could not determine model output size: {e}")


def get_model_output_size(model, input_shape: tuple[int, int, int], device: str = "cpu") -> tuple[int, int]:
    """Get the output spatial size of any segmentation model by running a dummy forward pass.

    Args:
        model: Segmentation model (adapter-wrapped).
        input_shape: Input image shape as (C, H, W).
        device: Device to run the dummy pass on.

    Returns:
        tuple[int, int]: Output spatial size (H_out, W_out).

    Raises:
        RuntimeError: If the model fails to produce valid output.
    """
    model = model.to(device)
    model.eval()

    try:
        with torch.no_grad():
            dummy = torch.zeros((1, *input_shape), device=device)  # [1, C, H, W]

            if hasattr(model, "logits"):
                out = model.logits(dummy)
            else:
                out = model(dummy)

            # Handle HuggingFace output objects
            if hasattr(out, "logits"):
                out = out.logits
            elif hasattr(out, "logits") and out.logits is not None:
                out = out.logits

            # out shape: [1, num_classes, H_out, W_out]
            if not hasattr(out, "dim"):
                raise RuntimeError(f"Output does not have 'dim' attribute: {type(out)}")

            if out.dim() != 4:
                raise RuntimeError(f"Expected 4D output tensor, got {out.dim()}D")

            _, _, H_out, W_out = out.shape
            logger.info(f"Detected model output size: {H_out}x{W_out} for input {input_shape[1]}x{input_shape[2]}")
            return H_out, W_out

    except Exception as e:
        logger.error(f"Failed to detect model output size: {e}")
        raise RuntimeError(f"Could not determine model output size: {e}")
