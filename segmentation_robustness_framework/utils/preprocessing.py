def prepare_inputs(sample, maybe_bundle, device="cuda"):
    """
    sample: PIL.Image | np.ndarray | torch.Tensor
    maybe_bundle: HFSegmentationBundle | nn.Module
    """
    if hasattr(maybe_bundle, "processor"):
        proc_inputs = maybe_bundle.processor(sample, return_tensors="pt")
        return {k: v.to(device) for k, v in proc_inputs.items()}
    return {"pixel_values": sample.to(device)}
