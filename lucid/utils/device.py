import contextlib

import torch


def pick_device(name: str = "auto") -> torch.device:
    if name != "auto":
        return torch.device(name)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def autocast_dtype(device: torch.device, precision: str) -> torch.dtype | None:
    """The autocast dtype to use, or None for full fp32. Falls back to fp32 if unsupported."""
    if precision == "fp32" or device.type == "cpu":
        return None
    dtype = {"bf16": torch.bfloat16, "fp16": torch.float16}[precision]
    if device.type == "cuda" and dtype == torch.bfloat16 and not torch.cuda.is_bf16_supported():
        return torch.float16
    try:
        with torch.autocast(device.type, dtype=dtype):
            x = torch.ones(2, 2, device=device)
            (x @ x).sum().item()
    except (RuntimeError, TypeError):
        return None
    return dtype


def autocast(device: torch.device, dtype: torch.dtype | None):
    if dtype is None:
        return contextlib.nullcontext()
    return torch.autocast(device.type, dtype=dtype)
