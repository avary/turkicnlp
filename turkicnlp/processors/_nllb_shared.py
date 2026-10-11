"""Shared NLLB model instances for the translation and embedding processors.

Both processors use the same seq2seq checkpoint; loading it once per
``(path, device)`` keeps one copy in memory when they run together.
"""

from __future__ import annotations

import threading
from typing import Any

_LOCK = threading.Lock()
_MODELS: dict[tuple[str, str], Any] = {}


def select_device(use_gpu: bool) -> str:
    """``cuda`` or ``mps`` when ``use_gpu`` and available, else ``cpu``."""
    if not use_gpu:
        return "cpu"
    import torch

    if torch.cuda.is_available():
        return "cuda"
    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        return "mps"
    return "cpu"


def load_nllb_model(load_from: str, device: str) -> Any:
    """Load (once) the NLLB seq2seq model from ``load_from`` onto ``device``."""
    key = (str(load_from), device)
    with _LOCK:
        model = _MODELS.get(key)
        if model is None:
            from transformers import AutoModelForSeq2SeqLM

            model = AutoModelForSeq2SeqLM.from_pretrained(load_from)
            if device != "cpu":
                model.to(device)
            if hasattr(model, "eval"):
                model.eval()
            _MODELS[key] = model
        return model


def clear() -> None:
    """Drop cached models (frees memory, useful in tests)."""
    with _LOCK:
        _MODELS.clear()


def to_device(tensor: Any, device: str) -> Any:
    """Move a tensor to ``device`` (no-op on CPU)."""
    return tensor if device == "cpu" else tensor.to(device)


def to_cpu(tensor: Any) -> Any:
    return tensor.cpu() if hasattr(tensor, "cpu") else tensor
