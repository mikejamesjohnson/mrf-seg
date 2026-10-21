"""Where the heavy steps run, and how they report progress."""

from __future__ import annotations

from collections.abc import Callable

import torch

#: Reports how far a long step has got, as a fraction in [0, 1].
Progress = Callable[[float], None]


def resolve_device(name: str) -> torch.device:
    """`"auto"` is the GPU when there is one -- CUDA, else the Apple GPU -- else the CPU."""
    if name == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        return torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    return torch.device(name)
