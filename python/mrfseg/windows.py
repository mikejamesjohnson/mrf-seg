"""Running a network over a volume in overlapping windows, blended with Gaussian weights.

Shared by the extractors that are networks (`anatomix`, `learned`). A window of `window` voxels
a side starts every `step` voxels along each axis (`tile_starts`), the last one flush with the
end; each window's output is weighed by a Gaussian about its middle (`gaussian_weights`), so a
voxel's features come mostly from the window it sits deepest in, and the sum is divided by the
summed weights. A side shorter than a window is padded (`pad_to`), mirrored and split between
both ends, as the networks' own convolutions pad.
"""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def tile_starts(n: int, window: int, step: int) -> list[int]:
    """Where windows of `window` start along an axis of `n` voxels, every `step` voxels and one
    flush with the end; one window at 0 when the axis fits in one."""
    if n <= window:
        return [0]
    return [*range(0, n - window, step), n - window]


def pad_to(tensor: torch.Tensor, pad: list[int]) -> tuple[torch.Tensor, list[int]]:
    """`tensor` (1, 1, z, y, x) grown by `pad` voxels per axis, and how many went before it.

    Split between both ends and mirrored, as the networks' own convolutions pad. All at one end,
    and replicated, put a slab of one repeated slice next to the image, and the features within
    reach of it changed enough to take the Dice of a test sphere from 0.98 to 0.89. A side too
    thin to mirror is replicated instead.
    """
    if not any(pad):
        return tensor, [0, 0, 0]
    before = [p // 2 for p in pad]
    after = [p - b for p, b in zip(pad, before)]
    mirror = all(max(b, a) < n for b, a, n in zip(before, after, tensor.shape[2:]))
    widths = (before[2], after[2], before[1], after[1], before[0], after[0])
    return F.pad(tensor, widths, mode="reflect" if mirror else "replicate"), before


def gaussian_weights(size: tuple[int, ...], sigma: float) -> np.ndarray:
    """A weight per voxel of a window: a Gaussian about its middle, `sigma` of its size across,
    as MONAI's `sliding_window_inference` weighs them (`mode="gaussian"`)."""
    axes = []
    for n in size:
        x = np.arange(n, dtype=np.float32) - (n - 1) / 2
        axes.append(np.exp(-0.5 * (x / max(sigma * n, 1e-6)) ** 2))
    return axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]
