"""Basic, the default feature extractor (`extractors.py`): the image's local brightness at
several scales, and optionally its edges and its shape there.

These are the features scikit-image's `multiscale_basic_features` gives (and ilastik's and
Weka's trainable segmentation): for each scale sigma,

- **intensity**: the image blurred with a Gaussian of sigma;
- **edges**: the magnitude of that blur's gradient (Sobel);
- **shape**: the three eigenvalues of its Hessian (finite differences of the blur), which tell a
  tube from a blob, a sheet or a flat region.

They are computed here with scipy, a scale at a time in float32, rather than all at once in
float64. The scales are sigma = 1, 2, 4, ... voxels of the feature grid: the grid spacing and
doublings of it.

Unlike anatomix's, these features are compared as they are, not as unit vectors: what they
measure is in their size (400 HU of lumen against -90 of fat). Each is centred and scaled to unit
standard deviation over the image, then all of them by 1/sqrt(C), so two voxels are about as far
apart as two unit vectors are and a feature length scale means much the same for both
extractors.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
from scipy.ndimage import gaussian_filter, sobel

from .compute import Progress

#: The most scales: by the ninth, sigma is 256 voxels, wider than most images.
MAX_SCALES = 8
#: Slices whose Hessians are diagonalised at a time: a (slices, y, x, 3, 3) array.
SLAB = 16


@dataclass(frozen=True)
class Basic:
    """A `FeatureExtractor`: `scales` blurs, each with its edges and shape if asked for."""

    scales: int = 4
    edges: bool = False
    shape: bool = False

    #: Its features are compared as they are (`FeatureVolume.unit`).
    unit_vectors = False

    @property
    def channels(self) -> int:
        return self.scales * (1 + int(self.edges) + 3 * int(self.shape))

    def __call__(self, volume: np.ndarray, device: torch.device, progress: Progress | None = None) -> np.ndarray:
        """(C, z, y, x) float32 features of a (z, y, x) volume; `device` is not needed."""
        volume = np.asarray(volume, dtype=np.float32)
        out = np.empty((self.channels, *volume.shape), dtype=np.float32)
        c = 0
        for k in range(self.scales):
            blurred = gaussian_filter(volume, 2.0**k, mode="nearest")
            out[c] = blurred
            c += 1
            if self.edges:
                out[c] = np.sqrt(sum(sobel(blurred, axis=axis) ** 2 for axis in range(3)))
                c += 1
            if self.shape:
                out[c : c + 3] = _hessian_eigenvalues(blurred)
                c += 3
            if progress:
                progress((k + 1) / self.scales)
        _standardise(out)
        return out


def _derivative(values: np.ndarray, axis: int) -> np.ndarray:
    """Central differences along `axis`, as `np.gradient` takes them; zero along an axis of one voxel."""
    return np.gradient(values, axis=axis) if values.shape[axis] > 1 else np.zeros_like(values)


def _hessian_eigenvalues(blurred: np.ndarray) -> np.ndarray:
    """(3, z, y, x): the eigenvalues of each voxel's Hessian, largest first, as scikit-image's
    `hessian_matrix_eigvals` gives them."""
    first = [_derivative(blurred, axis) for axis in range(3)]
    second = {(i, j): _derivative(first[i], j) for i in range(3) for j in range(i, 3)}
    del first
    out = np.empty((3, *blurred.shape), dtype=np.float32)
    for start in range(0, blurred.shape[0], SLAB):
        slab = slice(start, start + SLAB)
        matrices = np.empty((*blurred[slab].shape, 3, 3), dtype=np.float32)
        for (i, j), values in second.items():
            matrices[..., i, j] = matrices[..., j, i] = values[slab]
        out[:, slab] = np.moveaxis(np.linalg.eigvalsh(matrices)[..., ::-1], -1, 0)
    return out


def _standardise(features: np.ndarray) -> None:
    """In place: each feature to zero mean and unit standard deviation over the volume (a
    constant one to zero), then all by 1/sqrt(C)."""
    scale = 1 / math.sqrt(len(features))
    for channel in features:
        mean, std = float(channel.mean(dtype=np.float64)), float(channel.std(dtype=np.float64))
        channel -= mean
        channel *= scale / std if std > 0 else 0.0
