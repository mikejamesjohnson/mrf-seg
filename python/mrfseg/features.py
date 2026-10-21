"""Feature vectors per voxel: the grid they are extracted on, and reading them at any point.

The image is resampled to an isotropic grid of about `feature_spacing` mm, normalised to
[0, 1] (or left in HU, for a network trained on HU), and run once through a feature extractor;
the result is kept as float16, C features per feature voxel, beside a float16 copy of the
resampled image in HU (`FeatureVolume.intensity`), which the contrast-sensitive coupling reads.
The extractors themselves are in `extractors.py` (Basic, and the learned network); this module
knows only what one must provide (`FeatureExtractor`).

The feature grid is also the grid the CRF is inferred on (`mrf.py`): isotropic, so every edge
has one length. The clicks' features are read at *any* point, so `FeatureVolume` interpolates
trilinearly. An extractor says how its features are compared (`unit_vectors`): a network's may
be scaled to unit length at every sample, so distances lie in [0, 2] whatever its output scale;
Basic's are standardised once over the image and compared as they are, at much the same
distances.

The feature grid shares the image's orientation and spans exactly its outermost voxel
centres, so a CT index maps to a feature index by a scale per axis. How fine it may be is a
matter of memory (`MAX_FEATURE_BYTES`): an extractor's features cost about 6 bytes per voxel
per channel (float32 while they are made, float16 as kept), so the cap is in bytes and the
default spacing (`default_feature_spacing`) depends on the extractor as well as the image.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Protocol

import numpy as np
import torch
import torch.nn.functional as F
from vtkmodules.vtkCommonDataModel import vtkImageData

from .compute import Progress
from .errors import InputError
from .grid import voxels


class FeatureExtractor(Protocol):
    """What gives each voxel of the feature grid its features.

    Two attributes are optional: `input`, "unit" (the default: the grid min-max scaled to
    [0, 1]) or "hu" (the grid as resampled, in the image's own units); and `native_spacing`,
    the grid spacing in mm the extractor was made for, which the default feature spacing
    starts from when it is set (`default_feature_spacing`).
    """

    channels: int
    """Features per voxel."""
    unit_vectors: bool
    """Whether a voxel's features are compared as a direction (scaled to unit length at every
    sample) or as they are."""

    def __call__(self, volume: np.ndarray, device: torch.device, progress: Progress | None) -> np.ndarray:
        """(C, z, y, x) float32 features of a (z, y, x) volume in [0, 1] (or in HU, `input = "hu"`)."""


def extractor_input(extractor: FeatureExtractor) -> str:
    return getattr(extractor, "input", "unit")


def extractor_native_spacing(extractor: FeatureExtractor) -> float | None:
    return getattr(extractor, "native_spacing", None)


#: Memory the features of one image may take while they are made: about 6 bytes per voxel per
#: channel (`feature_bytes_per_voxel`). 4 GB is 16 channels on 40M voxels.
MAX_FEATURE_BYTES = 4 << 30
#: The default feature grid is no finer than this, mm: coronary lumen is about 2 voxels in
#: radius here (the grid the learned network is trained on), and a finer grid costs the
#: inference time and memory for nothing the maps would show.
MIN_DEFAULT_SPACING = 0.5
#: CT slices resampled at a time: a float32 copy of this many is all the CT costs to resample.
RESAMPLE_SLICES = 16
#: The anti-aliasing Gaussian is cut off this many sigma out, as scipy's is by default.
TRUNCATE = 4.0


def feature_grid_dims(ct_dims: tuple[int, int, int], ct_spacing: tuple[float, float, float], spacing: float) -> tuple[int, ...]:
    """Voxels per axis (x, y, z) of a grid of about `spacing` mm spanning the same voxel centres."""
    return tuple(1 if n == 1 else max(2, round((n - 1) * s / spacing) + 1) for n, s in zip(ct_dims, ct_spacing))


def feature_bytes_per_voxel(channels: int) -> int:
    """What a feature voxel costs while the features are made: the [0, 1] grid, the extractor's
    float32 output, and the float16 copy that is kept."""
    return 4 + 6 * channels


def feature_grid_bytes(ct_dims: tuple[int, int, int], ct_spacing: tuple[float, float, float], spacing: float, channels: int) -> int:
    return math.prod(feature_grid_dims(ct_dims, ct_spacing, spacing)) * feature_bytes_per_voxel(channels)


def default_feature_spacing(ct_dims: tuple[int, int, int], ct_spacing: tuple[float, float, float], channels: int,
                            native: float | None = None) -> float:
    """The image's finest voxel size, but at least `MIN_DEFAULT_SPACING` -- or `native`, the
    spacing an extractor was made for -- doubled until a grid that fine fits `MAX_FEATURE_BYTES`
    with `channels` features per voxel."""
    spacing = float(native) if native else max(float(min(ct_spacing)), MIN_DEFAULT_SPACING)
    while feature_grid_bytes(ct_dims, ct_spacing, spacing, channels) > MAX_FEATURE_BYTES:
        spacing *= 2
    return spacing


def check_feature_grid(ct_dims: tuple[int, int, int], ct_spacing: tuple[float, float, float], spacing: float, channels: int) -> None:
    """An InputError, saying what would fit, if a grid of `spacing` mm is too large for `channels` features."""
    size = feature_grid_bytes(ct_dims, ct_spacing, spacing, channels)
    if size <= MAX_FEATURE_BYTES:
        return
    fits = spacing
    while feature_grid_bytes(ct_dims, ct_spacing, fits, channels) > MAX_FEATURE_BYTES:
        fits *= 2
    raise InputError(
        f"A {spacing:g} mm feature grid would take {size / (1 << 30):.1f} GB for {channels} features per voxel; use at least {fits:g} mm"
    )


@dataclass
class FeatureVolume:
    """Features on one grid that spans the image's outermost voxel centres."""

    values: np.ndarray
    """(C, z, y, x) float16."""
    ct_dims: tuple[int, int, int]
    """The image's dimensions, (x, y, z) like VTK's."""
    unit: bool = True
    """Whether each sample is scaled to unit length (a network's), or used as it is (Basic's)."""
    intensity: np.ndarray | None = None
    """(z, y, x) float16: the image resampled to the feature grid, in its own units (HU), for
    the contrast-sensitive coupling. None for features made without it (tests)."""

    @property
    def channels(self) -> int:
        return self.values.shape[0]

    @property
    def dims(self) -> tuple[int, int, int]:
        """The feature grid's dimensions, (x, y, z)."""
        _, nz, ny, nx = self.values.shape
        return (nx, ny, nz)

    @property
    def shape(self) -> tuple[int, int, int]:
        """The feature grid's shape, (z, y, x) like the arrays'."""
        return tuple(self.values.shape[1:])  # type: ignore[return-value]

    def grid_scale(self) -> np.ndarray:
        """Feature index per CT index, per axis (x, y, z)."""
        return np.array([(f - 1) / (n - 1) if n > 1 else 0.0 for f, n in zip(self.dims, self.ct_dims)])

    def _scale(self) -> np.ndarray:
        return self.grid_scale()

    def ct_to_feature_index(self, ijk: np.ndarray) -> np.ndarray:
        """The nearest feature voxel, (n, 3) int64 as (z, y, x), of continuous CT indices (n, 3) as (i, j, k)."""
        upper = np.array(self.dims) - 1
        f = np.clip(np.rint(np.asarray(ijk, dtype=np.float64) * self.grid_scale()), 0, upper).astype(np.int64)
        return f[:, ::-1]

    def feature_slice(self, k: int, device: torch.device) -> torch.Tensor:
        """The features of feature slice `k` as the kernel compares them: (C, fy, fx) float32 on `device`."""
        plane = torch.from_numpy(self.values[:, k].astype(np.float32)).to(device)
        return _unit(plane, dim=0) if self.unit else plane

    def at_ct_indices(self, ijk: np.ndarray) -> np.ndarray:
        """Feature vectors at continuous CT indices (n, 3) as (i, j, k): (n, C) float32."""
        upper = np.array(self.dims) - 1
        f = np.clip(np.asarray(ijk, dtype=np.float64) * self._scale(), 0, upper)
        low = np.minimum(np.floor(f).astype(np.int64), np.maximum(upper - 1, 0))
        t = f - low
        out = np.zeros((len(f), self.channels), dtype=np.float32)
        for corner in range(8):
            offset = np.array([(corner >> a) & 1 for a in range(3)])
            index = np.minimum(low + offset, upper)
            weight = np.prod(np.where(offset, t, 1 - t), axis=1).astype(np.float32)
            out += weight[:, None] * self.values[:, index[:, 2], index[:, 1], index[:, 0]].T.astype(np.float32)
        return _unit(torch.from_numpy(out)).numpy() if self.unit else out

    def ct_slice(self, k: int, device: torch.device) -> torch.Tensor:
        """Feature vectors at every voxel of CT slice `k`: (C, ny, nx) float32."""
        nx, ny, _ = self.ct_dims
        fz = k * self._scale()[2]
        z0 = min(int(math.floor(fz)), max(self.values.shape[1] - 2, 0))
        t = fz - z0
        plane = torch.from_numpy(self.values[:, z0].astype(np.float32))
        if t > 0 and z0 + 1 < self.values.shape[1]:
            plane = (1 - t) * plane + t * torch.from_numpy(self.values[:, z0 + 1].astype(np.float32))
        plane = plane.to(device)
        if plane.shape[1:] != (ny, nx):
            plane = F.interpolate(plane[None], size=(ny, nx), mode="bilinear", align_corners=True)[0]
        return _unit(plane, dim=0) if self.unit else plane


def _unit(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    # A sum of squares, not `x.norm(dim)`: along a leading dimension that is ~30x slower on the CPU.
    return x * (x * x).sum(dim, keepdim=True).sqrt_().clamp_min_(1e-12).reciprocal_()


def resample(image: vtkImageData, spacing: float) -> np.ndarray:
    """`image` on the feature grid of about `spacing` mm (`feature_grid_dims`): (z, y, x) float32.

    Anti-aliased before it is subsampled, as `skimage.transform.resize` does: along each axis a
    Gaussian of sigma (step - 1) / 2 voxels, with the edge voxels held beyond the image, then
    linear interpolation. Both are linear, so each axis is one small matrix, applied a few CT
    slices at a time: there is no float copy of the whole CT, and a coarse grid is cheaper than
    a fine one. Non-finite voxels take the lowest finite value, so they do not spread.
    """
    ct_dims = image.GetDimensions()
    nz = ct_dims[2]
    fx, fy, fz = feature_grid_dims(ct_dims, image.GetSpacing(), spacing)
    wx, wy, wz = (torch.from_numpy(_axis_operator(n, f)) for n, f in zip(ct_dims, (fx, fy, fz)))
    source = voxels(image)
    fill: float | None = None
    planes = torch.empty((nz, fy, fx))
    for start in range(0, nz, RESAMPLE_SLICES):
        chunk = torch.from_numpy(np.array(source[start : start + RESAMPLE_SLICES], dtype=np.float32))  # a copy
        if not torch.isfinite(chunk).all():
            fill = _lowest_finite(source) if fill is None else fill
            chunk.nan_to_num_(nan=fill, posinf=fill, neginf=fill)
        planes[start : start + RESAMPLE_SLICES] = wy @ (chunk @ wx.T)
    return (wz @ planes.reshape(nz, -1)).reshape(fz, fy, fx).numpy()


def _axis_operator(n: int, f: int) -> np.ndarray:
    """(f, n) float32: from n samples along an axis to f spanning the same first and last,
    blurred first when that is fewer."""
    if n == 1:  # then f is 1 too (`feature_grid_dims`)
        return np.ones((f, 1), dtype=np.float32)
    step = (n - 1) / (f - 1)
    blur = np.eye(n)
    sigma = max(0.0, (step - 1) / 2)
    if sigma > 0:
        radius = int(TRUNCATE * sigma + 0.5)
        offsets = np.arange(-radius, radius + 1)
        taps = np.exp(-0.5 * (offsets / sigma) ** 2)
        blur = np.zeros((n, n))
        rows = np.arange(n)
        for offset, tap in zip(offsets, taps / taps.sum()):
            np.add.at(blur, (rows, np.clip(rows + offset, 0, n - 1)), tap)
    x = np.arange(f) * step
    low = np.minimum(np.floor(x).astype(np.int64), n - 2)
    t = x - low
    interpolate = np.zeros((f, n))
    interpolate[np.arange(f), low] = 1 - t
    interpolate[np.arange(f), low + 1] += t
    return (interpolate @ blur).astype(np.float32)


def _lowest_finite(values: np.ndarray) -> float:
    """The lowest finite voxel, a few slices at a time; 0 if there is none."""
    lowest = math.inf
    for start in range(0, len(values), RESAMPLE_SLICES):
        chunk = values[start : start + RESAMPLE_SLICES]
        finite = np.isfinite(chunk)
        if finite.any():
            lowest = min(lowest, float(chunk[finite].min()))
    return lowest if math.isfinite(lowest) else 0.0


def extract_features(
    image: vtkImageData,
    spacing: float,
    extractor: FeatureExtractor,
    device: torch.device,
    progress: Progress | None = None,
) -> FeatureVolume:
    """Resample `image` to about `spacing` mm, normalise it to [0, 1] (unless the extractor takes
    the image's own units, `input = "hu"`) and run `extractor` on it once. The resampled image
    is kept too, in its own units, as the result's `intensity`."""
    grid = resample(image, spacing)
    intensity = grid.astype(np.float16)
    if extractor_input(extractor) == "unit":
        # Min-max to [0, 1] after clipping, as anatomix's authors do, with the clip at percentiles
        # rather than given: a little metal or air then does not squeeze the anatomy into a sliver.
        # In place, in float32: `np.percentile` of a list is float64, and would promote the grid.
        low, high = (float(v) for v in np.percentile(grid, [0.5, 99.5]))
        grid -= low
        grid *= 1 / max(high - low, 1e-6)
        np.clip(grid, 0, 1, out=grid)

    features = extractor(grid, device, progress)
    if features.shape != (extractor.channels, *grid.shape):
        raise RuntimeError(f"The feature extractor gave {features.shape}, not ({extractor.channels}, {', '.join(map(str, grid.shape))})")
    if not np.isfinite(features).all():
        raise RuntimeError("The feature extractor gave features that are not finite")
    return FeatureVolume(values=features.astype(np.float16), ct_dims=image.GetDimensions(), unit=extractor.unit_vectors, intensity=intensity)
