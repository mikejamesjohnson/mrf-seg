"""Shared fixtures: small volumes, a feature extractor that needs no model, and MetaImage files."""

from __future__ import annotations

import math
import zlib
from pathlib import Path

import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonDataModel import vtkImageData


def rotation_z(degrees: float) -> np.ndarray:
    a = math.radians(degrees)
    return np.array([[math.cos(a), -math.sin(a), 0], [math.sin(a), math.cos(a), 0], [0, 0, 1]])


def sphere_volume(
    dims=(24, 22, 18), spacing=(2.0, 2.0, 2.5), radius=14.0, axes=None, origin=None
) -> tuple[vtkImageData, np.ndarray]:
    """A bright sphere at the centre of a (possibly rotated) grid -- at the world origin unless
    `origin` moves the grid -- and the world position of every voxel, (z, y, x, 3)."""
    axes = np.eye(3) if axes is None else np.asarray(axes)
    nx, ny, nz = dims
    spacing = np.asarray(spacing, dtype=float)
    centre_offset = axes @ ((np.array(dims) - 1) * spacing / 2)
    origin = -centre_offset if origin is None else np.asarray(origin)
    k, j, i = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
    world = origin + (np.stack([i, j, k], -1) * spacing) @ axes.T
    inside = np.linalg.norm(world - (origin + centre_offset), axis=-1) <= radius

    volume = vtkImageData()
    volume.SetDimensions(*dims)
    volume.SetSpacing(*spacing)
    volume.SetOrigin(*origin)
    volume.SetDirectionMatrix(*axes.ravel())
    volume.GetPointData().SetScalars(numpy_to_vtk(np.where(inside, 300, 0).astype(np.int16).ravel(), deep=True))
    return volume, world


class StubExtractor:
    """A `FeatureExtractor` that needs no model: three features from intensity alone, enough to
    tell a bright sphere from its background. Records what it was given, for tests of when and
    on what the features are extracted."""

    channels = 3
    unit_vectors = True

    def __init__(self) -> None:
        self.shapes: list[tuple[int, ...]] = []
        self.ranges: list[tuple[float, float]] = []
        #: Raised by the next call instead of extracting, once: a failing extractor.
        self.fail_with: Exception | None = None
        #: Seconds the next call dawdles, reporting progress every 50 ms, once: a slow extractor.
        self.slow_for: float = 0.0

    def __call__(self, volume: np.ndarray, device, progress=None) -> np.ndarray:
        if self.fail_with is not None:
            error, self.fail_with = self.fail_with, None
            raise error
        if self.slow_for > 0:
            import time

            seconds, self.slow_for = self.slow_for, 0.0
            started = time.perf_counter()
            while time.perf_counter() - started < seconds:
                time.sleep(0.05)
                if progress:
                    progress(min((time.perf_counter() - started) / seconds, 0.99))
        self.shapes.append(volume.shape)
        self.ranges.append((float(volume.min()), float(volume.max())))
        if progress:
            progress(1.0)
        return np.stack([volume, 1 - volume, np.full_like(volume, 0.5)]).astype(np.float32)


#: One to share where the calls do not matter.
stub_extractor = StubExtractor()


def sphere_points(axes=None) -> tuple[np.ndarray, np.ndarray]:
    """Labelled points relative to the sphere's centre: inside it, and outside it but inside
    `sphere_volume`'s grid -- given in the grid's own frame, then turned by `axes`."""
    inside = [[0, 0, 0], [5, -3, 2], [-4, 4, -3], [2, 6, 4]]
    outside = [[19, 15, 0], [-19, 14, 6], [17, -15, -10], [-18, -14, 12], [0, 0, 19], [20, 0, -15]]
    points = np.array(inside + outside, dtype=float)
    if axes is not None:
        points = points @ np.asarray(axes).T
    return points, np.array([1] * len(inside) + [-1] * len(outside))


def write_metaimage(
    path: Path,
    voxels: np.ndarray,
    element_type: str,
    *,
    dims,
    spacing,
    origin,
    axes,
    data_file: str | None = None,
    compressed: bool = False,
    big_endian: bool = False,
    extra: str = "",
) -> None:
    """MetaImage the way ITK writes it: `TransformMatrix` row n is voxel axis n."""
    raw = voxels.astype(voxels.dtype.newbyteorder(">" if big_endian else "<")).tobytes()
    if compressed:
        raw = zlib.compress(raw)
    header = "\n".join(
        [
            "ObjectType = Image",
            "NDims = 3",
            "BinaryData = True",
            f"BinaryDataByteOrderMSB = {big_endian}",
            f"CompressedData = {compressed}",
            "TransformMatrix = " + " ".join(f"{v:.17g}" for v in np.asarray(axes).T.ravel()),
            "Offset = " + " ".join(map(str, origin)),
            "ElementSpacing = " + " ".join(map(str, spacing)),
            "DimSize = " + " ".join(map(str, dims)),
            *([extra] if extra else []),
            f"ElementType = {element_type}",
            f"ElementDataFile = {data_file or 'LOCAL'}",
        ]
    )
    if data_file:
        path.write_text(header + "\n", encoding="latin-1")
        (path.parent / data_file).write_bytes(raw)
    else:
        path.write_bytes(header.encode("latin-1") + b"\n" + raw)
