"""A vtkImageData's voxel grid in numpy terms: its values, its orientation, and where each
voxel is in world space (LPS mm)."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkFloatArray, vtkUnsignedCharArray
from vtkmodules.vtkCommonDataModel import vtkImageData


def voxels(image: vtkImageData) -> np.ndarray:
    """The image's scalars as a (z, y, x) view; VTK stores x fastest."""
    nx, ny, nz = image.GetDimensions()
    return vtk_to_numpy(image.GetPointData().GetScalars()).reshape(nz, ny, nx)


def axes(image: vtkImageData) -> np.ndarray:
    """The direction matrix: column a is the world direction of voxel axis a."""
    matrix = image.GetDirectionMatrix()
    return np.array([[matrix.GetElement(r, c) for c in range(3)] for r in range(3)])


_ARRAYS = {np.float32: vtkFloatArray, np.uint8: vtkUnsignedCharArray}


def image_from_voxels(values: np.ndarray, spacing: float | tuple[float, float, float], origin=(0.0, 0.0, 0.0)) -> vtkImageData:
    """A vtkImageData on an axis-aligned grid with `values` (z, y, x) as its scalars (copied),
    `spacing` mm (one value, or per axis x, y, z) and `origin` at voxel (0, 0, 0)."""
    from vtkmodules.util.numpy_support import numpy_to_vtk

    values = np.ascontiguousarray(values)
    nz, ny, nx = values.shape
    image = vtkImageData()
    image.SetDimensions(nx, ny, nz)
    image.SetSpacing(*((spacing,) * 3 if np.isscalar(spacing) else tuple(spacing)))
    image.SetOrigin(*origin)
    image.GetPointData().SetScalars(numpy_to_vtk(values.ravel(), deep=True))
    return image


def image_like(reference: vtkImageData, name: str, dtype: type = np.float32) -> tuple[vtkImageData, np.ndarray]:
    """An image on `reference`'s grid (float32 or uint8), and a writable (z, y, x) view of its voxels."""
    image = vtkImageData()
    image.CopyStructure(reference)
    image.SetDirectionMatrix(reference.GetDirectionMatrix())
    values = _ARRAYS[dtype]()
    values.SetName(name)
    values.SetNumberOfTuples(reference.GetNumberOfPoints())
    image.GetPointData().SetScalars(values)
    return image, voxels(image)


class Grid:
    """A volume's voxels in world space: index <-> position."""

    def __init__(self, image: vtkImageData) -> None:
        self.dims = image.GetDimensions()
        self.spacing = np.array(image.GetSpacing())
        self.origin = np.array(image.GetOrigin())
        self.axes = axes(image)
        #: The middle of the grid. Positions are fitted and predicted relative to it, which
        #: keeps squared distances well inside float32's precision.
        self.center = self.origin + self.axes @ ((np.array(self.dims) - 1) * self.spacing / 2)

    def world_to_index(self, points: np.ndarray) -> np.ndarray:
        """Continuous (i, j, k) indices of (n, 3) world points."""
        return ((points - self.origin) @ self.axes) / self.spacing

    def contains(self, ijk: np.ndarray, tolerance: float = 1e-6) -> np.ndarray:
        """Which of (n, 3) indices lie between the outermost voxel centres."""
        return ((ijk >= -tolerance) & (ijk <= np.array(self.dims) - 1 + tolerance)).all(axis=1)

    def slice_positions(self, device: torch.device) -> Callable[[int], torch.Tensor]:
        """k -> the centred world position of every voxel of slice k, (ny * nx, 3) float32."""
        nx, ny, _ = self.dims
        j, i = torch.meshgrid(torch.arange(ny, dtype=torch.float64), torch.arange(nx, dtype=torch.float64), indexing="ij")
        steps = torch.from_numpy(self.axes * self.spacing)  # column a: one step along voxel axis a
        in_plane = i.reshape(-1, 1) * steps[:, 0] + j.reshape(-1, 1) * steps[:, 1] + torch.from_numpy(self.origin - self.center)
        return lambda k: (in_plane + k * steps[:, 2]).to(device=device, dtype=torch.float32)
