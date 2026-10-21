"""A segmentation's boundary as a triangle mesh: an isosurface of one of its maps.

VTK's flying edges (marching cubes, faster) runs on an axis-aligned copy of the map, and the
result is then placed with the map's own origin and orientation, which gives the same answer
in every VTK version. Normals point outward: every map here is higher inside, so outward is
the way the value falls. The triangles are wound to agree with the normals in world space, also
when the orientation is a reflection, because the renderer reads front and back off the winding.
"""

from __future__ import annotations

import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk, numpy_to_vtkIdTypeArray, vtk_to_numpy
from vtkmodules.vtkCommonCore import vtkPoints
from vtkmodules.vtkCommonDataModel import vtkCellArray, vtkImageData, vtkPolyData
from vtkmodules.vtkFiltersCore import vtkFlyingEdges3D

from .grid import axes, voxels


def extract_surface(image: vtkImageData, level: float) -> vtkPolyData:
    """The `level` isosurface of `image`'s scalars, in world coordinates, with outward normals.
    Empty for an image one voxel thick: there is no volume to bound."""
    if min(image.GetDimensions()) < 2:
        return vtkPolyData()
    plain = vtkImageData()
    plain.SetDimensions(image.GetDimensions())
    plain.SetSpacing(image.GetSpacing())
    plain.GetPointData().SetScalars(image.GetPointData().GetScalars())

    edges = vtkFlyingEdges3D()
    edges.SetInputData(plain)
    edges.SetValue(0, level)
    edges.ComputeNormalsOn()
    edges.ComputeGradientsOff()
    edges.ComputeScalarsOff()
    edges.Update()
    contour = edges.GetOutput()

    surface = vtkPolyData()
    if contour.GetNumberOfPoints() == 0:
        return surface

    local = vtk_to_numpy(contour.GetPoints().GetData()).astype(np.float64)
    normals = vtk_to_numpy(contour.GetPointData().GetNormals()).astype(np.float64)
    triangles = vtk_to_numpy(contour.GetPolys().GetConnectivityArray()).reshape(-1, 3)
    # Flying edges' normals follow its gradient's sign; make them point the way the value
    # falls, and wind the triangles to match.
    if _value_rises_along(image, local, normals):
        normals, triangles = -normals, triangles[:, ::-1]
    rotation = axes(image)
    # A direction matrix with a reflection in it (a voxel axis running against a world axis, as
    # NIfTI-born CT often has) maps the normals correctly but mirrors the winding; the renderer
    # takes front from the winding and would light every face from behind.
    if np.linalg.det(rotation) < 0:
        triangles = triangles[:, ::-1]

    points = vtkPoints()
    points.SetData(numpy_to_vtk((np.array(image.GetOrigin()) + local @ rotation.T).astype(np.float32), deep=True))
    surface.SetPoints(points)
    cells = vtkCellArray()
    offsets = np.arange(0, triangles.size + 1, 3, dtype=np.int64)
    cells.SetData(numpy_to_vtkIdTypeArray(offsets, deep=True), numpy_to_vtkIdTypeArray(triangles.ravel().astype(np.int64), deep=True))
    surface.SetPolys(cells)
    world_normals = numpy_to_vtk((normals @ rotation.T).astype(np.float32), deep=True)
    world_normals.SetName("Normals")
    surface.GetPointData().SetNormals(world_normals)
    return surface


def _value_rises_along(image: vtkImageData, points: np.ndarray, normals: np.ndarray) -> bool:
    """Whether the value rises along `normals`, judged at a sample of `points`; both are in the
    image's axis-aligned frame (index times spacing)."""
    values = voxels(image)
    spacing = np.array(image.GetSpacing())
    step = 0.5 * spacing.min()
    votes = 0
    for i in np.linspace(0, len(points) - 1, num=min(len(points), 64)).astype(int):
        ahead = _nearest(values, (points[i] + step * normals[i]) / spacing)
        behind = _nearest(values, (points[i] - step * normals[i]) / spacing)
        votes += int(np.sign(ahead - behind))
    return votes > 0


def _nearest(values: np.ndarray, ijk: np.ndarray) -> float:
    nz, ny, nx = values.shape
    i, j, k = (int(np.clip(round(c), 0, n - 1)) for c, n in zip(ijk, (nx, ny, nz)))
    return float(values[k, j, i])
