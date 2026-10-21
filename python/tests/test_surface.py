"""extract_surface: where the mesh is, which way it faces, and an empty level."""

import unittest

import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk, vtk_to_numpy
from vtkmodules.vtkCommonDataModel import vtkImageData

from mrfseg import extract_surface
from tests.helpers import rotation_z


def sphere_field(radius=12.0, n=40, spacing=(1.0, 1.2, 0.8), axes=None, origin=(5.0, -3.0, 2.0)):
    """radius - distance from the grid's centre: positive inside, like the latent mean."""
    axes = rotation_z(30) if axes is None else axes
    image = vtkImageData()
    image.SetDimensions(n, n, n)
    image.SetSpacing(*spacing)
    image.SetOrigin(*origin)
    image.SetDirectionMatrix(*np.asarray(axes).ravel())
    k, j, i = np.meshgrid(*[np.arange(n)] * 3, indexing="ij")
    local = np.stack([i, j, k], -1) * np.array(spacing)
    centre_local = (n - 1) / 2 * np.array(spacing)
    field = radius - np.linalg.norm(local - centre_local, axis=-1)
    image.GetPointData().SetScalars(numpy_to_vtk(field.ravel().astype(np.float32), deep=True))
    return image, np.array(origin) + np.asarray(axes) @ centre_local


class ExtractSurfaceTest(unittest.TestCase):
    def test_sphere_on_an_oblique_anisotropic_grid(self):
        image, centre = sphere_field()
        surface = extract_surface(image, 0.0)
        points = vtk_to_numpy(surface.GetPoints().GetData())
        normals = vtk_to_numpy(surface.GetPointData().GetNormals())
        radius = np.linalg.norm(points - centre, axis=1)
        self.assertGreater(surface.GetNumberOfPolys(), 1000)
        self.assertLess(abs(radius.mean() - 12.0), 0.05)
        self.assertLess(radius.std(), 0.05)
        outward = np.einsum("ij,ij->i", normals, (points - centre) / radius[:, None])
        self.assertGreater(outward.min(), 0.99)

    def test_triangles_wind_with_the_normals(self):
        for axes in (rotation_z(30), np.diag([1.0, -1.0, 1.0]), rotation_z(30) @ np.diag([1.0, 1.0, -1.0])):
            # The second and third are reflections (a voxel axis against a world axis, as a
            # NIfTI-born CT often has): mapping the mesh through one mirrors its winding, and the
            # renderer lights faces from behind unless it is wound back to agree with the normals.
            with self.subTest(det=round(float(np.linalg.det(axes)))):
                image, centre = sphere_field(axes=axes)
                surface = extract_surface(image, 0.0)
                points = vtk_to_numpy(surface.GetPoints().GetData())
                normals = vtk_to_numpy(surface.GetPointData().GetNormals())
                triangles = vtk_to_numpy(surface.GetPolys().GetConnectivityArray()).reshape(-1, 3)
                a, b, c = points[triangles[:, 0]], points[triangles[:, 1]], points[triangles[:, 2]]
                face = np.cross(b - a, c - a)
                outward = np.einsum("ij,ij->i", face, (a + b + c) / 3 - centre)
                self.assertGreater((outward > 0).mean(), 0.99)                   # wound counter-clockwise seen from outside
                self.assertGreater(np.einsum("ij,ij->i", normals, points - centre).min(), 0.0)   # and the normals outward

    def test_a_level_nowhere_in_the_field_gives_an_empty_mesh(self):
        image, _ = sphere_field()
        self.assertEqual(extract_surface(image, 1000.0).GetNumberOfPoints(), 0)


if __name__ == "__main__":
    unittest.main()
