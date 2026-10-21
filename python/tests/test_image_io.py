"""read_image: MetaImage in every layout the editor's loader reads, and .vti, into LPS mm."""

import tempfile
import unittest
from pathlib import Path

import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkIOXML import vtkXMLImageDataWriter

from mrfseg import ImageReadError, read_image
from mrfseg.grid import axes, voxels
from tests.helpers import write_metaimage

DIMS = (6, 5, 4)
SPACING = (0.8, 0.9, 1.5)
ORIGIN = (-12.5, 30.25, -7.0)


def rotation(degrees_z: float, degrees_x: float) -> np.ndarray:
    """A rotation; its columns are the directions of the voxel axes."""
    cz, sz = np.cos(np.radians(degrees_z)), np.sin(np.radians(degrees_z))
    cx, sx = np.cos(np.radians(degrees_x)), np.sin(np.radians(degrees_x))
    return np.array([[1, 0, 0], [0, cx, -sx], [0, sx, cx]]) @ np.array([[cz, -sz, 0], [sz, cz, 0], [0, 0, 1]])


AXES = rotation(30, 15)  # oblique, so a transposed matrix cannot go unnoticed


def write(path: Path, values: np.ndarray, element_type: str, dims=DIMS, **options) -> None:
    write_metaimage(path, values, element_type, dims=dims, spacing=SPACING, origin=ORIGIN, axes=AXES, **options)


class ReadImageTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        nx, ny, nz = DIMS
        self.values = np.arange(nx * ny * nz, dtype=np.int16).reshape(nz, ny, nx) - 50

    def tearDown(self):
        self.tmp.cleanup()

    def assert_geometry(self, image: vtkImageData):
        self.assertEqual(image.GetDimensions(), DIMS)
        np.testing.assert_allclose(image.GetSpacing(), SPACING)
        np.testing.assert_allclose(image.GetOrigin(), ORIGIN)
        np.testing.assert_allclose(axes(image), AXES, atol=1e-12)

    def test_mha_with_compressed_voxels(self):
        write(self.dir / "a.mha", self.values, "MET_SHORT", compressed=True)
        image = read_image(self.dir / "a.mha")
        self.assert_geometry(image)
        np.testing.assert_array_equal(voxels(image), self.values)

    def test_mhd_with_big_endian_raw_file(self):
        values = self.values.astype(np.uint16) + 1000
        write(self.dir / "b.mhd", values, "MET_USHORT", data_file="b.raw", big_endian=True)
        image = read_image(self.dir / "b.mhd")
        self.assert_geometry(image)
        np.testing.assert_array_equal(voxels(image), values)

    def test_mhd_with_zraw_file(self):
        values = self.values.astype(np.float32) / 7
        write(self.dir / "c.mhd", values, "MET_FLOAT", data_file="c.zraw", compressed=True)
        np.testing.assert_array_equal(voxels(read_image(self.dir / "c.mhd")), values)

    def test_header_size_minus_one_reads_from_the_end(self):
        write(self.dir / "d.mhd", self.values, "MET_SHORT", data_file="d.raw", extra="HeaderSize = -1")
        raw = (self.dir / "d.raw").read_bytes()
        (self.dir / "d.raw").write_bytes(b"junk header" + raw)
        np.testing.assert_array_equal(voxels(read_image(self.dir / "d.mhd")), self.values)

    def test_vti_matches_the_same_image_as_metaimage(self):
        image = vtkImageData()
        image.SetDimensions(*DIMS)
        image.SetSpacing(*SPACING)
        image.SetOrigin(*ORIGIN)
        image.SetDirectionMatrix(*AXES.ravel())
        image.GetPointData().SetScalars(numpy_to_vtk(self.values.ravel(), deep=True))
        writer = vtkXMLImageDataWriter()
        writer.SetFileName(str(self.dir / "e.vti"))
        writer.SetInputData(image)
        self.assertEqual(writer.Write(), 1)
        write(self.dir / "e.mha", self.values, "MET_SHORT")

        for read in (read_image(self.dir / "e.vti"), read_image(self.dir / "e.mha")):
            self.assert_geometry(read)
            np.testing.assert_array_equal(voxels(read), self.values)

    def test_vti_extent_not_starting_at_zero_is_folded_into_the_origin(self):
        image = vtkImageData()
        image.SetExtent(10, 15, 3, 7, 2, 5)  # DIMS, offset
        image.SetSpacing(*SPACING)
        image.SetOrigin(*ORIGIN)
        image.SetDirectionMatrix(*AXES.ravel())
        image.GetPointData().SetScalars(numpy_to_vtk(self.values.ravel(), deep=True))
        writer = vtkXMLImageDataWriter()
        writer.SetFileName(str(self.dir / "g.vti"))
        writer.SetInputData(image)
        self.assertEqual(writer.Write(), 1)
        first = [0.0, 0.0, 0.0]
        image.TransformIndexToPhysicalPoint(10, 3, 2, first)  # where VTK puts the first voxel

        read = read_image(self.dir / "g.vti")
        self.assertEqual(read.GetExtent(), (0, 5, 0, 4, 0, 3))
        np.testing.assert_allclose(read.GetOrigin(), first, atol=1e-9)
        np.testing.assert_allclose(axes(read), AXES, atol=1e-12)
        np.testing.assert_array_equal(voxels(read), self.values)

    def test_a_sheared_direction_is_refused(self):
        sheared = AXES.copy()
        sheared[:, 1] += 0.2 * sheared[:, 0]
        write_metaimage(self.dir / "s.mha", self.values, "MET_SHORT", dims=DIMS, spacing=SPACING, origin=ORIGIN, axes=sheared)
        with self.assertRaisesRegex(ImageReadError, "not a rotation"):
            read_image(self.dir / "s.mha")

    def test_two_dimensional_metaimage_is_one_slice_thick(self):
        path = self.dir / "f.mha"
        values = np.arange(12, dtype=np.uint8).reshape(3, 4)
        path.write_bytes(
            b"NDims = 2\nDimSize = 4 3\nElementSpacing = 0.5 0.25\nElementType = MET_UCHAR\n"
            b"ElementDataFile = LOCAL\n" + values.tobytes()
        )
        image = read_image(path)
        self.assertEqual(image.GetDimensions(), (4, 3, 1))
        np.testing.assert_allclose(image.GetSpacing(), (0.5, 0.25, 1))
        np.testing.assert_array_equal(voxels(image)[0], values)

    def test_errors(self):
        (self.dir / "x.nrrd").write_bytes(b"NRRD0004")
        write(self.dir / "rgb.mha", self.values, "MET_SHORT", extra="ElementNumberOfChannels = 3")
        write(self.dir / "short.mha", self.values[:2], "MET_SHORT")
        (self.dir / "nothing.mha").write_bytes(b"ObjectType = Image\nNDims = 3\n")
        (self.dir / "ndims.mha").write_bytes(b"NDims = 3.0\nDimSize = 2 2 2\nElementType = MET_UCHAR\nElementDataFile = LOCAL\n" + bytes(8))
        cases = {
            "unsupported suffix": self.dir / "x.nrrd",
            "missing file": self.dir / "missing.mha",
            "several channels": self.dir / "rgb.mha",
            "truncated voxels": self.dir / "short.mha",
            "no ElementDataFile": self.dir / "nothing.mha",
            "NDims not a whole number": self.dir / "ndims.mha",
        }
        for what, path in cases.items():
            with self.subTest(what), self.assertRaises(ImageReadError):
                read_image(path)


if __name__ == "__main__":
    unittest.main()
