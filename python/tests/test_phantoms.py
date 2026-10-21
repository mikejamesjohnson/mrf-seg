"""The synthetic CPR vessel: its grid, its geometry, what it is made of, and its files."""

import math
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from mrfseg.grid import voxels
from mrfseg.image_io import read_image
from mrfseg.phantoms import CPRPhantomConfig, make_cpr_phantom, write_cpr_phantom

QUICK = CPRPhantomConfig(seed=7, spacing=0.15)
HEALTHY = replace(QUICK, branches=(0, 0), stenosis=False, soft_plaque=False, calcification=False)


def lumen_area(phantom):
    """mm^2 of lumen in each z slice."""
    sx, sy, _ = phantom.spacing
    return phantom.lumen_fraction.sum(axis=(1, 2)) * sx * sy


class GridTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.phantom = make_cpr_phantom(QUICK)

    def test_runs_along_z_from_0_with_three_largest_radii_either_side(self):
        p = self.phantom
        nx, ny, nz = p.dimensions
        self.assertEqual(nz, round(10 / 0.15) + 1)
        np.testing.assert_allclose(p.z, np.arange(nz) * 0.15)
        half = round(3 * p.radius.max() / 0.15) * 0.15
        self.assertEqual(p.origin, (-half, -half, 0.0))
        self.assertEqual(nx, ny)
        self.assertAlmostEqual(p.origin[0] + (nx - 1) * 0.15, half)
        self.assertEqual(p.image.dtype, np.int16)
        self.assertEqual(p.label.dtype, np.uint8)

    def test_spacing_per_axis_and_a_fixed_width(self):
        p = make_cpr_phantom(replace(QUICK, spacing=(0.1, 0.2, 0.25), half_width=2.0))
        self.assertEqual(p.spacing, (0.1, 0.2, 0.25))
        self.assertEqual(p.dimensions, (41, 21, 41))
        self.assertEqual(p.origin, (-2.0, -2.0, 0.0))

    def test_the_same_seed_makes_the_same_volume(self):
        again = make_cpr_phantom(QUICK)
        np.testing.assert_array_equal(again.image, self.phantom.image)
        self.assertFalse(np.array_equal(make_cpr_phantom(replace(QUICK, seed=8)).label, self.phantom.label))

    def test_the_geometry_does_not_depend_on_the_noise_or_the_voxels(self):
        quiet = make_cpr_phantom(replace(QUICK, noise_std=0.0))
        np.testing.assert_array_equal(quiet.label, self.phantom.label)
        difference = self.phantom.image.astype(float) - quiet.image
        self.assertAlmostEqual(difference.std(), 30.0, delta=3.0)
        finer = make_cpr_phantom(replace(QUICK, spacing=0.1))
        self.assertEqual([lesion.z for lesion in finer.lesions], [lesion.z for lesion in self.phantom.lesions])


class GeometryTest(unittest.TestCase):
    def test_the_vessel_runs_the_whole_length(self):
        p = make_cpr_phantom(QUICK)
        self.assertTrue((p.label.sum(axis=(1, 2)) > 0).all())
        self.assertLess(np.hypot(*p.centerline.T).max(), 1.0)  # near the axis, not on it
        self.assertGreater(np.hypot(*p.centerline.T).max(), 0.05)

    def test_a_healthy_lumen_is_as_wide_as_its_radius(self):
        p = make_cpr_phantom(HEALTHY)
        np.testing.assert_allclose(lumen_area(p), math.pi * p.radius**2, rtol=0.03)
        self.assertTrue(((p.radius >= 0.5) & (p.radius <= 1.5)).all())
        np.testing.assert_array_equal(p.label, (p.lumen_fraction > 0.5).astype(np.uint8))

    def test_a_stenosis_narrows_the_lumen_all_round(self):
        p = make_cpr_phantom(replace(HEALTHY, stenosis=True))
        (stenosis,) = p.lesions
        k = int(round(stenosis.z / 0.15))
        narrowed = math.pi * (p.radius[k] * (1 - stenosis.details["severity"])) ** 2
        self.assertAlmostEqual(lumen_area(p)[k], narrowed, delta=0.1 * narrowed)

    def test_a_soft_plaque_narrows_it_on_one_side(self):
        p = make_cpr_phantom(replace(HEALTHY, soft_plaque=True))
        (plaque,) = p.lesions
        k = int(round(plaque.z / 0.15))
        self.assertLess(lumen_area(p)[k], 0.9 * math.pi * p.radius[k] ** 2)

    def test_a_calcification_is_bright_and_not_lumen(self):
        quiet = replace(HEALTHY, calcification=True, noise_std=0.0, texture=0.0, blur_fwhm=0.0)
        p = make_cpr_phantom(quiet)
        (calcium,) = p.lesions
        k = int(round(calcium.z / 0.15))
        mostly_calcium = p.image[k] > 0.9 * calcium.details["HU"]
        self.assertTrue(mostly_calcium.any())
        self.assertFalse((p.label[k] & mostly_calcium).any())

    def test_a_wandering_centre_reaches_the_mean_radius_and_changes_nothing_else(self):
        drifting = make_cpr_phantom(replace(QUICK, centerline_reach=1.0))
        plain = make_cpr_phantom(QUICK)
        distance = np.hypot(*drifting.centerline.T)
        self.assertAlmostEqual(float(distance.max()), float(drifting.radius.mean()), delta=0.02 * drifting.radius.mean())
        np.testing.assert_allclose(drifting.centerline.mean(axis=0), 0, atol=0.05)  # about the axis, on average
        # Smooth: from one slice to the next it moves less than half a voxel.
        self.assertLess(float(np.hypot(*np.diff(drifting.centerline, axis=0).T).max()), 0.5 * 0.15)
        # The same path, scaled: the radius and the lesions are those of the same seed without it.
        np.testing.assert_array_equal(drifting.radius, plain.radius)
        self.assertEqual([(lesion.kind, lesion.z) for lesion in drifting.lesions], [(lesion.kind, lesion.z) for lesion in plain.lesions])
        self.assertTrue((drifting.label.sum(axis=(1, 2)) > 0).all())

    def test_branches_leave_the_vessel(self):
        p = make_cpr_phantom(replace(HEALTHY, branches=(2, 2)))
        self.assertEqual(len(p.branches), 2)
        plain = make_cpr_phantom(HEALTHY)
        self.assertGreater(int(p.label.sum()), int(plain.label.sum()))
        for branch in p.branches:
            self.assertGreater(branch.direction[2], 0)  # towards larger z
            self.assertAlmostEqual(float(np.linalg.norm(branch.direction)), 1.0)


class FilesTest(unittest.TestCase):
    def test_written_as_read_image_reads_them(self):
        p = make_cpr_phantom(replace(QUICK, spacing=(0.15, 0.15, 0.3)))
        with tempfile.TemporaryDirectory() as folder:
            image_path, label_path = write_cpr_phantom(p, Path(folder) / "vessel")
            self.assertEqual((image_path.name, label_path.name), ("vessel.mha", "vessel_label.mha"))
            for path, values in ((image_path, p.image), (label_path, p.label)):
                read = read_image(path)
                self.assertEqual(read.GetDimensions(), p.dimensions)
                np.testing.assert_allclose(read.GetSpacing(), p.spacing)
                np.testing.assert_allclose(read.GetOrigin(), p.origin)
                np.testing.assert_array_equal(voxels(read), values)


class ConfigTest(unittest.TestCase):
    def test_rejects_nonsense(self):
        for bad in [dict(spacing=0), dict(spacing=(0.1, 0.1)), dict(radius=(2.0, 1.0)), dict(length=-1),
                    dict(supersample=0), dict(branches=(3, 1)), dict(stenosis_severity=(0.5, 1.2)), dict(half_width=0),
                    dict(centerline_reach=-0.5)]:
            with self.subTest(bad), self.assertRaises(ValueError):
                CPRPhantomConfig(**bad)

    def test_too_many_lesions_for_the_length_says_so(self):
        with self.assertRaisesRegex(ValueError, "do not fit"):
            make_cpr_phantom(replace(QUICK, length=2.0))


if __name__ == "__main__":
    unittest.main()
