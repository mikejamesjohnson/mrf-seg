"""MRFSegmentation: its reference volume, its settings, and segmenting a sphere end to end."""

import dataclasses
import math
import unittest

import numpy as np
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkCommonDataModel import vtkImageData

from mrfseg import InputError, MRFSegmentation, MRFSettings
from mrfseg.features import MAX_FEATURE_BYTES, default_feature_spacing, feature_bytes_per_voxel, feature_grid_dims
from mrfseg.grid import Grid, voxels
from mrfseg.rw import MAX_GRID_BYTES
from mrfseg.segmentation import MAX_POINTS
from tests.helpers import StubExtractor, rotation_z, sphere_points, sphere_volume, stub_extractor

# The test sphere is a hard step of 300 HU in a background of 0 HU: the contrast affinities cut
# its wall, and the outside points seed the background (nothing is below 0 HU to seed itself).
FAST = MRFSettings(feature_spacing=2.0, device="cpu")


class ReferenceVolumeTest(unittest.TestCase):
    def test_reference_volume_defaults_to_none(self):
        self.assertIsNone(MRFSegmentation().reference_volume)

    def test_reference_volume_can_be_given_to_constructor(self):
        volume = vtkImageData()
        self.assertIs(MRFSegmentation(volume).reference_volume, volume)

    def test_set_and_clear_reference_volume(self):
        volume = vtkImageData()
        segmentation = MRFSegmentation()
        segmentation.reference_volume = volume
        self.assertIs(segmentation.reference_volume, volume)
        segmentation.reference_volume = None  # None is a valid choice
        self.assertIsNone(segmentation.reference_volume)

    def test_rejects_reference_volume_that_is_not_image_data(self):
        for not_a_volume in [[[1, 2], [3, 4]], "CT", 42]:
            with self.subTest(not_a_volume=type(not_a_volume).__name__):
                with self.assertRaises(TypeError):
                    MRFSegmentation(not_a_volume)
                segmentation = MRFSegmentation()
                with self.assertRaises(TypeError):
                    segmentation.reference_volume = not_a_volume
                self.assertIsNone(segmentation.reference_volume)


class SettingsTest(unittest.TestCase):
    def test_rejects_nonsense(self):
        for bad in [dict(feature_spacing=0), dict(contrast=0.5), dict(contrast=float("inf")), dict(device="gpu"), dict(feature_extractor="cnn"),
                    dict(affinity_scale=0), dict(affinity_scale=6), dict(seed_hu=float("nan")), dict(seed_erosion=-1), dict(seed_erosion=1.5),
                    dict(tolerance=0), dict(tolerance=0.5), dict(max_iterations=0), dict(max_iterations=2.5), dict(preconditioner="none"),
                    # What JSON could send: none of these is what it looks like.
                    dict(max_iterations=True), dict(contrast=True), dict(background_seeds="false"), dict(feature_spacing="1"),
                    dict(seed_hu="0"), dict(seed_hu=True), dict(background_seeds=1)]:
            with self.subTest(bad), self.assertRaises(InputError):
                MRFSettings(**bad)

    def test_every_field_is_checked(self):
        for field in dataclasses.fields(MRFSettings):
            with self.subTest(field.name), self.assertRaises(InputError):
                MRFSettings(**{field.name: object()})
        # Out of bounds, and too large for a float at all.
        for bad in [dict(contrast=100000.0), dict(seed_hu=-2000), dict(feature_spacing=0.0001), dict(contrast=10**400), dict(max_iterations=10**400)]:
            with self.subTest(bad), self.assertRaisesRegex(InputError, "must be"):
                MRFSettings(**bad)

    def test_by_default_the_image_chooses_the_grid(self):
        defaults = MRFSettings()
        self.assertIsNone(defaults.feature_spacing)
        self.assertEqual((defaults.feature_extractor, defaults.contrast, defaults.background_seeds, defaults.seed_hu, defaults.seed_erosion), ("basic", 40.0, True, 0.0, 1))
        self.assertEqual((defaults.tolerance, defaults.max_iterations, defaults.preconditioner), (1e-4, 1000, "multigrid"))
        volume, _ = sphere_volume(spacing=(2.0, 1.5, 2.5))
        segmentation = MRFSegmentation(volume, MRFSettings(), stub_extractor)
        self.assertEqual(segmentation.settings.feature_spacing, 1.5)  # the finest voxel size
        # A given value stays; a new image chooses again.
        self.assertEqual(MRFSegmentation(volume, MRFSettings(feature_spacing=3.0)).settings.feature_spacing, 3.0)
        segmentation.reference_volume, _ = sphere_volume(spacing=(1.0, 1.0, 1.25))
        self.assertEqual(segmentation.settings.feature_spacing, 1.0)
        # But no finer than 0.5 mm, the grid the network is trained on and lumen is 2 voxels across.
        segmentation.reference_volume, _ = sphere_volume(spacing=(0.3, 0.3, 0.4))
        self.assertEqual(segmentation.settings.feature_spacing, 0.5)
        self.assertEqual((defaults.solver.tolerance, defaults.solver.max_iterations), (1e-4, 1000))

    def test_too_fine_an_image_extracts_on_a_grid_that_fits_in_memory(self):
        self.assertEqual(default_feature_spacing((100, 100, 100), (0.5, 0.5, 0.6), 16), 0.5)
        # 512 x 512 x 361 CT at 0.33 mm: 34M voxels at the 0.5 mm floor, 100 bytes each for 16 features
        # (3.4 GB, fits); 40 features do not, and double the spacing.
        ct = ((512, 512, 361), (0.333984375, 0.333984375, 0.4))
        self.assertAlmostEqual(default_feature_spacing(*ct, 16), 0.5)
        self.assertAlmostEqual(default_feature_spacing(*ct, 4), 0.5)
        self.assertAlmostEqual(default_feature_spacing(*ct, 40), 1.0)
        self.assertAlmostEqual(default_feature_spacing(*ct, 4, native=0.75), 0.75)
        # Doubled as often as it takes, and the result always fits.
        huge = ((1024, 1024, 1500), (0.2, 0.2, 0.4))
        for channels in (4, 16, 40):
            spacing = default_feature_spacing(*huge, channels)
            self.assertLessEqual(math.prod(feature_grid_dims(*huge, spacing)) * feature_bytes_per_voxel(channels), MAX_FEATURE_BYTES)
            self.assertGreater(math.prod(feature_grid_dims(*huge, spacing / 2)) * feature_bytes_per_voxel(channels), MAX_FEATURE_BYTES)

    def test_a_feature_grid_too_large_says_what_would_fit(self):
        volume, _ = sphere_volume()
        segmentation = MRFSegmentation(volume, dataclasses.replace(FAST, feature_spacing=0.02), stub_extractor)
        with self.assertRaisesRegex(InputError, r"use at least 0\.\d+ mm") as raised:
            segmentation.extract_features()
        with self.assertRaises(InputError):
            segmentation.check_settings()
        fits = float(str(raised.exception).rsplit("use at least ", 1)[1].split()[0])
        MRFSegmentation(volume, dataclasses.replace(FAST, feature_spacing=fits), stub_extractor).check_settings()  # no error
        self.assertIsNone(segmentation.features)  # nothing extracted

    def test_a_grid_too_large_to_solve_on_is_refused_too(self):
        from mrfseg.rw import check_grid

        check_grid((100, 100, 100))
        with self.assertRaisesRegex(InputError, "coarser resampling spacing"):
            check_grid((2000, 2000, 1000))
        from mrfseg.segmentation import check_grids

        check_grids((512, 512, 400), (0.4, 0.4, 0.5), 0.5, 4)
        with self.assertRaisesRegex(InputError, r"use at least 0\.8 mm"):
            check_grids((512, 512, 400), (0.4, 0.4, 0.5), 0.2, 4)
        # The features fit at 0.4 (3.6 GB for 4 features), but the solver does not until 0.8.
        with self.assertRaisesRegex(InputError, r"to solve on.*use at least 0\.8 mm"):
            check_grids((512, 512, 400), (0.4, 0.4, 0.5), 0.4, 4)
        self.assertGreater(MAX_GRID_BYTES, 1 << 30)


class GenerateSegmentationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # A rotated grid, so world and index space differ.
        cls.volume, cls.world = sphere_volume(axes=rotation_z(20))
        cls.points, cls.labels = sphere_points(rotation_z(20))
        cls.segmentation = MRFSegmentation(cls.volume, FAST, stub_extractor)
        cls.result = cls.segmentation.generate_segmentation(cls.points, cls.labels)
        cls.truth = np.linalg.norm(cls.world, axis=-1) <= 14.0

    def test_needs_a_volume_and_sensible_points(self):
        with self.assertRaises(ValueError):
            MRFSegmentation(None, FAST, stub_extractor).generate_segmentation(self.points, self.labels)
        segmentation = MRFSegmentation(self.volume, FAST, stub_extractor)
        bad = {
            "labels not +1/-1": (self.points, np.where(self.labels > 0, 1, 0)),
            "outside points only": (self.points[4:], self.labels[4:]),
            "no points": (np.zeros((0, 3)), np.zeros(0, dtype=int)),
            "mismatched lengths": (self.points, self.labels[:-1]),
            "outside the image": (np.vstack([self.points, [[500, 0, 0]]]), np.append(self.labels, -1)),
            "not finite": (np.vstack([self.points, [[np.nan, 0, 0]]]), np.append(self.labels, -1)),
            "two labels on one voxel": (np.vstack([self.points, self.points[:1]]), np.append(self.labels, -1)),
        }
        many = np.zeros((MAX_POINTS + 1, 3))
        bad["too many points"] = (many, np.where(np.arange(len(many)) % 2, 1, -1))
        for what, (points, labels) in bad.items():
            with self.subTest(what), self.assertRaises(InputError):
                segmentation.generate_segmentation(points, labels)
        self.assertTrue(segmentation.needs_inside_point())

    def test_bad_points_keep_the_last_segmentation(self):
        segmentation = MRFSegmentation(self.volume, FAST, stub_extractor)
        first = segmentation.generate_segmentation(self.points, self.labels)
        with self.assertRaises(InputError):
            segmentation.generate_segmentation(self.points[4:], self.labels[4:])
        self.assertIs(segmentation.last_segmentation, first)

    def test_two_images_on_the_reference_grid(self):
        for name in ("probability", "affinity"):
            with self.subTest(name):
                image = getattr(self.result, name)
                self.assertIsNot(image, self.volume)
                self.assertEqual(image.GetDimensions(), self.volume.GetDimensions())
                self.assertEqual(image.GetSpacing(), self.volume.GetSpacing())
                self.assertEqual(image.GetOrigin(), self.volume.GetOrigin())
                for r in range(3):
                    for c in range(3):
                        self.assertEqual(
                            image.GetDirectionMatrix().GetElement(r, c), self.volume.GetDirectionMatrix().GetElement(r, c)
                        )
                scalars = image.GetPointData().GetScalars()
                self.assertEqual((scalars.GetName(), scalars.GetDataTypeAsString()), (name, "float"))
                self.assertTrue(np.isfinite(voxels(image)).all())

    def test_segments_the_sphere(self):
        probability = voxels(self.result.probability)
        accuracy = ((probability > 0.5) == self.truth).mean()
        self.assertGreater(accuracy, 0.95, accuracy)
        self.assertGreater(probability[self.truth].mean(), 0.9)
        self.assertLess(probability[~self.truth].mean(), 0.1)
        self.assertTrue((probability >= 0).all() and (probability <= 1).all())

    def test_the_points_are_seeds(self):
        probability = voxels(self.result.probability)
        ijk = np.rint(Grid(self.volume).world_to_index(self.points)).astype(int)
        at_points = probability[ijk[:, 2], ijk[:, 1], ijk[:, 0]]
        # The seeded voxel is on the feature grid, read back onto the CT's grid between two of
        # its planes (the CT's 2.5 mm slices against the 2 mm grid): 1 and 0, but for that.
        np.testing.assert_allclose(at_points, (self.labels > 0).astype(float), atol=1e-2)

    def test_the_affinity_is_low_across_the_sphere_s_wall(self):
        affinity = voxels(self.result.affinity)
        radius = np.linalg.norm(self.world, axis=-1)
        self.assertLess(affinity[np.abs(radius - 14.0) < 2.0].mean(), affinity[radius < 8.0].mean() - 3.0)
        self.assertAlmostEqual(float(affinity[radius < 8.0].mean()), 0.0, delta=0.05)  # flat inside: log 1

    def test_reports_the_solve_and_the_seeds(self):
        inference = self.result.inference
        self.assertLessEqual(inference.iterations, FAST.max_iterations)
        self.assertTrue(inference.converged)
        self.assertLessEqual(inference.residual, FAST.tolerance)
        self.assertEqual(inference.components, 1)
        self.assertEqual(inference.seeds, {"inside": 4, "outside": 6, "background": 0})  # nothing is below 0 HU
        self.assertEqual(set(self.result.seconds), {"features", "affinities", "seeds", "solve", "maps"})
        stages = []
        MRFSegmentation(self.volume, FAST, stub_extractor).generate_segmentation(self.points, self.labels, stage=stages.append)
        self.assertEqual(stages, ["features", "affinities", "seeds", "solve", "maps"])

    def test_background_seeds_stand_in_for_outside_points(self):
        # With the threshold above the background's 0 HU, the background seeds itself, and the
        # inside points alone segment the sphere.
        seeded = dataclasses.replace(FAST, seed_hu=50.0)
        result = MRFSegmentation(self.volume, seeded, stub_extractor).generate_segmentation(self.points[:4], self.labels[:4])
        self.assertGreater(result.inference.seeds["background"], 1000)
        self.assertGreater(((voxels(result.probability) > 0.5) == self.truth).mean(), 0.95)
        # Eroded: the voxels next to the sphere are not seeds, and more erosion seeds fewer.
        more = MRFSegmentation(self.volume, dataclasses.replace(seeded, seed_erosion=3), stub_extractor).generate_segmentation(self.points[:4], self.labels[:4])
        self.assertLess(more.inference.seeds["background"], result.inference.seeds["background"])
        # Off: inside points alone leave the whole image at 1 (only the sink pulls down).
        off = MRFSegmentation(self.volume, dataclasses.replace(FAST, background_seeds=False), stub_extractor).generate_segmentation(self.points[:4], self.labels[:4])
        self.assertEqual(off.inference.seeds["background"], 0)
        self.assertGreater(voxels(off.probability).min(), 0.9)

    def test_an_outside_point_only_lowers_the_probability(self):
        """The comparison principle, end to end."""
        more = MRFSegmentation(self.volume, FAST, stub_extractor).generate_segmentation(
            np.vstack([self.points, [[0.0, 0.0, 10.0]]]), np.append(self.labels, -1))  # an outside point inside the sphere
        p, q = voxels(self.result.probability), voxels(more.probability)
        self.assertTrue(bool((q <= p + 1e-5).all()))
        self.assertLess(q[self.truth].mean(), p[self.truth].mean() - 0.1)

    def test_the_contrast_setting_changes_the_affinities(self):
        soft = MRFSegmentation(self.volume, dataclasses.replace(FAST, contrast=10000.0), stub_extractor).generate_segmentation(self.points, self.labels)
        # With a contrast scale far above any step the graph is uniform: the probability blurs
        # out of the sphere, and the affinity map is flat.
        self.assertLess(((voxels(soft.probability) > 0.5) == self.truth).mean(), 0.9)
        self.assertAlmostEqual(float(np.abs(voxels(soft.affinity)).max()), 0.0, places=3)

    def test_features_are_extracted_once_per_spacing_and_the_solve_restarts_from_the_last(self):
        counting = StubExtractor()
        segmentation = MRFSegmentation(self.volume, FAST, counting)
        first = segmentation.generate_segmentation(self.points, self.labels)
        again = segmentation.generate_segmentation(self.points, self.labels)
        self.assertEqual(len(counting.shapes), 1)
        self.assertEqual(again.seconds["features"], 0.0)
        self.assertLess(again.inference.iterations, first.inference.iterations)  # from the last probability: nothing to do
        np.testing.assert_allclose(voxels(again.probability), voxels(first.probability), atol=1e-3)
        segmentation.settings = dataclasses.replace(FAST, feature_spacing=3.0)
        segmentation.generate_segmentation(self.points, self.labels)
        self.assertEqual(len(counting.shapes), 2)
        self.assertNotEqual(counting.shapes[0], counting.shapes[1])

    def test_a_generate_that_fails_keeps_the_last_segmentation(self):
        extractor = StubExtractor()
        segmentation = MRFSegmentation(self.volume, FAST, extractor)
        first = segmentation.generate_segmentation(self.points, self.labels)
        segmentation.settings = dataclasses.replace(FAST, feature_spacing=0.02)  # a grid too large for memory
        with self.assertRaises(InputError):
            segmentation.generate_segmentation(self.points, self.labels)
        self.assertIs(segmentation.last_segmentation, first)
        segmentation.settings = dataclasses.replace(FAST, feature_spacing=3.0)  # new features, and extracting them fails
        extractor.fail_with = RuntimeError("no features today")
        with self.assertRaisesRegex(RuntimeError, "no features today"):
            segmentation.generate_segmentation(self.points, self.labels)
        self.assertIs(segmentation.last_segmentation, first)
        self.assertIsNone(segmentation.features)  # the old ones are gone, and nothing says otherwise
        segmentation.generate_segmentation(self.points, self.labels)  # and once it works, the new one replaces it
        self.assertIsNot(segmentation.last_segmentation, first)

    def test_reference_volume_is_left_alone(self):
        before = voxels(self.volume).copy()
        MRFSegmentation(self.volume, FAST, stub_extractor).generate_segmentation(self.points, self.labels)
        np.testing.assert_array_equal(voxels(self.volume), before)

    def test_surfaces(self):
        with self.assertRaises(ValueError):
            MRFSegmentation(self.volume, FAST, stub_extractor).extract_surface()
        half = self.segmentation.extract_surface("probability", 0.5)
        default = self.segmentation.extract_surface()
        self.assertGreater(half.GetNumberOfPolys(), 100)
        self.assertEqual(half.GetNumberOfPolys(), default.GetNumberOfPolys())
        # Thresholds either side of 0.5 nest inside and outside it.
        radius = lambda mesh: float(np.linalg.norm(vtk_to_numpy(mesh.GetPoints().GetData()), axis=1).mean())  # noqa: E731
        self.assertLessEqual(radius(self.segmentation.extract_surface("probability", 0.8)), radius(half) + 1e-6)
        self.assertGreaterEqual(radius(self.segmentation.extract_surface("probability", 0.2)), radius(half) - 1e-6)
        # Near the sphere's radius, with outward normals.
        a = vtk_to_numpy(half.GetPoints().GetData())
        self.assertLess(abs(np.linalg.norm(a, axis=1).mean() - 14.0), 3.0)
        normals = vtk_to_numpy(half.GetPointData().GetNormals())
        self.assertGreater(np.einsum("ij,ij->i", normals, a / np.linalg.norm(a, axis=1, keepdims=True)).mean(), 0.8)
        for bad in [("probability", 1.5), ("probability", 0.0), ("contour", None), ("unary", None), ("affinity", 0.0)]:
            with self.subTest(bad), self.assertRaises(ValueError):
                self.segmentation.extract_surface(*bad)

    def test_labelmap(self):
        with self.assertRaises(ValueError):
            MRFSegmentation(self.volume, FAST, stub_extractor).labelmap()
        half = self.segmentation.labelmap(0.5)
        self.assertEqual(half.GetDimensions(), self.volume.GetDimensions())
        self.assertEqual(half.GetPointData().GetScalars().GetDataTypeAsString(), "unsigned char")
        labels = voxels(half)
        self.assertEqual(set(np.unique(labels)), {0, 1})
        probability = voxels(self.result.probability)
        np.testing.assert_array_equal(labels, (probability >= 0.5).astype(np.uint8))
        strict = voxels(self.segmentation.labelmap(0.8))
        np.testing.assert_array_equal(strict, (probability >= 0.8).astype(np.uint8))
        self.assertTrue((strict <= labels).all())
        self.assertGreater((labels.astype(bool) == self.truth).mean(), 0.95)
        for bad in (0.0, 1.0, -0.2):
            with self.subTest(bad), self.assertRaises(ValueError):
                self.segmentation.labelmap(bad)


class OddImagesTest(unittest.TestCase):
    def test_a_single_slice(self):
        volume, world = sphere_volume(dims=(30, 26, 1), spacing=(1.5, 1.5, 3.0), radius=12.0)
        points = np.array([[0, 0, 0], [4, 3, 0], [-16, 0, 0], [0, 15, 0], [16, -15, 0], [-16, 14, 0]], dtype=float)
        labels = np.array([1, 1, -1, -1, -1, -1])
        segmentation = MRFSegmentation(volume, FAST, stub_extractor)
        result = segmentation.generate_segmentation(points, labels)
        truth = np.linalg.norm(world, axis=-1) <= 12.0
        self.assertGreater(((voxels(result.probability) > 0.5) == truth).mean(), 0.9)
        self.assertEqual(int(voxels(segmentation.labelmap()).sum()), int((voxels(result.probability) >= 0.5).sum()))
        self.assertEqual(segmentation.extract_surface().GetNumberOfPoints(), 0)  # nothing to bound in one slice

    def test_nan_voxels_do_not_poison_the_features(self):
        volume, _ = sphere_volume()
        values = voxels(volume).astype(np.float32)
        values[0, 0, :5] = np.nan
        from vtkmodules.util.numpy_support import numpy_to_vtk

        volume.GetPointData().SetScalars(numpy_to_vtk(values.ravel(), deep=True))
        segmentation = MRFSegmentation(volume, FAST, stub_extractor)
        self.assertTrue(np.isfinite(segmentation.extract_features().values).all())
        self.assertTrue(np.isfinite(segmentation.features.intensity).all())
        points, labels = sphere_points()
        self.assertTrue(np.isfinite(voxels(segmentation.generate_segmentation(points, labels).probability)).all())


if __name__ == "__main__":
    unittest.main()
