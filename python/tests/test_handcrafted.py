"""The handcrafted potentials: the kernel-density unary from the points and the image, and the
contrast-sensitive coupling."""

import unittest

import numpy as np
import torch

from mrfseg import InputError, MRFSegmentation, MRFSettings
from mrfseg.features import FeatureVolume
from mrfseg.grid import Grid, image_from_voxels
from mrfseg.handcrafted import Background, HandcraftedParameters, contrast_coupling, feature_slice_positions, handcrafted_potentials, kde_unary
from mrfseg.mrf import AXES
from tests.helpers import sphere_points, sphere_volume

CPU = torch.device("cpu")
PARAMETERS = HandcraftedParameters(feature_lengthscale=0.5, spatial_lengthscale=50.0, prior_logit=-2.0, unary_scale=1.0, coupling=1.0, contrast=100.0, coupling_floor=0.1)


def tube_image(shape=(16, 20, 24), radius=2.5, hu_inside=400.0, hu_outside=0.0):
    """A bright tube along x through the middle, as a vtkImageData at 1 mm, and its label."""
    z, y, x = np.mgrid[: shape[0], : shape[1], : shape[2]].astype(np.float32)
    inside = ((z - shape[0] / 2 + 0.5) ** 2 + (y - shape[1] / 2 + 0.5) ** 2) <= radius**2
    hu = np.where(inside, hu_inside, hu_outside).astype(np.float32)
    rng = np.random.default_rng(0)
    hu += rng.normal(0, 10, hu.shape).astype(np.float32)
    return image_from_voxels(hu.astype(np.int16), 1.0), inside


class CouplingTest(unittest.TestCase):
    def test_falls_with_contrast_and_keeps_the_floor(self):
        intensity = torch.zeros((2, 2, 3))
        intensity[0, 0, 1] = 100.0   # a 100 HU step along x between (0,0,0) and (0,0,1), and back to 0 at (0,0,2)
        intensity[1, 1, 2] = 1000.0  # a huge step along z between (0,1,2) and (1,1,2)
        w = contrast_coupling(intensity, PARAMETERS)
        self.assertEqual(tuple(w.shape), (AXES, 2, 2, 3))
        lam, floor = PARAMETERS.coupling, PARAMETERS.coupling_floor
        self.assertAlmostEqual(float(w[2, 0, 0, 0]), lam * (floor + (1 - floor) * np.exp(-0.5)), places=5)  # 100 HU: 61 % of the rest
        self.assertAlmostEqual(float(w[2, 1, 0, 0]), lam, places=5)  # no contrast: lambda
        self.assertAlmostEqual(float(w[0, 0, 1, 2]), lam * floor, places=5)  # 1000 HU: the floor alone
        # The last plane along each axis is unused.
        for d in range(AXES):
            self.assertEqual(float(w[d].narrow(d, w.shape[1 + d] - 1, 1).abs().max()), 0.0)
        self.assertTrue((w >= 0).all())

    def test_floor_is_checked(self):
        with self.assertRaises(InputError):
            contrast_coupling(torch.zeros((2, 2, 2)), HandcraftedParameters(0.5, 10.0, 0.0, 1.0, 1.0, 100.0, 1.5))


class BackgroundTest(unittest.TestCase):
    def test_whitens_the_image_s_features(self):
        rng = np.random.default_rng(0)
        values = rng.normal(size=(3, 10, 12, 14)).astype(np.float32)
        values[1] = values[0] * 0.6 + rng.normal(scale=0.8, size=(10, 12, 14))  # two correlated channels
        values[2] = values[2] * 3 + 5  # another scale and offset
        features = FeatureVolume(values.astype(np.float16), ct_dims=(14, 12, 10), unit=False)
        background = Background.of(features, CPU)
        whitened = background.whiten(torch.from_numpy(values.reshape(3, -1).T))
        # Zero mean and unit covariance, but for the sample and the shrinkage toward the diagonal.
        self.assertLess(float(whitened.mean(0).abs().max()), 0.1)
        torch.testing.assert_close(torch.cov(whitened.T), torch.eye(3), atol=0.2, rtol=0)
        # The Mahalanobis distance of a voxel at the mean is 0, and grows with the offset in units of the spread.
        at_mean = background.whiten(background.mean[None])
        self.assertLess(float(at_mean.abs().max()), 1e-5)
        off = background.whiten(background.mean[None] + torch.tensor([[0.0, 0.0, 3.0]]))  # one standard deviation along channel 2
        self.assertAlmostEqual(float(off.norm()), 1.0, delta=0.25)


class UnaryTest(unittest.TestCase):
    def test_positions_follow_the_grid(self):
        volume, world = sphere_volume(spacing=(2.0, 2.0, 2.5))
        grid = Grid(volume)
        features = FeatureVolume(np.zeros((1, 18, 22, 24), dtype=np.float16), ct_dims=volume.GetDimensions(), unit=False)
        positions = feature_slice_positions(features, grid, CPU)
        for k in (0, 7, 17):
            got = positions(k).numpy().reshape(22, 24, 3) + grid.center
            np.testing.assert_allclose(got, world[k], atol=1e-4)

    def test_lumen_like_voxels_get_positive_logits_and_the_rest_negative(self):
        image, inside = tube_image()
        segmentation = MRFSegmentation(image, MRFSettings(feature_spacing=1.0, device="cpu", basic_scales=2))
        features = segmentation.extract_features()
        grid = Grid(image)
        z, y, x = np.argwhere(inside)[0]
        centre = np.array(inside.shape) // 2
        points = np.array([[centre[2] + 4, centre[1], centre[0]], [centre[2] - 5, centre[1], centre[0]], [1.0, 1.0, 1.0], [22.0, 18.0, 14.0]], dtype=float)  # (x, y, z) mm at 1 mm
        labels = np.array([1, 1, -1, -1])
        parameters = HandcraftedParameters(feature_lengthscale=1.0, spatial_lengthscale=100.0, prior_logit=0.0, unary_scale=1.0, coupling=1.0, contrast=100.0, coupling_floor=0.1)
        unary = kde_unary(features, grid, points, labels, parameters, CPU).numpy()
        self.assertEqual(unary.shape, inside.shape)
        self.assertTrue(np.isfinite(unary).all())
        self.assertGreater(unary[inside].mean(), 0)
        self.assertLess(unary[~inside].mean(), 0)
        self.assertGreater((unary[inside] > 0).mean(), 0.7)
        self.assertGreater((unary[~inside] < 0).mean(), 0.9)
        # Inside points alone: still a ratio against the image's own background.
        alone = kde_unary(features, grid, points[:2], labels[:2], parameters, CPU).numpy()
        self.assertGreater(alone[inside].mean(), alone[~inside].mean())
        # No inside point: refused.
        with self.assertRaisesRegex(InputError, "inside"):
            kde_unary(features, grid, points[2:], labels[2:], parameters, CPU)
        # The prior logit shifts every voxel by as much; the evidence scale stretches the rest.
        shifted = kde_unary(features, grid, points, labels, HandcraftedParameters(1.0, 100.0, -3.0, 1.0, 1.0, 100.0, 0.1), CPU).numpy()
        np.testing.assert_allclose(shifted, unary - 3.0, atol=1e-4)
        doubled = kde_unary(features, grid, points, labels, HandcraftedParameters(1.0, 100.0, 0.0, 2.0, 1.0, 100.0, 0.1), CPU).numpy()
        np.testing.assert_allclose(doubled, 2 * unary, atol=1e-3)

    def test_the_spatial_length_scale_makes_a_point_local(self):
        image, inside = tube_image(shape=(16, 20, 40))
        segmentation = MRFSegmentation(image, MRFSettings(feature_spacing=1.0, device="cpu", basic_scales=2))
        features = segmentation.extract_features()
        grid = Grid(image)
        points = np.array([[3.0, 10.0, 8.0]])  # inside the tube, at one end (x = 3 mm)
        labels = np.array([1])
        far = inside.copy()
        far[:, :, :30] = False  # the tube's far end
        near = inside.copy()
        near[:, :, 8:] = False
        local = kde_unary(features, grid, points, labels, HandcraftedParameters(1.0, 3.0, 0.0, 1.0, 1.0, 100.0, 0.1), CPU).numpy()
        everywhere = kde_unary(features, grid, points, labels, HandcraftedParameters(1.0, 1000.0, 0.0, 1.0, 1.0, 100.0, 0.1), CPU).numpy()
        self.assertLess(local[far].mean(), local[near].mean() - 5)
        self.assertLess(abs(everywhere[far].mean() - everywhere[near].mean()), 3)

    def test_potentials_need_the_intensity(self):
        image, _ = tube_image()
        segmentation = MRFSegmentation(image, MRFSettings(feature_spacing=1.0, device="cpu", basic_scales=2))
        features = segmentation.extract_features()
        without = FeatureVolume(features.values, features.ct_dims, features.unit, intensity=None)
        points, labels = np.array([[12.0, 10.0, 8.0]]), np.array([1])
        with self.assertRaisesRegex(InputError, "intensity"):
            handcrafted_potentials(without, Grid(image), points, labels, PARAMETERS, CPU)
        potentials = handcrafted_potentials(features, Grid(image), points, labels, PARAMETERS, CPU)
        self.assertEqual(potentials.shape, features.shape)
        self.assertTrue((potentials.coupling >= 0).all())


if __name__ == "__main__":
    unittest.main()
