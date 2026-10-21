"""The feature grid, its interpolation, and running an extractor over it in windows."""

import os
import unittest

import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import gaussian_filter
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonDataModel import vtkImageData

from mrfseg.learned import IN_CHANNELS, network_input
from mrfseg.learned.extractor import Learned
from mrfseg.learned.model import OUT_CHANNELS
from mrfseg.basic import Basic
from mrfseg.features import FeatureVolume, extract_features, feature_grid_dims, resample
from mrfseg.grid import voxels
from tests.helpers import StubExtractor, sphere_volume, stub_extractor

CPU = torch.device("cpu")


def unit(x, axis=-1):
    return x / np.linalg.norm(x, axis=axis, keepdims=True)


def image_of(values, spacing):
    """A vtkImageData holding (z, y, x) `values`."""
    image = vtkImageData()
    nz, ny, nx = values.shape
    image.SetDimensions(nx, ny, nz)
    image.SetSpacing(*spacing)
    image.GetPointData().SetScalars(numpy_to_vtk(values.ravel(), deep=True))
    return image


def blur_then_interpolate(values, dims):
    """What `resample` replaced, done the obvious way on the whole volume: scipy's Gaussian of
    sigma (step - 1) / 2 voxels per axis, edges held, then PyTorch's trilinear interpolation."""
    volume = values.astype(np.float32)
    finite = np.isfinite(volume)
    volume[~finite] = volume[finite].min()
    sigma = [max(0.0, ((n - 1) / max(f - 1, 1) - 1) / 2) for n, f in zip(volume.shape, reversed(dims))]
    volume = gaussian_filter(volume, sigma=sigma, mode="nearest")
    return F.interpolate(torch.from_numpy(volume)[None, None], size=tuple(reversed(dims)), mode="trilinear", align_corners=True)[0, 0].numpy()


class ResampleTest(unittest.TestCase):
    def test_is_a_gaussian_then_trilinear_interpolation(self):
        rng = np.random.default_rng(6)
        cases = {
            "coarser": ((23, 29, 37), (1.0, 1.0, 1.0), 2.5),
            "anisotropic, finer along one axis": ((23, 29, 37), (0.5, 0.7, 1.2), 0.4),
            "much coarser": ((35, 18, 21), (2.0, 2.0, 2.5), 8.0),
            "the same grid": ((20, 20, 20), (1.0, 1.0, 1.0), 1.0),
            "one slice": ((1, 30, 26), (1.5, 1.5, 3.0), 2.0),
            "more slices than a chunk": ((40, 12, 11), (1.0, 1.0, 0.5), 1.5),
        }
        for what, (shape, spacing, target) in cases.items():
            with self.subTest(what):
                values = (rng.normal(size=shape) * 100).astype(np.int16)
                dims = feature_grid_dims(tuple(reversed(shape)), spacing, target)
                got = resample(image_of(values, spacing), target)
                self.assertEqual(got.shape, tuple(reversed(dims)))
                np.testing.assert_allclose(got, blur_then_interpolate(values, dims), atol=1e-5 * np.ptp(values))

    def test_non_finite_voxels_take_the_lowest_finite_value(self):
        values = np.random.default_rng(7).normal(size=(20, 21, 22)).astype(np.float32)
        values[3, 4, 5], values[0, 0, :3], values[19, 2, 2] = np.nan, np.inf, -np.inf
        image = image_of(values, (1.0, 1.0, 1.0))
        before = values.copy()
        got = resample(image, 2.0)
        np.testing.assert_allclose(got, blur_then_interpolate(values, feature_grid_dims((22, 21, 20), (1.0, 1.0, 1.0), 2.0)), atol=1e-5)
        np.testing.assert_array_equal(voxels(image), before)  # the image is left alone


class FeatureGridTest(unittest.TestCase):
    def test_grid_dims(self):
        self.assertEqual(feature_grid_dims((11, 11, 11), (1.0, 1.0, 1.0), 1.0), (11, 11, 11))
        self.assertEqual(feature_grid_dims((11, 21, 1), (1.0, 0.5, 1.0), 2.0), (6, 6, 1))
        # A demo-CT-like grid at 1 mm.
        self.assertEqual(feature_grid_dims((512, 512, 361), (0.333984375, 0.333984375, 0.4), 1.0), (172, 172, 145))

    def linear_volume(self, dims=(5, 4, 3), ct_dims=(9, 7, 5)):
        """Features that are linear in the feature index, which trilinear interpolation reproduces."""
        nx, ny, nz = dims
        k, j, i = np.meshgrid(np.arange(nz), np.arange(ny), np.arange(nx), indexing="ij")
        values = np.stack([i + 1.0, j + 2.0, k + 3.0, i + j + k + 4.0]).astype(np.float16)
        return FeatureVolume(values=values, ct_dims=ct_dims)

    def test_trilinear_at_any_point(self):
        volume = self.linear_volume()
        ijk = np.array([[0, 0, 0], [8, 6, 4], [3.3, 2.7, 1.25], [4, 3, 2]])
        f = ijk * np.array([(5 - 1) / (9 - 1), (4 - 1) / (7 - 1), (3 - 1) / (5 - 1)])
        expected = unit(np.stack([f[:, 0] + 1, f[:, 1] + 2, f[:, 2] + 3, f.sum(1) + 4], 1))
        np.testing.assert_allclose(volume.at_ct_indices(ijk), expected, atol=1e-5)

    def test_a_slice_matches_its_points(self):
        rng = np.random.default_rng(1)
        values = rng.normal(size=(6, 5, 4, 7)).astype(np.float16)
        for unit in (True, False):
            volume = FeatureVolume(values=values, ct_dims=(13, 10, 9), unit=unit)
            for k in (0, 4, 8):
                plane = volume.ct_slice(k, CPU).numpy()  # (C, ny, nx)
                j, i = np.meshgrid(np.arange(10), np.arange(13), indexing="ij")
                ijk = np.stack([i.ravel(), j.ravel(), np.full(i.size, k)], 1)
                np.testing.assert_allclose(plane.reshape(6, -1).T, volume.at_ct_indices(ijk), atol=1e-5)

    def test_features_compared_as_they_are_are_not_rescaled(self):
        linear = self.linear_volume()
        volume = FeatureVolume(values=linear.values, ct_dims=linear.ct_dims, unit=False)
        ijk = np.array([[0, 0, 0], [8, 6, 4], [3.3, 2.7, 1.25]])
        f = ijk * np.array([(5 - 1) / (9 - 1), (4 - 1) / (7 - 1), (3 - 1) / (5 - 1)])
        expected = np.stack([f[:, 0] + 1, f[:, 1] + 2, f[:, 2] + 3, f.sum(1) + 4], 1)
        np.testing.assert_allclose(volume.at_ct_indices(ijk), expected, atol=1e-5)

    def test_the_extractor_says_how_its_features_are_compared(self):
        image, _ = sphere_volume(dims=(24, 22, 18), spacing=(2.0, 2.0, 2.5))
        self.assertTrue(extract_features(image, 4.0, stub_extractor, CPU).unit)
        self.assertFalse(extract_features(image, 4.0, Basic(scales=2), CPU).unit)

    def test_extract_gives_the_extractor_a_unit_range_volume_on_the_feature_grid(self):
        image, _ = sphere_volume(dims=(24, 22, 18), spacing=(2.0, 2.0, 2.5))
        recording = StubExtractor()
        features = extract_features(image, 4.0, recording, CPU)
        grid = feature_grid_dims((24, 22, 18), (2.0, 2.0, 2.5), 4.0)
        self.assertEqual(recording.shapes, [tuple(reversed(grid))])  # (z, y, x)
        self.assertEqual(features.dims, grid)
        self.assertLess(features.dims[0], 24)
        self.assertEqual(features.values.dtype, np.float16)
        (low, high), = recording.ranges
        self.assertEqual((low, high), (0.0, 1.0))  # min-max: the ends are reached

    def test_an_extractor_that_gives_the_wrong_shape_or_non_finite_features_is_refused(self):
        image, _ = sphere_volume(dims=(24, 22, 18), spacing=(2.0, 2.0, 2.5))

        class Wrong(StubExtractor):
            def __call__(self, volume, device, progress=None):
                return super().__call__(volume, device, progress)[:, ::2]

        class Nan(StubExtractor):
            def __call__(self, volume, device, progress=None):
                out = super().__call__(volume, device, progress)
                out[0, 0, 0, 0] = np.nan
                return out

        with self.assertRaisesRegex(RuntimeError, "gave"):
            extract_features(image, 4.0, Wrong(), CPU)
        with self.assertRaisesRegex(RuntimeError, "not finite"):
            extract_features(image, 4.0, Nan(), CPU)


class PointwiseModel(torch.nn.Module):
    """A model with no spatial extent, so windowed and whole-volume results must agree exactly."""

    def __init__(self):
        super().__init__()
        self.conv = torch.nn.Conv3d(IN_CHANNELS, OUT_CHANNELS, 1)

    def forward(self, x):
        return self.conv(x)


class WindowingTest(unittest.TestCase):
    def extractor(self, model, window=32, overlap=0.5):
        extractor = Learned(model.eval(), {"args": {"patch": window}})
        extractor.overlap = overlap
        return extractor

    def test_windows_blend_back_to_the_whole(self):
        extractor = self.extractor(PointwiseModel())
        volume = np.random.default_rng(2).random((40, 70, 55), dtype=np.float32)
        steps = []
        windowed = extractor(volume, CPU, steps.append)
        with torch.no_grad():
            whole = extractor.model()(torch.from_numpy(network_input(volume))[None])[0].numpy()
        np.testing.assert_allclose(windowed, whole, atol=1e-5)
        self.assertEqual(len(steps), 2 * 4 * 3)  # windows of 32 every 16 voxels
        self.assertEqual(steps[-1], 1.0)

    def test_a_side_shorter_than_a_window_is_padded(self):
        extractor = self.extractor(PointwiseModel())
        volume = np.random.default_rng(4).random((20, 45, 33), dtype=np.float32)  # 20 < 32, none a multiple of 16
        windowed = extractor(volume, CPU)
        with torch.no_grad():
            whole = extractor.model()(torch.from_numpy(network_input(volume))[None])[0].numpy()
        self.assertEqual(windowed.shape, (OUT_CHANNELS, 20, 45, 33))
        np.testing.assert_allclose(windowed, whole, atol=1e-5)

    def test_a_volume_that_fits_a_window_is_one_pass(self):
        # A model that sees its neighbours: in windows it would differ near their edges. A volume
        # that fits one window is run whole, as it is (the network pads itself to its stride).
        model = torch.nn.Sequential(torch.nn.Conv3d(IN_CHANNELS, 8, 3, padding=1, padding_mode="reflect"), torch.nn.ReLU(), torch.nn.Conv3d(8, OUT_CHANNELS, 3, padding=1))
        extractor = self.extractor(model, window=48)
        volume = np.random.default_rng(6).random((40, 45, 33), dtype=np.float32)
        steps = []
        got = extractor(volume, CPU, steps.append)
        with torch.no_grad():
            whole = model(torch.from_numpy(network_input(volume))[None])[0].numpy()
        self.assertEqual(steps, [1.0])
        np.testing.assert_allclose(got, whole, atol=1e-5)

    def test_each_voxel_comes_mostly_from_the_window_it_is_deepest_in(self):
        from mrfseg.windows import gaussian_weights

        weights = gaussian_weights((64, 1, 1), 0.25)[:, 0, 0]
        self.assertEqual(int(np.argmax(weights)), 31)
        self.assertLess(weights[0] / weights.max(), 0.15)  # e^-2 at the edge, sigma a quarter of the window


if __name__ == "__main__":
    unittest.main()
