"""Basic features: what they are, how they are scaled, and that an MRFSegmentation uses them."""

import dataclasses
import importlib.util
import unittest

import numpy as np
import torch
from scipy.ndimage import gaussian_filter

from mrfseg import Basic, MRFSegmentation, MRFSettings, InputError
from mrfseg.learned.extractor import Learned
from mrfseg.basic import _hessian_eigenvalues
from mrfseg.compute import resolve_device
from mrfseg.extractors import features_key
from mrfseg.grid import voxels
from tests.helpers import StubExtractor, rotation_z, sphere_points, sphere_volume, stub_extractor

CPU = torch.device("cpu")


def smooth_noise(shape=(20, 24, 22), seed=0):
    return gaussian_filter(np.random.default_rng(seed).random(shape, dtype=np.float32), 1.5)


class BasicTest(unittest.TestCase):
    def test_how_many_features(self):
        volume = smooth_noise()
        for options, channels in [({}, 4), ({"edges": True}, 8), ({"shape": True}, 16), ({"scales": 5, "edges": True, "shape": True}, 25), ({"scales": 3}, 3)]:
            with self.subTest(**options):
                features = Basic(**options)(volume, CPU)
                self.assertEqual((features.shape, features.dtype), ((channels, *volume.shape), np.float32))
                self.assertEqual(Basic(**options).channels, channels)

    def test_the_intensities_are_blurs_at_doubling_scales(self):
        volume = smooth_noise()
        features = Basic(scales=4)(volume, CPU)
        for k in range(4):
            blurred = gaussian_filter(volume, 2.0**k, mode="nearest")
            np.testing.assert_allclose(features[k], (blurred - blurred.mean()) / blurred.std() / 2, atol=1e-4)  # 1/sqrt(4)

    def test_each_is_standardised_then_all_scaled_by_root_c(self):
        features = Basic(scales=5, edges=True, shape=True)(smooth_noise(), CPU)
        np.testing.assert_allclose(features.mean(axis=(1, 2, 3)), 0, atol=1e-5)
        np.testing.assert_allclose(features.std(axis=(1, 2, 3)), 1 / np.sqrt(25), rtol=1e-3)
        self.assertFalse(Basic.unit_vectors)  # compared as they are, not as directions

    def test_shape_tells_a_tube_from_a_blob(self):
        z, y, x = np.mgrid[:21, :21, :21].astype(np.float32) - 10
        tube = np.exp(-(x**2 + y**2) / 8)  # along z
        blob = np.exp(-(x**2 + y**2 + z**2) / 8)
        along, *across = _hessian_eigenvalues(tube)[:, 10, 10, 10]
        self.assertAlmostEqual(float(along), 0.0, delta=1e-4)
        self.assertTrue(all(value < -0.1 for value in across))
        self.assertTrue((_hessian_eigenvalues(blob)[:, 10, 10, 10] < -0.1).all())

    def test_one_slice_and_progress(self):
        steps = []
        features = Basic(scales=5, edges=True, shape=True)(smooth_noise((1, 24, 22)), CPU, steps.append)
        self.assertEqual(features.shape, (25, 1, 24, 22))
        self.assertTrue(np.isfinite(features).all())
        self.assertEqual(steps, [0.2, 0.4, 0.6, 0.8, 1.0])


@unittest.skipUnless(importlib.util.find_spec("skimage"), "scikit-image is not installed")
class ScikitImageTest(unittest.TestCase):
    def test_the_same_features_as_multiscale_basic_features(self):
        from skimage.feature import multiscale_basic_features

        volume = smooth_noise((18, 20, 22), seed=3)
        ours = Basic(scales=3, edges=True, shape=True)(volume, CPU)
        theirs = np.moveaxis(multiscale_basic_features(volume, sigma_min=1, sigma_max=4, num_sigma=3), -1, 0)
        theirs = (theirs - theirs.mean(axis=(1, 2, 3), keepdims=True)) / theirs.std(axis=(1, 2, 3), keepdims=True) / np.sqrt(len(theirs))
        np.testing.assert_allclose(ours, theirs, atol=1e-3)


class ChoosingTest(unittest.TestCase):
    def test_the_settings_choose_the_extractor(self):
        volume, _ = sphere_volume()
        basic = MRFSegmentation(volume, MRFSettings(basic_scales=3, basic_shape=True)).feature_extractor()
        self.assertEqual(basic, Basic(3, False, True))
        learned = MRFSegmentation(volume, MRFSettings(feature_extractor="learned"))
        self.assertIsInstance(learned.feature_extractor(), Learned)
        self.assertIs(MRFSegmentation(volume, MRFSettings(), stub_extractor).feature_extractor(), stub_extractor)
        with self.assertRaisesRegex(InputError, "feature_extractor must be one of"):
            MRFSettings(feature_extractor="cnn")
        with self.assertRaises(InputError):
            features_key(dataclasses.replace(MRFSettings(), feature_extractor="cnn"))  # a name no entry has, however it got there

    def test_the_features_key_is_what_the_chosen_extractor_reads(self):
        basic = MRFSettings(feature_spacing=1.0, basic_scales=3, basic_edges=True, basic_shape=False, device="cpu")
        self.assertEqual(features_key(basic), ("basic", 1.0, 3, True, False))
        self.assertEqual(features_key(dataclasses.replace(basic, device="auto")), features_key(basic))
        learned = dataclasses.replace(basic, feature_extractor="learned")
        self.assertEqual(features_key(learned)[:3], ("learned", 1.0, "cpu"))  # then the checkpoint's path and mtime
        self.assertEqual(features_key(dataclasses.replace(learned, basic_scales=8)), features_key(learned))

    def test_extracted_again_only_for_what_the_extractor_reads(self):
        volume, _ = sphere_volume()
        counting = StubExtractor()
        calls = counting.shapes
        settings = MRFSettings(feature_spacing=2.0, device="cpu")
        gp = MRFSegmentation(volume, settings, counting)
        steps = [
            ({}, 1),
            ({"basic_scales": 3}, 2),
            ({"basic_scales": 3, "device": "auto"}, 2),  # Basic runs on the CPU whatever this says
            ({"feature_extractor": "learned"}, 3),
            ({"feature_extractor": "learned", "basic_shape": True}, 3),  # not read by the network
            # "auto" is another device only where there is an Apple GPU.
            ({"feature_extractor": "learned", "device": "auto"}, 4 if resolve_device("auto") != resolve_device("cpu") else 3),
        ]
        for change, extracted in steps:
            with self.subTest(**change):
                gp.settings = dataclasses.replace(settings, **change)
                gp.extract_features()
                self.assertEqual(len(calls), extracted)

    def test_segments_the_sphere_with_basic_features(self):
        volume, world = sphere_volume(axes=rotation_z(20))
        points, labels = sphere_points(rotation_z(20))
        # The sphere is a third of its image, so it is not the rare structure the handcrafted unary
        # expects (lumen is a fraction of a percent of a CT): the prior logit is raised to say so.
        settings = MRFSettings(feature_spacing=2.0, device="cpu", spatial_lengthscale=15.0, prior_logit=2.0, feature_lengthscale=1.0)
        segmentation = MRFSegmentation(volume, settings)
        probability = voxels(segmentation.generate_segmentation(points, labels).probability)
        self.assertEqual((segmentation.features.channels, segmentation.features.unit), (4, False))  # four blurs, by default
        truth = np.linalg.norm(world, axis=-1) <= 14.0
        self.assertGreater(((probability > 0.5) == truth).mean(), 0.95)


if __name__ == "__main__":
    unittest.main()
