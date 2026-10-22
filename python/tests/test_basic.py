"""Basic, the contrast affinities, and choosing an extractor through the settings."""

import dataclasses
import math
import unittest

import numpy as np
import torch

from mrfseg import Basic, InputError, MRFSegmentation, MRFSettings, contrast_log_affinities
from mrfseg.compute import resolve_device
from mrfseg.extractors import features_key
from mrfseg.grid import voxels
from mrfseg.learned.extractor import Learned
from mrfseg.rw import AXES, LOG_AFFINITY_RANGE, affinities
from tests.helpers import StubExtractor, rotation_z, sphere_points, sphere_volume, stub_extractor

CPU = torch.device("cpu")


class BasicTest(unittest.TestCase):
    def test_the_image_is_its_one_feature(self):
        basic = Basic()
        volume = np.random.default_rng(0).normal(100, 200, size=(5, 6, 7)).astype(np.float32)
        features = basic(volume, CPU)
        self.assertEqual((features.shape, features.dtype), ((1, 5, 6, 7), np.float32))
        np.testing.assert_array_equal(features[0], volume)
        self.assertEqual((basic.channels, basic.unit_vectors, basic.input), (1, False, "hu"))

    def test_contrast_log_affinities_are_grady_s(self):
        hu = torch.zeros((2, 3, 4))
        hu[0, 0, 1] = 40.0   # one delta from (0, 0, 0) along x
        hu[0, 1, 0] = 120.0  # three deltas along y
        z = contrast_log_affinities(hu, contrast=40.0)
        self.assertEqual(tuple(z.shape), (AXES, 2, 3, 4))
        self.assertAlmostEqual(float(z[2, 0, 0, 0]), -0.5, places=6)
        self.assertAlmostEqual(float(z[1, 0, 0, 0]), -4.5, places=6)
        self.assertEqual(float(z[0, 0, 0, 0]), 0.0)  # no step along z there
        self.assertEqual(float(z[2, :, :, 3].abs().max()), 0.0)  # the last plane along x has no edge
        w = affinities(z)
        self.assertAlmostEqual(float(w[2, 0, 0, 0]), math.exp(-0.5), places=6)
        self.assertAlmostEqual(float(w[0, 0, 0, 0]), 1.0)
        # A step of 500 HU is clamped at the floor.
        hu[1, 2, 3] = 500.0
        w = affinities(contrast_log_affinities(hu, 40.0))
        self.assertAlmostEqual(float(w[0, 0, 2, 3]) / math.exp(-LOG_AFFINITY_RANGE), 1.0, places=4)
        with self.assertRaises(InputError):
            contrast_log_affinities(hu, 0.0)


class ChoosingTest(unittest.TestCase):
    def test_the_settings_choose_the_extractor(self):
        volume, _ = sphere_volume()
        self.assertIsInstance(MRFSegmentation(volume, MRFSettings()).feature_extractor(), Basic)
        learned = MRFSegmentation(volume, MRFSettings(feature_extractor="learned"))
        self.assertIsInstance(learned.feature_extractor(), Learned)
        self.assertIs(MRFSegmentation(volume, MRFSettings(), stub_extractor).feature_extractor(), stub_extractor)
        with self.assertRaisesRegex(InputError, "feature_extractor must be one of"):
            MRFSettings(feature_extractor="cnn")
        with self.assertRaises(InputError):
            features_key(dataclasses.replace(MRFSettings(), feature_extractor="cnn"))  # a name no entry has, however it got there

    def test_the_features_key_is_what_the_chosen_extractor_reads(self):
        basic = MRFSettings(feature_spacing=1.0, device="cpu", contrast=60.0)
        self.assertEqual(features_key(basic), ("basic", 1.0))
        self.assertEqual(features_key(dataclasses.replace(basic, device="auto", contrast=20.0)), features_key(basic))  # the affinities read these, not the features
        learned = dataclasses.replace(basic, feature_extractor="learned")
        self.assertEqual(features_key(learned)[:3], ("learned", 1.0, "cpu"))  # then the checkpoint's path and mtime
        self.assertEqual(features_key(dataclasses.replace(learned, affinity_scale=2.0)), features_key(learned))

    def test_extracted_again_only_for_what_the_extractor_reads(self):
        volume, _ = sphere_volume()
        counting = StubExtractor()
        calls = counting.shapes
        settings = MRFSettings(feature_spacing=2.0, device="cpu")
        segmentation = MRFSegmentation(volume, settings, counting)
        steps = [
            ({}, 1),
            ({"contrast": 30.0}, 1),  # the affinities' business, not the features'
            ({"device": "auto"}, 1),  # Basic runs on the CPU whatever this says
            ({"feature_spacing": 3.0}, 2),
            ({"feature_extractor": "learned"}, 3),
            ({"feature_extractor": "learned", "affinity_scale": 2.0}, 3),  # not read by the network
            # "auto" is another device only where there is an Apple GPU.
            ({"feature_extractor": "learned", "device": "auto"}, 4 if resolve_device("auto") != resolve_device("cpu") else 3),
        ]
        for change, extracted in steps:
            with self.subTest(**change):
                segmentation.settings = dataclasses.replace(settings, **change)
                segmentation.extract_features()
                self.assertEqual(len(calls), extracted)

    def test_segments_the_sphere_with_basic(self):
        volume, world = sphere_volume(axes=rotation_z(20))
        points, labels = sphere_points(rotation_z(20))
        segmentation = MRFSegmentation(volume, MRFSettings(feature_spacing=2.0, device="cpu"))
        probability = voxels(segmentation.generate_segmentation(points, labels).probability)
        self.assertEqual((segmentation.features.channels, segmentation.features.unit), (1, False))
        truth = np.linalg.norm(world, axis=-1) <= 14.0
        self.assertGreater(((probability > 0.5) == truth).mean(), 0.95)


if __name__ == "__main__":
    unittest.main()
