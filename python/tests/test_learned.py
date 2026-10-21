"""The learned potentials: the network, its training items and loss, the simulated user, and
the extractor the editor runs it as. Nothing here needs the cache or a GPU."""

import os
import tempfile
import unittest

import numpy as np
import torch

from mrfseg.learned import normalize
from mrfseg.learned.clicks import clicks_to_reach, dice, next_click, random_clicks, simulate
from mrfseg.learned.data import MAX_CLICKS, PatchConfig
from mrfseg.learned.extractor import Learned, learned_potentials
from mrfseg.learned.model import OUT_CHANNELS, ChannelNorm, PotentialNet, UNet3DConfig, load, parameter_count, save
from mrfseg.learned.train import clamps_of, lr_at, segmentation_loss, soft_dice_loss
from mrfseg.features import FeatureVolume
from mrfseg.mrf import AXES

SMALL = UNet3DConfig(base=4, levels=3)


class ModelTest(unittest.TestCase):
    def test_four_potentials_per_voxel_on_any_size(self):
        model = PotentialNet(SMALL).eval()
        for shape in [(16, 16, 16), (20, 24, 28), (9, 10, 11)]:
            with self.subTest(shape), torch.no_grad():
                raw = model(torch.randn(2, 1, *shape))
                self.assertEqual(tuple(raw.shape), (2, OUT_CHANNELS, *shape))
                self.assertEqual(raw.dtype, torch.float32)
        self.assertGreater(parameter_count(model), 1000)
        self.assertEqual(parameter_count(PotentialNet()), 22_578_372)  # the default network

    def test_starts_as_mostly_background_with_unit_couplings(self):
        model = PotentialNet(SMALL).eval()
        with torch.no_grad():
            raw = model(torch.zeros(1, 1, 16, 16, 16))
        self.assertLess(float(torch.sigmoid(raw[0, 0]).mean()), 0.2)
        potentials = PotentialNet.potentials(raw[0])
        self.assertAlmostEqual(float(potentials.coupling[:, :-1, :-1, :-1].mean()), PotentialNet.COUPLING_INIT, delta=0.5)

    def test_potentials_are_non_negative_couplings_with_the_last_planes_zeroed(self):
        raw = torch.randn(OUT_CHANNELS, 5, 6, 7) * 3
        potentials = PotentialNet.potentials(raw, unary_shift=1.5, coupling_scale=2.0)
        torch.testing.assert_close(potentials.unary, raw[0] + 1.5)
        self.assertTrue((potentials.coupling >= 0).all())
        inner = potentials.coupling[:, :-1, :-1, :-1]
        torch.testing.assert_close(inner, 2.0 * torch.nn.functional.softplus(raw[1:])[:, :-1, :-1, :-1])
        for d in range(AXES):
            self.assertEqual(float(potentials.coupling[d].narrow(d, raw.shape[1 + d] - 1, 1).abs().max()), 0.0)

    def test_channel_norm_matches_its_reference(self):
        norm = ChannelNorm(6)
        with torch.no_grad():
            norm.weight.copy_(torch.rand(6) + 0.5)
            norm.bias.copy_(torch.randn(6))
        x = torch.randn(2, 6, 3, 4, 5, requires_grad=True)
        y = norm(x)
        torch.testing.assert_close(y, ChannelNorm.reference(x, norm.weight, norm.bias), atol=1e-5, rtol=1e-5)
        y.square().sum().backward()
        x2 = x.detach().clone().requires_grad_(True)
        ChannelNorm.reference(x2, norm.weight.detach(), norm.bias.detach()).square().sum().backward()
        torch.testing.assert_close(x.grad, x2.grad, atol=1e-4, rtol=1e-4)

    def test_save_and_load_round_trip(self):
        model = PotentialNet(SMALL).eval()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "best.pt")
            save(model, path, step=7, args={"patch": 32, "sweeps": 12})
            loaded, extra = load(path)
            self.assertEqual((extra["step"], extra["spacing"], extra["args"]["sweeps"]), (7, 0.5, 12))
            x = torch.randn(1, 1, 16, 16, 16)
            with torch.no_grad():
                torch.testing.assert_close(loaded(x), model(x))
            extractor = Learned(loaded, extra)
            self.assertEqual((extractor.channels, extractor.unit_vectors, extractor.input, extractor.native_spacing, extractor.window), (OUT_CHANNELS, False, "hu", 0.5, 32))
            self.assertEqual(extractor.setting_defaults, {"sweeps": 12, "coupling": 1.0, "unary_shift": 0.0})


class ExtractorTest(unittest.TestCase):
    def test_runs_in_windows_and_its_features_are_the_potentials(self):
        model = PotentialNet(SMALL).eval()
        extractor = Learned(model, {"args": {"patch": 16}})
        hu = np.random.default_rng(0).normal(100, 200, size=(20, 18, 24)).astype(np.float32)
        steps = []
        raw = extractor(hu, torch.device("cpu"), steps.append)
        self.assertEqual(raw.shape, (OUT_CHANNELS, 20, 18, 24))
        self.assertTrue(np.isfinite(raw).all())
        self.assertEqual(steps[-1], 1.0)
        self.assertGreater(len(steps), 1)  # several windows of 16 over 20 x 18 x 24
        features = FeatureVolume(raw.astype(np.float16), ct_dims=(24, 18, 20), unit=False)
        potentials = learned_potentials(features, unary_shift=0.0, coupling_scale=1.0, device=torch.device("cpu"))
        self.assertEqual(potentials.shape, (20, 18, 24))
        self.assertTrue((potentials.coupling >= 0).all())
        with self.assertRaises(ValueError):
            learned_potentials(FeatureVolume(raw[:2].astype(np.float16), ct_dims=(24, 18, 20), unit=False), 0.0, 1.0, torch.device("cpu"))

    def test_the_input_window(self):
        hu = np.array([-2000.0, -200.0, 100.0, 400.0, 1000.0, 3000.0], dtype=np.float32)
        np.testing.assert_allclose(normalize(hu), [-1.0, -1.0, 0.0, 1.0, 3.0, 3.0])


class ClicksTest(unittest.TestCase):
    def tube(self):
        lumen = np.zeros((12, 12, 24), dtype=bool)
        lumen[5:7, 5:7, 2:22] = True
        return lumen

    def test_random_clicks_fall_where_they_should(self):
        lumen = self.tube()
        rng = np.random.default_rng(0)
        clicks = random_clicks(lumen, rng, 3, 4, spacing=0.5, near_wall=1.0, near_share=1.0, margin=1)
        self.assertEqual(clicks.shape, (7, 4))
        inside = clicks[clicks[:, 3] > 0]
        outside = clicks[clicks[:, 3] < 0]
        self.assertEqual((len(inside), len(outside)), (3, 4))
        self.assertTrue(all(lumen[z, y, x] for z, y, x, _ in inside))
        self.assertTrue(all(not lumen[z, y, x] for z, y, x, _ in outside))
        self.assertTrue((clicks[:, :3] >= 1).all() and (clicks[:, :3] <= np.array(lumen.shape) - 2).all())  # the margin
        # Outside clicks near the wall are within near_wall of the lumen (two voxels at 0.5 mm).
        for z, y, x, _ in outside:
            self.assertLessEqual(np.abs(np.argwhere(lumen) - [z, y, x]).max(1).min(), 2)
        # No lumen: no inside clicks, outside ones anywhere.
        empty = random_clicks(np.zeros((8, 8, 8), dtype=bool), rng, 3, 2, 0.5)
        self.assertEqual((len(empty), set(empty[:, 3].tolist())), (2, {-1}))
        self.assertEqual(random_clicks(lumen, rng, 0, 0, 0.5).shape, (0, 4))

    def test_the_simulated_user_clicks_at_the_centre_of_the_largest_error(self):
        target = self.tube()
        prediction = np.zeros_like(target)
        z, y, x, label = next_click(prediction, target)
        self.assertEqual(label, 1)
        self.assertTrue(target[z, y, x])
        self.assertEqual((z, y), (5, 5) if target[5, 5, x] else (6, 6))
        self.assertIsNone(next_click(target, target))
        # A false positive asks for an outside click.
        wrong = target.copy()
        wrong[0:3, 0:3, 0:3] = True
        self.assertEqual(next_click(wrong, target)[3], -1)
        self.assertEqual(dice(target, target), 1.0)
        self.assertEqual(dice(np.zeros_like(target), np.zeros_like(target)), 1.0)
        self.assertAlmostEqual(dice(wrong, target), 2 * target.sum() / (wrong.sum() + target.sum()))

    def test_simulate_runs_a_segmenter_click_by_click(self):
        target = self.tube()
        seen = []

        def segment(clicks):
            seen.append(len(clicks))
            # Label everything within 3 voxels of any inside click.
            prediction = np.zeros_like(target)
            for z, y, x, label in clicks:
                if label > 0:
                    prediction[max(z - 3, 0):z + 4, max(y - 3, 0):y + 4, max(x - 3, 0):x + 4] = True
            return prediction

        result = simulate(segment, target, n_clicks=4, others=np.zeros_like(target), first=(5, 5, 12))
        self.assertEqual(seen, [1, 2, 3, 4])
        self.assertEqual(result["clicks"][0], (5, 5, 12, 1))
        self.assertEqual(len(result["dice"]), 4)
        # The 7^3 block around the first click is mostly a false positive, so the user clicks outside next
        # (which this segmenter ignores), and never makes things worse.
        self.assertEqual(result["clicks"][1][3], -1)
        self.assertGreaterEqual(result["dice"][-1], result["dice"][0])
        self.assertEqual(result["leak"], [0.0] * 4)
        self.assertEqual(clicks_to_reach([0.2, 0.5, 0.9], 0.85), 3)
        self.assertEqual(clicks_to_reach([0.2, 0.5], 0.85), 3)


class LossTest(unittest.TestCase):
    def test_clamps_from_padded_clicks(self):
        clicks = torch.zeros((2, MAX_CLICKS, 4), dtype=torch.int64)
        clicks[0, 0] = torch.tensor([1, 2, 3, 1])
        clicks[0, 1] = torch.tensor([0, 0, 0, -1])
        clicks[1, 0] = torch.tensor([4, 4, 4, -1])
        mask, value = clamps_of(clicks, (5, 5, 5))
        self.assertEqual(int(mask.sum()), 3)
        self.assertEqual((float(value[0, 1, 2, 3]), float(value[0, 0, 0, 0]), float(value[1, 4, 4, 4])), (1.0, 0.0, 0.0))
        self.assertTrue(mask[0, 1, 2, 3] and mask[0, 0, 0, 0] and mask[1, 4, 4, 4])

    def test_soft_dice(self):
        target = torch.zeros((1, 4, 4, 4))
        target[0, 1:3, 1:3, 1:3] = 1
        self.assertLess(float(soft_dice_loss(target, target)), 0.02)
        self.assertGreater(float(soft_dice_loss(1 - target, target)), 0.9)

    def test_the_loss_trains_the_network(self):
        torch.manual_seed(0)
        model = PotentialNet(SMALL)
        x = torch.randn(2, 1, 16, 16, 16)
        label = torch.zeros((2, 16, 16, 16), dtype=torch.uint8)
        label[:, 6:10, 6:10, :] = 1
        x[:, 0][label.bool()] += 2.0  # the "lumen" is bright
        clicks = torch.zeros((2, MAX_CLICKS, 4), dtype=torch.int64)
        clicks[0, 0] = torch.tensor([8, 8, 8, 1])
        clicks[0, 1] = torch.tensor([1, 1, 1, -1])
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        losses = []
        for _ in range(12):
            raw = model(x)
            loss, stats = segmentation_loss(raw, label, clicks, sweeps=3)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss))
        self.assertLess(losses[-1], losses[0])
        self.assertEqual(set(stats), {"bce", "dice_loss", "unary_bce", "dice", "coupling", "lumen"})
        # Checkpointed sweeps give the same loss.
        with torch.no_grad():
            raw = model(x)
        plain, _ = segmentation_loss(raw, label, clicks, sweeps=3)
        checkpointed, _ = segmentation_loss(raw, label, clicks, sweeps=3, checkpoint=True)
        self.assertAlmostEqual(float(plain), float(checkpointed), places=5)

    def test_the_learning_rate_schedule(self):
        self.assertAlmostEqual(lr_at(0, 100, 10, 1.0), 0.1)
        self.assertAlmostEqual(lr_at(9, 100, 10, 1.0), 1.0)
        self.assertAlmostEqual(lr_at(10, 100, 10, 1.0), 1.0)
        self.assertAlmostEqual(lr_at(100, 100, 10, 1.0), 0.1)
        self.assertGreater(lr_at(50, 100, 10, 1.0), lr_at(80, 100, 10, 1.0))

    def test_patch_config_limits_the_clicks(self):
        with self.assertRaises(ValueError):
            PatchConfig(max_inside=6, max_outside=6)
        self.assertEqual(PatchConfig().size, 96)


if __name__ == "__main__":
    unittest.main()
