"""The learned potentials: the network, its training items and loss, the simulated user, and
the extractor the editor runs it as. Nothing here needs the cache or a GPU."""

import os
import tempfile
import unittest

import numpy as np
import torch

from mrfseg.learned import CLICK_SIGMA, IN_CHANNELS, click_maps, network_input, normalize
from mrfseg.learned.clicks import clicks_to_reach, conditioned_target, dice, next_click, random_clicks, simulate
from mrfseg.learned.data import MAX_CLICKS, PatchConfig
from mrfseg.learned.extractor import Learned, blend_weights, learned_potentials, window_origins
from mrfseg.learned.model import OUT_CHANNELS, ChannelNorm, PotentialNet, UNet3DConfig, load, parameter_count, save
from mrfseg.learned.train import clamps_of, lr_at, segmentation_loss, soft_dice_loss
from mrfseg.features import FeatureVolume
from mrfseg.grid import Grid, image_from_voxels
from mrfseg.mrf import AXES
from mrfseg.settings import MRFSettings

SMALL = UNet3DConfig(base=4, levels=3)


class ModelTest(unittest.TestCase):
    def test_four_potentials_per_voxel_on_any_size(self):
        model = PotentialNet(SMALL).eval()
        for shape in [(16, 16, 16), (20, 24, 28), (9, 10, 11)]:
            with self.subTest(shape), torch.no_grad():
                raw = model(torch.randn(2, IN_CHANNELS, *shape))
                self.assertEqual(tuple(raw.shape), (2, OUT_CHANNELS, *shape))
                self.assertEqual(raw.dtype, torch.float32)
        self.assertGreater(parameter_count(model), 1000)
        self.assertEqual(parameter_count(PotentialNet()), 22_580_100)  # the default network

    def test_starts_as_mostly_background_with_unit_couplings(self):
        model = PotentialNet(SMALL).eval()
        with torch.no_grad():
            raw = model(torch.zeros(1, IN_CHANNELS, 16, 16, 16))
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
            x = torch.randn(1, IN_CHANNELS, 16, 16, 16)
            with torch.no_grad():
                torch.testing.assert_close(loaded(x), model(x))
            extractor = Learned(loaded, extra)
            self.assertEqual((extractor.channels, extractor.unit_vectors, extractor.input, extractor.native_spacing, extractor.window), (OUT_CHANNELS, False, "hu", 0.5, 32))
            self.assertEqual(extractor.setting_defaults, {"sweeps": 12, "coupling": 1.0, "unary_shift": 0.0})


class InputTest(unittest.TestCase):
    def test_the_hu_window(self):
        hu = np.array([-2000.0, -200.0, 100.0, 400.0, 1000.0, 3000.0], dtype=np.float32)
        np.testing.assert_allclose(normalize(hu), [-1.0, -1.0, 0.0, 1.0, 3.0, 3.0])

    def test_click_maps_mark_each_click_in_its_channel(self):
        maps = click_maps(np.array([[4, 5, 6, 1], [0, 11, 13, -1], [0, 0, 0, 0]]), (10, 12, 14))
        self.assertEqual(maps.shape, (2, 10, 12, 14))
        self.assertEqual((float(maps[0, 4, 5, 6]), float(maps[1, 0, 11, 13])), (1.0, 1.0))
        self.assertEqual(float(maps[1, 4, 5, 6]), 0.0)  # the inside click marks only the inside channel; the outside one is out of reach
        self.assertAlmostEqual(float(maps[0, 4, 5, 8]), np.exp(-4 / (2 * CLICK_SIGMA**2)), places=5)  # two voxels away
        self.assertEqual(float(maps[0, 4, 5, 13]), 0.0)  # beyond the reach
        self.assertEqual(float(maps.max()), 1.0)
        # Two clicks of one label: the larger mark wins, they do not add up.
        two = click_maps(np.array([[4, 5, 6, 1], [4, 5, 7, 1]]), (10, 12, 14))
        self.assertEqual(float(two[0].max()), 1.0)
        # A window of a larger grid: the click's position is relative to the window's origin.
        windowed = click_maps(np.array([[14, 15, 16, 1]]), (10, 12, 14), origin=(10, 10, 10))
        self.assertEqual(float(windowed[0, 4, 5, 6]), 1.0)
        # Padding (label 0) and a click outside the window are harmless.
        self.assertEqual(float(click_maps(np.array([[50, 50, 50, 1]]), (10, 12, 14)).max()), 0.0)
        self.assertEqual(float(click_maps(np.zeros((0, 4), dtype=np.int64), (4, 4, 4)).max()), 0.0)

    def test_network_input_stacks_the_image_and_the_maps(self):
        hu = np.full((6, 7, 8), 100.0, dtype=np.float32)
        x = network_input(hu, np.array([[3, 3, 4, 1]]))
        self.assertEqual(x.shape, (IN_CHANNELS, 6, 7, 8))
        self.assertEqual(float(x[0].max()), 0.0)  # 100 HU is the window's centre
        self.assertEqual((float(x[1, 3, 3, 4]), float(x[2].max())), (1.0, 0.0))
        self.assertEqual(float(network_input(hu)[1:].max()), 0.0)  # no clicks: empty maps


class ExtractorTest(unittest.TestCase):
    def test_runs_in_windows_and_its_features_are_the_automatic_potentials(self):
        model = PotentialNet(SMALL).eval()
        extractor = Learned(model, {"args": {"patch": 16}})
        hu = np.random.default_rng(0).normal(100, 200, size=(20, 18, 24)).astype(np.float32)
        steps = []
        raw = extractor(hu, torch.device("cpu"), steps.append)
        self.assertEqual(raw.shape, (OUT_CHANNELS, 20, 18, 24))
        self.assertTrue(np.isfinite(raw).all())
        self.assertEqual(steps[-1], 1.0)
        self.assertGreater(len(steps), 1)  # several windows of 16 over 20 x 18 x 24
        image = image_from_voxels(hu.astype(np.int16), 0.5)
        features = FeatureVolume(raw.astype(np.float16), ct_dims=(24, 18, 20), unit=False, intensity=hu.astype(np.float16))
        settings = MRFSettings(feature_spacing=0.5, device="cpu")
        none = learned_potentials(extractor, features, Grid(image), np.zeros((0, 3)), np.zeros(0), settings, torch.device("cpu"))
        self.assertEqual(none.shape, (20, 18, 24))
        self.assertTrue((none.coupling >= 0).all())
        torch.testing.assert_close(none.unary, torch.from_numpy(raw[0].astype(np.float16).astype(np.float32)))  # no points: the base as kept
        with self.assertRaises(ValueError):
            learned_potentials(extractor, FeatureVolume(raw[:2].astype(np.float16), ct_dims=(24, 18, 20), unit=False), Grid(image), np.zeros((0, 3)), np.zeros(0), settings, torch.device("cpu"))

    def test_points_re_run_the_network_around_them_and_blend_into_the_base(self):
        model = PotentialNet(SMALL).eval()
        extractor = Learned(model, {"args": {"patch": 16}})
        hu = np.random.default_rng(1).normal(100, 200, size=(40, 40, 40)).astype(np.float32)
        base = extractor(hu, torch.device("cpu"))
        clicks = np.array([[20, 20, 20, 1]])
        steps = []
        refined = extractor.refine(hu, base, clicks, torch.device("cpu"), steps.append)
        self.assertEqual(steps, [1.0])  # one window for one click
        self.assertEqual(refined.shape, base.shape)
        self.assertFalse(np.shares_memory(refined, base))
        # At the click the window's output stands; far from it the base does.
        with torch.no_grad():
            x = torch.from_numpy(network_input(hu[12:28, 12:28, 12:28], clicks, (12, 12, 12)))[None]
            window = model(x)[0].numpy()
        np.testing.assert_allclose(refined[:, 20, 20, 20], window[:, 8, 8, 8], atol=1e-5)
        np.testing.assert_array_equal(refined[:, :10], base[:, :10])
        np.testing.assert_array_equal(refined[:, 30:], base[:, 30:])
        # No clicks: a copy of the base.
        np.testing.assert_array_equal(extractor.refine(hu, base, np.zeros((0, 4), dtype=np.int64), torch.device("cpu")), base)
        # Through the potentials, from points in world mm.
        image = image_from_voxels(hu.astype(np.int16), 0.5)
        features = FeatureVolume(base.astype(np.float16), ct_dims=(40, 40, 40), unit=False, intensity=hu.astype(np.float16))
        settings = MRFSettings(feature_spacing=0.5, device="cpu")
        potentials = learned_potentials(extractor, features, Grid(image), np.array([[10.0, 10.0, 10.0]]), np.array([1]), settings, torch.device("cpu"))
        self.assertAlmostEqual(float(potentials.unary[20, 20, 20]), float(window[0, 8, 8, 8]), places=3)

    def test_windows_cover_the_clicks_greedily(self):
        origins = window_origins(np.array([[50, 50, 50], [52, 50, 50], [10, 10, 10], [95, 95, 95]]), (100, 100, 100), 32)
        self.assertEqual(origins, [(34, 34, 34), (0, 0, 0), (68, 68, 68)])  # the second click shares the first's window; the edges clip
        self.assertEqual(window_origins(np.array([[5, 5, 5]]), (10, 10, 10), 32), [(0, 0, 0)])  # a volume smaller than a window

    def test_blend_weights_are_one_in_the_middle_and_zero_at_the_faces(self):
        w = blend_weights((16, 16, 16))
        self.assertEqual(float(w[8, 8, 8]), 1.0)
        self.assertTrue((w[4:12, 4:12, 4:12] == 1.0).all())
        self.assertLess(float(w[0, 8, 8]), 0.2)
        self.assertEqual(float(w[8, 8, 15]), float(w[8, 8, 0]))


class ClicksTest(unittest.TestCase):
    def tube(self):
        lumen = np.zeros((12, 12, 24), dtype=bool)
        lumen[5:7, 5:7, 2:22] = True
        return lumen

    def test_random_clicks_fall_where_they_should(self):
        lumen = self.tube()
        rng = np.random.default_rng(0)
        clicks = random_clicks(lumen, rng, 3, 4, spacing=0.5, near_wall=1.0, near_share=1.0, margin=1, on_lumen_share=0.0)
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
        # With two vessels and one inside click, outside clicks may land on the vessel that got none.
        two = lumen.copy()
        two[1:3, 1:3, 2:22] = True
        on_lumen = random_clicks(two, rng, 1, 6, 0.5, on_lumen_share=1.0)
        inside, outside = on_lumen[on_lumen[:, 3] > 0], on_lumen[on_lumen[:, 3] < 0]
        self.assertTrue(all(two[z, y, x] for z, y, x, _ in outside))  # on lumen...
        clicked_vessel = lumen if lumen[tuple(inside[0, :3])] else two & ~lumen
        self.assertFalse(any(clicked_vessel[z, y, x] for z, y, x, _ in outside))  # ...but never the clicked vessel

    def test_the_target_is_what_the_clicks_select(self):
        lumen = self.tube()
        other = np.zeros_like(lumen)
        other[1:3, 1:3, 2:22] = True
        both = lumen | other
        # No clicks: everything. An inside click: its vessel. An outside click alone: everything but its vessel.
        np.testing.assert_array_equal(conditioned_target(both, np.zeros((0, 4), dtype=np.int64)), both)
        np.testing.assert_array_equal(conditioned_target(both, np.array([[5, 5, 10, 1]])), lumen)
        np.testing.assert_array_equal(conditioned_target(both, np.array([[1, 1, 10, -1]])), lumen)
        np.testing.assert_array_equal(conditioned_target(both, np.array([[5, 5, 10, 1], [1, 1, 10, 1]])), both)
        # An outside click on the clicked vessel does not drop it (it clamps its own voxel only).
        np.testing.assert_array_equal(conditioned_target(both, np.array([[5, 5, 10, 1], [5, 6, 12, -1]])), lumen)
        # An outside click in the background changes nothing; padding is ignored.
        np.testing.assert_array_equal(conditioned_target(both, np.array([[8, 8, 8, -1], [0, 0, 0, 0]])), both)
        # Outside clicks on every vessel, no inside click: nothing.
        self.assertEqual(int(conditioned_target(both, np.array([[5, 5, 10, -1], [1, 1, 10, -1]])).sum()), 0)

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
        x = torch.randn(2, IN_CHANNELS, 16, 16, 16)
        label = torch.zeros((2, 16, 16, 16), dtype=torch.uint8)
        label[:, 6:10, 6:10, :] = 1
        x[:, 0][label.bool()] += 2.0  # the "lumen" is bright
        clicks = torch.zeros((2, MAX_CLICKS, 4), dtype=torch.int64)
        clicks[0, 0] = torch.tensor([8, 8, 8, 1])
        clicks[0, 1] = torch.tensor([1, 1, 1, -1])
        x[:, 1:] = torch.from_numpy(np.stack([click_maps(c.numpy(), (16, 16, 16)) for c in clicks]))
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
