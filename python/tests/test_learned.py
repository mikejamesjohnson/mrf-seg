"""The learned affinities: the network, its targets, training items and losses, the simulated
user, and the extractor the editor runs it as. Nothing here needs the cache or a GPU."""

import os
import tempfile
import unittest

import numpy as np
import torch

from mrfseg.features import FeatureVolume
from mrfseg.learned import IN_CHANNELS, network_input, normalize
from mrfseg.learned.clicks import clicks_to_reach, dice, next_click, simulate
from mrfseg.learned.data import PatchConfig, cut, symmetry
from mrfseg.learned.extractor import Learned, learned_log_affinities
from mrfseg.learned.model import OUT_CHANNELS, AffinityNet, ChannelNorm, UNet3DConfig, load, parameter_count, save
from mrfseg.learned.targets import BACKGROUND_CONTRAST, edge_targets, random_seed, tree_of
from mrfseg.learned.train import affinity_loss, lr_at, soft_dice_loss, total_loss, walker_loss
from mrfseg.rw import AXES, SolverOptions, affinities
from mrfseg.settings import MRFSettings

SMALL = UNet3DConfig(base=4, levels=3)
CPU = torch.device("cpu")


class ModelTest(unittest.TestCase):
    def test_three_log_affinities_per_voxel_on_any_size(self):
        model = AffinityNet(SMALL).eval()
        for shape in [(16, 16, 16), (20, 24, 28), (9, 10, 11)]:
            with self.subTest(shape), torch.no_grad():
                z = model(torch.randn(2, IN_CHANNELS, *shape))
                self.assertEqual(tuple(z.shape), (2, OUT_CHANNELS, *shape))
                self.assertEqual(z.dtype, torch.float32)
        self.assertGreater(parameter_count(model), 1000)
        self.assertEqual(parameter_count(AffinityNet()), 22_578_339)  # the default network

    def test_starts_near_a_uniform_graph(self):
        model = AffinityNet(SMALL).eval()
        with torch.no_grad():
            z = model(torch.zeros(1, IN_CHANNELS, 16, 16, 16))
        self.assertLess(float(z.abs().mean()), 1.0)
        w = affinities(z[0])
        self.assertTrue(bool((w[:, :-1, :-1, :-1] > 0).all()))

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
        model = AffinityNet(SMALL).eval()
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "best.pt")
            save(model, path, step=7, args={"patch": 160}, window=32)
            loaded, extra = load(path)
            self.assertEqual((extra["step"], extra["spacing"], extra["window"]), (7, 0.5, 32))
            x = torch.randn(1, IN_CHANNELS, 16, 16, 16)
            with torch.no_grad():
                torch.testing.assert_close(loaded(x), model(x))
            extractor = Learned(loaded, extra)
            self.assertEqual((extractor.channels, extractor.unit_vectors, extractor.input, extractor.native_spacing, extractor.window), (OUT_CHANNELS, False, "hu", 0.5, 32))
            self.assertEqual(extractor.setting_defaults, {"affinity_scale": 1.0})
            self.assertEqual(Learned(loaded, {"args": {"patch": 48}}).window, 48)  # without a window: the patch


class InputTest(unittest.TestCase):
    def test_the_hu_window(self):
        hu = np.array([-2000.0, -200.0, 100.0, 400.0, 1000.0, 3000.0], dtype=np.float32)
        np.testing.assert_allclose(normalize(hu), [-1.0, -1.0, 0.0, 1.0, 3.0, 3.0])
        x = network_input(np.full((6, 7, 8), 100.0, dtype=np.float32))
        self.assertEqual(x.shape, (IN_CHANNELS, 6, 7, 8))
        self.assertEqual(float(np.abs(x).max()), 0.0)  # 100 HU is the window's centre


class TargetsTest(unittest.TestCase):
    def test_edge_targets(self):
        label = torch.zeros((1, 3, 3, 4), dtype=torch.uint8)
        label[0, 1, 1, 1:3] = 1  # two lumen voxels side by side along x
        hu = torch.full((1, 3, 3, 4), 50.0)
        hu[0, 1, 1, 1:3] = 350.0
        hu[0, 0, 0, 0] = 50.0 + BACKGROUND_CONTRAST  # one contrast step in the background
        targets, touching, exists = edge_targets(label, hu)
        self.assertEqual(tuple(targets.shape), (1, AXES, 3, 3, 4))
        self.assertEqual(float(targets[0, 2, 1, 1, 1]), 1.0)   # lumen -- lumen
        self.assertEqual(float(targets[0, 2, 1, 1, 0]), 0.0)   # background -- lumen (the wall)
        self.assertEqual(float(targets[0, 2, 1, 1, 2]), 0.0)   # lumen -- background
        self.assertEqual(float(targets[0, 0, 0, 1, 1]), 0.0)   # the z edge into the lumen from above
        self.assertTrue(bool(touching[0, 2, 1, 1, 1]) and bool(touching[0, 2, 1, 1, 0]) and not bool(touching[0, 2, 0, 0, 0]))
        self.assertAlmostEqual(float(targets[0, 2, 0, 0, 0]), np.exp(-0.5), places=5)  # background, one delta apart
        self.assertEqual(float(targets[0, 2, 2, 2, 0]), 1.0)  # background, no contrast
        self.assertFalse(bool(exists[2, 0, 0, 3]) or bool(exists[0, 2, 0, 0]) or bool(exists[1, 0, 2, 0]))  # the last planes
        self.assertTrue(bool(exists[2, 0, 0, 2]) and bool(exists[0, 1, 0, 0]))
        self.assertEqual(float(targets[0, 2, :, :, 3].abs().max()), 0.0)

    def test_trees_and_seeds(self):
        label = np.zeros((7, 7, 12), dtype=np.uint8)
        label[2:5, 2:5, 1:11] = 1  # a tube, 3 x 3 across
        label[0, 0, 0] = 1         # a lone voxel
        tree = tree_of(label, (3, 3, 5))
        self.assertEqual(int(tree.sum()), 90)
        self.assertFalse(tree[0, 0, 0])
        self.assertEqual(int(tree_of(label, (6, 6, 5)).sum()), 0)  # not lumen
        rng = np.random.default_rng(0)
        seeds = {random_seed(label, rng) for _ in range(30)}
        self.assertTrue(all(label[s] for s in seeds))
        self.assertTrue(all(s[:2] == (3, 3) for s in seeds))  # deep: the tube's axis, never its rim or the lone voxel
        thin = np.zeros((4, 4, 4), dtype=np.uint8)
        thin[1, 1, 1] = 1
        self.assertEqual(random_seed(thin, rng), (1, 1, 1))  # nothing deep: any lumen voxel
        self.assertIsNone(random_seed(np.zeros((4, 4, 4), dtype=np.uint8), rng))


class DataTest(unittest.TestCase):
    def test_cut_pads_and_copies(self):
        volume = np.arange(4 * 5 * 6, dtype=np.int16).reshape(4, 5, 6)
        block = cut(volume, np.array([2, 3, 4]), 4, fill=-7)
        self.assertEqual(block.shape, (4, 4, 4))
        self.assertEqual(int(block[0, 0, 0]), int(volume[2, 3, 4]))
        self.assertEqual(int(block[3, 3, 3]), -7)
        self.assertFalse(np.shares_memory(block, volume))

    def test_symmetry_applies_the_same_transform_to_every_array(self):
        rng = np.random.default_rng(3)
        a = rng.random((5, 6, 7))
        b = (a > 0.5).astype(np.uint8)
        a2, b2 = symmetry([a, b], rng)
        np.testing.assert_array_equal((a2 > 0.5).astype(np.uint8), b2)
        self.assertEqual(sorted(a2.shape), sorted(a.shape))
        self.assertEqual(PatchConfig().size, 96)


class ExtractorTest(unittest.TestCase):
    def test_runs_in_windows_and_its_features_are_the_log_affinities(self):
        model = AffinityNet(SMALL).eval()
        extractor = Learned(model, {"window": 16})
        hu = np.random.default_rng(0).normal(100, 200, size=(20, 18, 24)).astype(np.float32)
        steps = []
        z = extractor(hu, CPU, steps.append)
        self.assertEqual(z.shape, (OUT_CHANNELS, 20, 18, 24))
        self.assertTrue(np.isfinite(z).all())
        self.assertEqual(steps[-1], 1.0)
        self.assertGreater(len(steps), 1)  # several windows of 16 over 20 x 18 x 24
        features = FeatureVolume(z.astype(np.float16), ct_dims=(24, 18, 20), unit=False, intensity=hu.astype(np.float16))
        settings = MRFSettings(feature_extractor="learned", feature_spacing=0.5, device="cpu")
        got = learned_log_affinities(extractor, features, settings, CPU)
        torch.testing.assert_close(got, torch.from_numpy(z.astype(np.float16).astype(np.float32)))
        scaled = learned_log_affinities(extractor, features, MRFSettings(feature_extractor="learned", feature_spacing=0.5, device="cpu", affinity_scale=2.0), CPU)
        torch.testing.assert_close(scaled, 2 * got)
        with self.assertRaises(ValueError):
            learned_log_affinities(extractor, FeatureVolume(z[:2].astype(np.float16), ct_dims=(24, 18, 20), unit=False), settings, CPU)

    def test_windows_agree_where_they_overlap(self):
        """ChannelNorm reads nothing but the voxel, so a voxel's affinities are the same whichever
        window it is in, up to the convolutions' zero padding near a window's faces."""
        model = AffinityNet(SMALL).eval()
        hu = np.random.default_rng(2).normal(100, 200, size=(16, 16, 64)).astype(np.float32)
        with torch.no_grad():
            whole = model(torch.from_numpy(network_input(hu))[None])[0].numpy()
            left = model(torch.from_numpy(network_input(hu[:, :, :48]))[None])[0].numpy()
        np.testing.assert_allclose(whole[:, :, :, :8], left[:, :, :, :8], atol=1e-4)  # beyond the receptive field of the cut face


class ClicksTest(unittest.TestCase):
    def tube(self):
        lumen = np.zeros((12, 12, 24), dtype=bool)
        lumen[5:7, 5:7, 2:22] = True
        return lumen

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


def tube_batch(size: int = 16) -> dict:
    """Two patches with a bright tube along x, one of them with a seed in it."""
    hu = torch.full((2, size, size, size), 50.0)
    label = torch.zeros((2, size, size, size), dtype=torch.uint8)
    label[:, 6:10, 6:10, :] = 1
    hu[label.bool()] = 350.0
    hu[:, :3] = -100.0  # some fat to seed
    hu[:, -3:] = -100.0
    seed = torch.tensor([[8, 8, 8], [-1, -1, -1]])
    tree = label.clone()
    tree[1] = 0
    return dict(image=torch.from_numpy(np.stack([network_input(h.numpy()) for h in hu])), hu=hu, label=label, seed=seed, tree=tree)


class LossTest(unittest.TestCase):
    def test_soft_dice(self):
        target = torch.zeros((1, 4, 4, 4))
        target[0, 1:3, 1:3, 1:3] = 1
        self.assertLess(float(soft_dice_loss(target, target)), 0.02)
        self.assertGreater(float(soft_dice_loss(1 - target, target)), 0.9)

    def test_affinity_loss_is_least_at_the_targets(self):
        batch = tube_batch()
        targets, touching, exists = edge_targets(batch["label"], batch["hu"])
        perfect = torch.logit(targets.clamp(1e-4, 1 - 1e-4))
        loss, stats = affinity_loss(perfect, batch["label"], batch["hu"])
        self.assertLess(stats["lumen_bce"], 0.01)  # hard targets: no entropy floor
        self.assertLess(stats["background_bce"], 0.1)  # soft targets: their own entropy, no more
        self.assertEqual(stats["edge_accuracy"], 1.0)
        worse, _ = affinity_loss(torch.zeros_like(perfect), batch["label"], batch["hu"])
        self.assertGreater(float(worse), float(loss))
        self.assertEqual(set(stats), {"lumen_bce", "background_bce", "edge_accuracy", "lumen_edges"})

    def test_walker_loss_scores_the_seeded_tree(self):
        batch = tube_batch()
        targets, _, _ = edge_targets(batch["label"], batch["hu"])
        good = torch.logit(targets.clamp(1e-4, 1 - 1e-4)) * 2  # confident affinities: the tube is insulated
        loss, stats = walker_loss(good, batch["hu"], batch["label"], batch["seed"], batch["tree"], SolverOptions(1e-4, 500))
        self.assertIsNotNone(loss)
        self.assertGreater(stats["dice"], 0.9)
        self.assertGreater(stats["p_tree"], 0.9)
        self.assertTrue(stats["converged"] == 1.0)
        uniform, stats_uniform = walker_loss(torch.zeros_like(good), batch["hu"], batch["label"], batch["seed"], batch["tree"], SolverOptions(1e-4, 500))
        self.assertGreater(float(uniform), float(loss))
        self.assertLess(stats_uniform["dice"], stats["dice"])
        none, empty = walker_loss(good, batch["hu"], batch["label"], torch.full((2, 3), -1), batch["tree"], SolverOptions())
        self.assertIsNone(none)
        self.assertEqual(empty, {})

    def test_the_losses_train_the_network(self):
        torch.manual_seed(0)
        model = AffinityNet(SMALL)
        batch = tube_batch()
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-2)
        losses = []
        for _ in range(10):
            z = model(batch["image"])
            loss, stats = total_loss(z, batch, CPU, walker_weight=1.0, options=SolverOptions(1e-3, 100))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(float(loss))
        self.assertLess(losses[-1], losses[0])
        self.assertIn("dice", stats)
        self.assertIn("lumen_bce", stats)
        self.assertTrue(all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None))

    def test_the_learning_rate_schedule(self):
        self.assertAlmostEqual(lr_at(0, 100, 10, 1.0), 0.1)
        self.assertAlmostEqual(lr_at(9, 100, 10, 1.0), 1.0)
        self.assertAlmostEqual(lr_at(10, 100, 10, 1.0), 1.0)
        self.assertAlmostEqual(lr_at(100, 100, 10, 1.0), 0.1)
        self.assertGreater(lr_at(50, 100, 10, 1.0), lr_at(80, 100, 10, 1.0))


if __name__ == "__main__":
    unittest.main()
