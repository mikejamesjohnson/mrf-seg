"""The CRF and its mean-field inference, against brute force on grids small enough to enumerate."""

import itertools
import math
import unittest

import numpy as np
import torch

from mrfseg import InputError
from mrfseg.mrf import (
    AXES,
    CONVERGED,
    Clamps,
    Potentials,
    energy,
    free_energy,
    mean_field,
    mean_field_batched,
    neighbour_field,
    parity_mask,
    resample_to,
)

CPU = torch.device("cpu")


def random_potentials(shape, seed=0, scale=1.5) -> Potentials:
    generator = torch.Generator().manual_seed(seed)
    unary = torch.randn(shape, generator=generator) * scale
    coupling = torch.rand((AXES, *shape), generator=generator) * 1.2
    for d in range(AXES):
        coupling[d].narrow(d, shape[d] - 1, 1).zero_()
    return Potentials(unary, coupling)


def brute_force(potentials: Potentials, clamps: Clamps | None = None) -> tuple[torch.Tensor, float]:
    """The exact marginals and log Z by enumerating every labeling (clamped voxels held)."""
    shape = potentials.shape
    n = math.prod(shape)
    labelings = torch.tensor(list(itertools.product([0, 1], repeat=n)), dtype=torch.float32).reshape(-1, *shape)
    if clamps is not None and len(clamps):
        keep = torch.ones(len(labelings), dtype=torch.bool)
        for (z, y, x), value in zip(clamps.index.tolist(), clamps.value.tolist()):
            keep &= labelings[:, z, y, x] == value
        labelings = labelings[keep]
    energies = torch.tensor([energy(potentials, y) for y in labelings])
    weights = torch.softmax(-energies, 0)
    return (weights[:, None, None, None] * labelings).sum(0), float(torch.logsumexp(-energies, 0))


class PotentialsTest(unittest.TestCase):
    def test_shapes_are_checked(self):
        with self.assertRaises(ValueError):
            Potentials(torch.zeros((2, 3)), torch.zeros((3, 2, 3)))
        with self.assertRaises(ValueError):
            Potentials(torch.zeros((2, 3, 4)), torch.zeros((2, 2, 3, 4)))
        p = random_potentials((2, 3, 4))
        self.assertEqual((p.shape, p.device.type), ((2, 3, 4), "cpu"))

    def test_mean_coupling_averages_the_edges_at_each_voxel(self):
        shape = (2, 2, 2)
        coupling = torch.zeros((AXES, *shape))
        coupling[2, 0, 0, 0] = 1.0  # one edge, along x, between (0,0,0) and (0,0,1)
        mean = Potentials(torch.zeros(shape), coupling).mean_coupling()
        # Each corner voxel of a 2^3 grid has three edges.
        self.assertAlmostEqual(float(mean[0, 0, 0]), 1 / 3)
        self.assertAlmostEqual(float(mean[0, 0, 1]), 1 / 3)
        self.assertEqual(float(mean[1, 1, 1]), 0.0)

    def test_the_energy_is_the_model_s(self):
        p = random_potentials((2, 2, 2))
        y = torch.tensor([[[1, 0], [0, 1]], [[1, 1], [0, 0]]], dtype=torch.float32)
        expected = -(p.unary * y).sum().item()
        for d in range(AXES):
            for index in itertools.product(*(range(n) for n in (2, 2, 2))):
                neighbour = list(index)
                neighbour[d] += 1
                if neighbour[d] < 2 and y[index] != y[tuple(neighbour)]:
                    expected += float(p.coupling[(d, *index)])
        self.assertAlmostEqual(energy(p, y), expected, places=5)


class ClampsTest(unittest.TestCase):
    def test_from_voxels_merges_duplicates_and_refuses_conflicts(self):
        clamps = Clamps.from_voxels(np.array([[0, 0, 0], [0, 0, 0], [1, 2, 3]]), np.array([1, 1, -1]), (2, 3, 4), CPU)
        self.assertEqual(len(clamps), 2)
        self.assertEqual(clamps.value.tolist(), [1.0, 0.0])
        self.assertEqual(clamps.inside().tolist(), [[0, 0, 0]])
        with self.assertRaisesRegex(InputError, "same voxel"):
            Clamps.from_voxels(np.array([[0, 0, 0], [0, 0, 0]]), np.array([1, -1]), (2, 3, 4), CPU)
        with self.assertRaisesRegex(InputError, "outside the grid"):
            Clamps.from_voxels(np.array([[0, 0, 4]]), np.array([1]), (2, 3, 4), CPU)
        with self.assertRaises(InputError):
            Clamps.from_voxels(np.array([[0, 0, 0]]), np.array([1, -1]), (2, 3, 4), CPU)
        self.assertEqual(len(Clamps.none(CPU)), 0)

    def test_apply_holds_the_labels(self):
        clamps = Clamps.from_voxels(np.array([[0, 1, 1], [1, 0, 0]]), np.array([1, -1]), (2, 2, 2), CPU)
        m = clamps.apply(torch.full((2, 2, 2), 0.5))
        self.assertEqual((float(m[0, 1, 1]), float(m[1, 0, 0]), float(m[0, 0, 0])), (1.0, 0.0, 0.5))


class MeanFieldTest(unittest.TestCase):
    def test_the_parity_mask_separates_neighbours(self):
        odd = parity_mask((3, 4, 5), CPU)
        for d in range(AXES):
            n = odd.shape[d]
            self.assertTrue((odd.narrow(d, 0, n - 1) != odd.narrow(d, 1, n - 1)).all())

    def test_the_neighbour_field_sums_the_coupled_spins(self):
        shape = (2, 2, 2)
        coupling = torch.zeros((AXES, *shape))
        coupling[0, 0, 0, 0] = 0.5  # (0,0,0)-(1,0,0)
        coupling[2, 1, 1, 0] = 2.0  # (1,1,0)-(1,1,1)
        spins = torch.tensor([[[1.0, -1.0], [1.0, 1.0]], [[-1.0, 1.0], [1.0, -1.0]]])
        h = neighbour_field(coupling, spins)
        self.assertAlmostEqual(float(h[0, 0, 0]), 0.5 * spins[1, 0, 0])
        self.assertAlmostEqual(float(h[1, 0, 0]), 0.5 * spins[0, 0, 0])
        self.assertAlmostEqual(float(h[1, 1, 0]), 2.0 * spins[1, 1, 1])
        self.assertAlmostEqual(float(h[1, 1, 1]), 2.0 * spins[1, 1, 0])
        self.assertEqual(float(h[0, 1, 1]), 0.0)

    def test_reaches_a_fixed_point_near_the_exact_marginals(self):
        p = random_potentials((2, 3, 3))
        result = mean_field(p, Clamps.none(CPU), sweeps=500)
        m = result.marginal
        self.assertTrue(result.converged)
        self.assertLess(result.sweeps, 500)  # stopped early
        # The fixed point: m_i = sigmoid(a_i + sum_j w_ij (2 m_j - 1)).
        fixed = torch.sigmoid(p.unary + neighbour_field(p.coupling, 2 * m - 1))
        self.assertLess(float((fixed - m).abs().max()), 2 * CONVERGED)
        exact, log_z = brute_force(p)
        self.assertLess(float((m - exact).abs().max()), 0.15)
        # The free energy bounds -log Z from above, and never rises from sweep to sweep.
        self.assertGreaterEqual(free_energy(p, m), -log_z - 1e-4)
        energies = [free_energy(p, mean_field(p, Clamps.none(CPU), sweeps=t).marginal) for t in range(1, 8)]
        for before, after in zip(energies, energies[1:]):
            self.assertLessEqual(after, before + 1e-5)

    def test_clamps_hold_and_condition_the_rest(self):
        p = random_potentials((2, 3, 3), seed=1)
        clamps = Clamps.from_voxels(np.array([[0, 0, 0], [1, 2, 2]]), np.array([1, -1]), p.shape, CPU)
        m = mean_field(p, clamps, sweeps=500).marginal
        self.assertEqual((float(m[0, 0, 0]), float(m[1, 2, 2])), (1.0, 0.0))
        exact, _ = brute_force(p, clamps)
        self.assertLess(float((m - exact).abs().max()), 0.2)  # mean field is an approximation: close, not exact

    def test_no_coupling_is_the_unary_s_sigmoid_and_no_sweeps_is_the_start(self):
        p = random_potentials((3, 3, 3))
        none = Potentials(p.unary, torch.zeros_like(p.coupling))
        result = mean_field(none, Clamps.none(CPU), sweeps=10)
        torch.testing.assert_close(result.marginal, torch.sigmoid(p.unary))
        self.assertEqual(result.sweeps, 1)
        start = mean_field(p, Clamps.none(CPU), sweeps=0)
        torch.testing.assert_close(start.marginal, torch.sigmoid(p.unary))
        self.assertEqual((start.sweeps, start.delta), (0, 0.0))
        with self.assertRaises(InputError):
            mean_field(p, Clamps.none(CPU), sweeps=-1)

    def test_strong_coupling_spreads_a_clamped_label(self):
        shape = (1, 1, 9)
        unary = torch.zeros(shape)
        coupling = torch.zeros((AXES, *shape))
        coupling[2, 0, 0, :-1] = 3.0
        clamps = Clamps.from_voxels(np.array([[0, 0, 0]]), np.array([1]), shape, CPU)
        m = mean_field(Potentials(unary, coupling), clamps, sweeps=100).marginal
        self.assertTrue((m[0, 0] > 0.9).all())
        # Through a cut edge it does not.
        coupling[2, 0, 0, 4] = 0.0
        m = mean_field(Potentials(unary, coupling), clamps, sweeps=100).marginal
        self.assertTrue((m[0, 0, :5] > 0.9).all())
        self.assertTrue((m[0, 0, 5:] - 0.5).abs().max() < 1e-6)

    def test_reports_progress(self):
        steps = []
        mean_field(random_potentials((2, 2, 2)), Clamps.none(CPU), sweeps=3, progress=steps.append)
        self.assertGreaterEqual(len(steps), 1)
        self.assertLessEqual(max(steps), 1.0)

    def test_batched_agrees_with_sequential_and_is_differentiable(self):
        p = random_potentials((3, 4, 5), seed=2)
        clamps = Clamps.from_voxels(np.array([[0, 0, 0], [2, 3, 4]]), np.array([1, -1]), p.shape, CPU)
        sequential = mean_field(p, clamps, sweeps=6)
        mask = torch.zeros(p.shape, dtype=torch.bool)
        value = torch.zeros(p.shape)
        mask[clamps.index[:, 0], clamps.index[:, 1], clamps.index[:, 2]] = True
        value[clamps.index[:, 0], clamps.index[:, 1], clamps.index[:, 2]] = clamps.value
        batched = mean_field_batched(p.unary[None], p.coupling[None], mask[None], value[None], sweeps=sequential.sweeps)
        torch.testing.assert_close(batched[0], sequential.marginal, atol=1e-6, rtol=0)
        # Two items in a batch are two independent problems.
        both = mean_field_batched(torch.stack([p.unary, -p.unary]), torch.stack([p.coupling, p.coupling]), torch.stack([mask, mask]), torch.stack([value, value]), sweeps=4)
        alone = mean_field_batched(-p.unary[None], p.coupling[None], mask[None], value[None], sweeps=4)
        torch.testing.assert_close(both[1], alone[0])
        # Gradients flow to the potentials, the same with and without checkpointing, and not to clamped unaries.
        grads = []
        for checkpoint in (False, True):
            unary = p.unary.clone().requires_grad_(True)
            coupling = p.coupling.clone().requires_grad_(True)
            m = mean_field_batched(unary[None], coupling[None], mask[None], value[None], sweeps=4, checkpoint=checkpoint)
            (m - 0.3).pow(2).mean().backward()
            grads.append((unary.grad.clone(), coupling.grad.clone()))
        self.assertTrue(torch.isfinite(grads[0][0]).all() and torch.isfinite(grads[0][1]).all())
        self.assertGreater(float(grads[0][0].abs().max()), 0)
        torch.testing.assert_close(grads[0][0], grads[1][0])
        torch.testing.assert_close(grads[0][1], grads[1][1])
        self.assertEqual(float(grads[0][0][mask].abs().max()), 0.0)


class ResampleToTest(unittest.TestCase):
    def test_reads_a_feature_grid_volume_on_the_ct_s_grid(self):
        volume = torch.arange(2 * 3 * 4, dtype=torch.float32).reshape(2, 3, 4)
        # The same grid: unchanged.
        for k in range(2):
            torch.testing.assert_close(resample_to(volume, (4, 3, 2), k), volume[k])
        # Twice as many CT slices as feature planes: the middle one is halfway between.
        middle = resample_to(volume, (4, 3, 3), 1)
        torch.testing.assert_close(middle, (volume[0] + volume[1]) / 2)
        # In-plane interpolation keeps the corners.
        plane = resample_to(volume, (7, 5, 2), 0)
        self.assertEqual(tuple(plane.shape), (5, 7))
        self.assertAlmostEqual(float(plane[0, 0]), float(volume[0, 0, 0]))
        self.assertAlmostEqual(float(plane[-1, -1]), float(volume[0, -1, -1]))


if __name__ == "__main__":
    unittest.main()
