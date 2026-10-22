"""The random walker: against a dense solve on grids small enough for one, and its principles."""

import math
import unittest

import numpy as np
import torch

from mrfseg import InputError
from mrfseg.rw import (
    AXES,
    LOG_AFFINITY_RANGE,
    SINK,
    Level,
    Multigrid,
    RandomWalkerFunction,
    Seeds,
    SolverOptions,
    affinities,
    check_grid,
    coarsen,
    components,
    degree,
    dense_system,
    dirichlet_energy,
    edge_differences,
    eroded,
    mean_log_affinity,
    neighbour_field,
    random_walker,
    random_walker_batched,
    random_walker_differentiable,
    system,
)

CPU = torch.device("cpu")
TIGHT = SolverOptions(tolerance=1e-9, max_iterations=5000)


def random_affinities(shape, seed=0, spread=6.0, dtype=torch.float32) -> torch.Tensor:
    """Log-affinities uniform over [-spread, spread]: ratios of exp(2 spread) between edges."""
    generator = torch.Generator().manual_seed(seed)
    z = (torch.rand((AXES, *shape), generator=generator, dtype=dtype) * 2 - 1) * spread
    return affinities(z)


def point_seeds(shape, inside, outside, device=CPU) -> Seeds:
    voxels = np.array(list(inside) + list(outside)).reshape(-1, 3)
    labels = np.array([1] * len(inside) + [-1] * len(outside))
    return Seeds.from_voxels(voxels, labels, shape, device)


def dense_solution(w: torch.Tensor, seeds: Seeds) -> torch.Tensor:
    A, b = dense_system(w, seeds)
    p = torch.where(seeds.mask, seeds.value, torch.zeros_like(seeds.value)).double().ravel()
    p[~seeds.mask.ravel()] = torch.linalg.solve(A, b)
    return p.reshape(seeds.shape)


class GraphTest(unittest.TestCase):
    def test_affinities_clamp_and_zero_the_last_planes(self):
        z = torch.tensor([[[[0.0, 40.0]], [[-40.0, 1.0]]]]).repeat(3, 1, 1, 1)  # (3, 2, 1, 2)
        w = affinities(z)
        self.assertEqual(tuple(w.shape), (3, 2, 1, 2))
        self.assertAlmostEqual(float(w[2, 0, 0, 0]), 1.0)                           # z = 0, an x edge that exists
        self.assertEqual(float(w[2, 0, 0, 1]), 0.0)                                 # the last plane along x
        self.assertEqual(float(w[0, 1, 0, 0]), 0.0)                                 # the last plane along z
        self.assertAlmostEqual(float(w[0, 0, 0, 1]) / math.exp(LOG_AFFINITY_RANGE), 1.0, places=5)   # clamped from 40
        self.assertAlmostEqual(float(w[2, 1, 0, 0]) / math.exp(-LOG_AFFINITY_RANGE), 1.0, places=5)  # clamped from -40
        self.assertAlmostEqual(float(affinities(z, scale=0.5)[2, 1, 0, 0]) / math.exp(-LOG_AFFINITY_RANGE), 1.0, places=5)
        self.assertAlmostEqual(float(affinities(torch.full_like(z, 2.0), scale=0.5)[2, 0, 0, 0]), math.e, places=5)
        with self.assertRaises(ValueError):
            affinities(torch.zeros((2, 3, 3, 3)))

    def test_degree_and_neighbour_field_agree_with_the_dense_laplacian(self):
        shape = (3, 4, 2)
        w = random_affinities(shape, dtype=torch.float64)
        seeds = Seeds.none(shape, CPU)
        A, _ = dense_system(w, seeds)  # every voxel free: the whole Laplacian plus SINK
        v = torch.randn(shape, dtype=torch.float64)
        Av = (degree(w) + SINK) * v - neighbour_field(w, v)
        torch.testing.assert_close(Av.ravel(), A @ v.ravel(), rtol=1e-10, atol=1e-10)
        torch.testing.assert_close(torch.diagonal(A), (degree(w) + SINK).ravel(), rtol=1e-12, atol=1e-12)

    def test_edge_differences_give_the_energy(self):
        shape = (3, 3, 3)
        w = random_affinities(shape, dtype=torch.float64)
        p = torch.rand(shape, dtype=torch.float64)
        energy = float((w * edge_differences(p, p)).sum()) + SINK * float((p * p).sum())
        self.assertAlmostEqual(dirichlet_energy(w, p), energy, places=12)
        # And edge by edge: the x edge from (0, 0, 0) to (0, 0, 1).
        self.assertAlmostEqual(float(edge_differences(p, p)[2, 0, 0, 0]), float((p[0, 0, 0] - p[0, 0, 1]) ** 2), places=12)
        self.assertEqual(float(edge_differences(p, p)[2, 0, 0, 2]), 0.0)

    def test_mean_log_affinity(self):
        w = affinities(torch.full((3, 2, 2, 2), 3.0))
        torch.testing.assert_close(mean_log_affinity(w), torch.full((2, 2, 2), 3.0))

    def test_check_grid(self):
        check_grid((100, 100, 100))
        with self.assertRaises(InputError):
            check_grid((1000, 1000, 1000))


class SeedsTest(unittest.TestCase):
    def test_from_voxels_merges_and_counts(self):
        seeds = Seeds.from_voxels([[0, 0, 0], [0, 0, 0], [1, 1, 1]], [1, 1, -1], (2, 2, 2), CPU)
        self.assertEqual((seeds.inside, seeds.outside, seeds.background, len(seeds)), (1, 1, 0, 2))
        self.assertEqual(int(seeds.mask.sum()), 2)
        self.assertEqual(float(seeds.value[0, 0, 0]), 1.0)
        self.assertEqual(float(seeds.value[1, 1, 1]), 0.0)
        self.assertEqual(seeds.inside_voxels().tolist(), [[0, 0, 0]])
        with self.assertRaises(InputError):
            Seeds.from_voxels([[0, 0, 0], [0, 0, 0]], [1, -1], (2, 2, 2), CPU)
        with self.assertRaises(InputError):
            Seeds.from_voxels([[0, 0, 2]], [1], (2, 2, 2), CPU)
        with self.assertRaises(InputError):
            Seeds.from_voxels([[0, 0, 0]], [1, 1], (2, 2, 2), CPU)

    def test_background_seeds_are_below_the_threshold_eroded_and_never_on_a_point(self):
        hu = torch.full((1, 5, 5), 300.0)
        hu[0, :, :2] = -100.0  # two columns of fat
        hu[0, 2, 2] = -50.0    # one dark voxel in the middle: eroded away
        points = Seeds.from_voxels([[0, 2, 0], [0, 4, 4]], [1, -1], (1, 5, 5), CPU)  # an inside point on fat, an outside one on lumen
        seeds = points.with_background(hu, threshold=0.0, erosion=1)
        expected = torch.zeros((1, 5, 5), dtype=torch.bool)
        expected[0, :, 0] = True    # the outer fat column: the grid's face counts as fat, the inner column borders lumen
        expected[0, 2, 1] = True    # but for the inner voxel next to the dark one, whose neighbours are all dark
        expected[0, 4, 4] = True    # the outside point
        torch.testing.assert_close(seeds.mask, expected)
        self.assertEqual(float(seeds.value[0, 2, 0]), 1.0)  # the point's label stands
        self.assertEqual(seeds.counts(), {"inside": 1, "outside": 1, "background": 5})
        none = points.with_background(hu, threshold=0.0, erosion=0)
        self.assertTrue(bool(none.mask[0, 2, 2]))
        self.assertEqual(none.background, 10)
        self.assertEqual(points.with_background(hu, threshold=-1000.0).background, 0)

    def test_erosion_removes_a_rim(self):
        mask = torch.zeros((5, 5, 5), dtype=torch.bool)
        mask[1:4, 1:4, 1:4] = True
        once = eroded(mask)
        self.assertEqual(int(once.sum()), 1)
        self.assertTrue(bool(once[2, 2, 2]))
        self.assertEqual(int(eroded(once).sum()), 0)
        self.assertEqual(int(eroded(torch.ones((3, 3, 3), dtype=torch.bool)).sum()), 27)  # the faces count as inside


class SolverTest(unittest.TestCase):
    def test_matches_the_dense_solution(self):
        shape = (4, 5, 3)
        w = random_affinities(shape, seed=1, dtype=torch.float64)
        seeds = point_seeds(shape, inside=[(1, 2, 1), (3, 4, 2)], outside=[(0, 0, 0), (2, 0, 2)])
        result = random_walker(w, seeds, TIGHT)
        self.assertTrue(result.converged)
        torch.testing.assert_close(result.probability, dense_solution(w, seeds), rtol=1e-7, atol=1e-7)
        # In float32 too, to the solver's tolerance.
        result32 = random_walker(w.float(), seeds, SolverOptions(tolerance=1e-6, max_iterations=2000))
        self.assertTrue(result32.converged)
        torch.testing.assert_close(result32.probability.double(), dense_solution(w, seeds), rtol=1e-4, atol=1e-4)

    def test_is_harmonic_and_exact_at_the_seeds(self):
        shape = (5, 4, 4)
        w = random_affinities(shape, seed=2, dtype=torch.float64)
        seeds = point_seeds(shape, inside=[(2, 2, 2)], outside=[(0, 0, 0), (4, 3, 3)])
        p = random_walker(w, seeds, TIGHT).probability
        self.assertEqual(float(p[2, 2, 2]), 1.0)
        self.assertEqual(float(p[0, 0, 0]), 0.0)
        mean = neighbour_field(w, p) / (degree(w) + SINK)  # the affinity-weighted mean of the neighbours
        free = ~seeds.mask
        torch.testing.assert_close(p[free], mean[free], rtol=1e-7, atol=1e-7)
        self.assertTrue(bool((p >= 0).all()) and bool((p <= 1).all()))

    def test_maximum_principle(self):
        """Every component of {p >= t} holds an inside seed, for any t > 0."""
        from scipy import ndimage

        shape = (6, 6, 6)
        w = random_affinities(shape, seed=3, spread=8.0)
        seeds = point_seeds(shape, inside=[(1, 1, 1), (4, 4, 4)], outside=[(0, 5, 0), (5, 0, 5), (3, 3, 0)])
        p = random_walker(w, seeds, SolverOptions(tolerance=1e-7, max_iterations=5000)).probability
        inside = seeds.inside_voxels().numpy()
        for t in (0.05, 0.3, 0.5, 0.9):
            labels, count = ndimage.label((p >= t).numpy(), structure=ndimage.generate_binary_structure(3, 1))
            self.assertGreater(count, 0)
            held = set(labels[inside[:, 0], inside[:, 1], inside[:, 2]].tolist()) - {0}
            self.assertEqual(set(range(1, count + 1)), held, f"a component of {{p >= {t}}} holds no inside seed")
        self.assertLessEqual(components(p, 0.5), seeds.inside)

    def test_comparison_principle(self):
        """Another inside seed raises p everywhere; another outside seed lowers it."""
        shape = (5, 5, 5)
        w = random_affinities(shape, seed=4, dtype=torch.float64)
        base = point_seeds(shape, inside=[(2, 2, 2)], outside=[(0, 0, 0)])
        p = random_walker(w, base, TIGHT).probability
        more_in = random_walker(w, point_seeds(shape, inside=[(2, 2, 2), (4, 1, 3)], outside=[(0, 0, 0)]), TIGHT).probability
        more_out = random_walker(w, point_seeds(shape, inside=[(2, 2, 2)], outside=[(0, 0, 0), (4, 1, 3)]), TIGHT).probability
        self.assertTrue(bool((more_in >= p - 1e-9).all()))
        self.assertTrue(bool((more_out <= p + 1e-9).all()))
        self.assertGreater(float((more_in - p).max()), 0.01)
        self.assertLess(float((more_out - p).min()), -0.01)

    def test_solution_minimises_the_energy(self):
        shape = (4, 4, 4)
        w = random_affinities(shape, seed=5, dtype=torch.float64)
        seeds = point_seeds(shape, inside=[(1, 1, 1)], outside=[(3, 3, 3)])
        p = random_walker(w, seeds, TIGHT).probability
        best = dirichlet_energy(w, p)
        generator = torch.Generator().manual_seed(0)
        for _ in range(5):
            other = seeds.apply(p + 0.05 * torch.randn(shape, generator=generator, dtype=torch.float64))
            self.assertGreater(dirichlet_energy(w, other), best)

    def test_no_seeds_gives_zero_and_no_free_voxels_gives_the_seeds(self):
        shape = (3, 3, 3)
        w = random_affinities(shape)
        result = random_walker(w, Seeds.none(shape, CPU))
        self.assertEqual((result.iterations, float(result.probability.abs().max())), (0, 0.0))
        all_seeded = Seeds(torch.ones(shape, dtype=torch.bool), torch.rand(shape))
        result = random_walker(w, all_seeded)
        torch.testing.assert_close(result.probability, all_seeded.value)
        self.assertTrue(result.converged)

    def test_warm_start_and_iteration_cap(self):
        shape = (6, 6, 6)
        w = random_affinities(shape, seed=6)
        seeds = point_seeds(shape, inside=[(3, 3, 3)], outside=[(0, 0, 0)])
        full = random_walker(w, seeds, SolverOptions(tolerance=1e-6, max_iterations=3000))
        self.assertTrue(full.converged)
        again = random_walker(w, seeds, SolverOptions(tolerance=1e-6, max_iterations=3000), initial=full.probability)
        self.assertLessEqual(again.iterations, 1)
        torch.testing.assert_close(again.probability, full.probability, rtol=1e-5, atol=1e-5)
        capped = random_walker(w, seeds, SolverOptions(tolerance=1e-9, max_iterations=2, preconditioner="jacobi"))
        self.assertEqual((capped.iterations, capped.converged), (2, False))
        # A grid small enough to be solved densely is done in one preconditioned iteration.
        self.assertLessEqual(random_walker(w, seeds, SolverOptions(tolerance=1e-6)).iterations, 2)
        with self.assertRaises(InputError):
            SolverOptions(tolerance=0)
        with self.assertRaises(InputError):
            SolverOptions(max_iterations=0)
        with self.assertRaises(InputError):
            SolverOptions(preconditioner="none")

    def test_progress_is_reported(self):
        shape = (8, 8, 8)
        w = random_affinities(shape, seed=7)
        seeds = point_seeds(shape, inside=[(4, 4, 4)], outside=[(0, 0, 0)])
        reports = []
        random_walker(w, seeds, SolverOptions(tolerance=1e-6, max_iterations=3000), progress=reports.append)
        self.assertTrue(reports and reports[-1] == 1.0 and all(0 <= r <= 1 for r in reports))

    def test_a_click_fills_a_bright_tube_and_not_the_background(self):
        """A tube of affinity 1 inside and exp(-10) across its wall in a seeded background: one
        click fills the whole tube at p ~ 1 and leaves the background near 0."""
        shape = (40, 9, 9)
        inside = torch.zeros(shape, dtype=torch.bool)
        inside[:, 3:6, 3:6] = True  # a 3 x 3 tube along z, 40 voxels long
        z = torch.full((AXES, *shape), -10.0)
        for d in range(AXES):
            n = shape[d]
            both = inside.narrow(d, 0, n - 1) & inside.narrow(d, 1, n - 1)
            neither = ~inside.narrow(d, 0, n - 1) & ~inside.narrow(d, 1, n - 1)
            z[d].narrow(d, 0, n - 1)[both] = 0.0
            z[d].narrow(d, 0, n - 1)[neither] = 0.0
        w = affinities(z)
        hu = torch.where(inside, torch.full(shape, 350.0), torch.full(shape, -100.0))
        seeds = Seeds.from_voxels([[0, 4, 4]], [1], shape, CPU).with_background(hu, 0.0, erosion=1)
        result = random_walker(w, seeds, SolverOptions(tolerance=1e-5, max_iterations=2000))
        self.assertTrue(result.converged, result)
        p = result.probability
        self.assertGreater(float(p[inside].min()), 0.95)
        self.assertLess(float(p[~inside].max()), 0.05)
        self.assertEqual(components(p), 1)


class MultigridTest(unittest.TestCase):
    def dense_of(self, level: Level) -> torch.Tensor:
        n = math.prod(level.shape)
        A = torch.zeros(n, n, dtype=torch.float64)
        for k in range(n):
            e = torch.zeros(n, dtype=torch.float64)
            e[k] = 1
            A[:, k] = level.apply(e.reshape(level.shape) * level.free).ravel()
        return A

    def test_level_zero_is_the_dense_system(self):
        shape = (3, 4, 3)
        w = random_affinities(shape, seed=14, dtype=torch.float64)
        seeds = point_seeds(shape, inside=[(1, 1, 1)], outside=[(0, 3, 2)])
        level, b = system(w, seeds)
        A, rhs = dense_system(w, seeds)
        f = (~seeds.mask).ravel()
        torch.testing.assert_close(self.dense_of(level)[f][:, f], A, rtol=1e-12, atol=1e-12)
        torch.testing.assert_close(b.ravel()[f], rhs, rtol=1e-12, atol=1e-12)
        self.assertEqual(float(level.w[0][:-1][seeds.mask[:-1] | seeds.mask[1:]].abs().max()), 0.0)  # no edges to seeds
        torch.testing.assert_close(torch.diagonal(self.dense_of(level))[f], level.d.ravel()[f], rtol=1e-12, atol=1e-12)

    def test_coarse_operator_is_the_galerkin_one(self):
        shape = (5, 4, 3)  # odd sides get padded
        w = random_affinities(shape, seed=12, dtype=torch.float64)
        seeds = point_seeds(shape, inside=[(1, 2, 1)], outside=[(0, 0, 0), (4, 3, 2)])
        free = ~seeds.mask
        fine, _ = system(w, seeds)
        coarse = coarsen(fine)
        self.assertEqual(coarse.shape, (3, 2, 2))
        n0, n1 = math.prod(shape), math.prod(coarse.shape)
        P = torch.zeros(n0, n1, dtype=torch.float64)  # free fine voxel -> its aggregate
        aggregate = torch.arange(n1).reshape(coarse.shape)
        for i in np.ndindex(*shape):
            if free[i]:
                P[np.ravel_multi_index(i, shape), aggregate[tuple(c // 2 for c in i)]] = 1
        galerkin = P.T @ self.dense_of(fine) @ P
        absent = ~coarse.free.ravel()
        galerkin[absent, absent] = 1  # an aggregate of seeds alone is an identity row
        torch.testing.assert_close(self.dense_of(coarse), galerkin, rtol=1e-10, atol=1e-10)
        self.assertTrue(bool(coarse.free[0, 1, 0]) and not bool(coarse.free.ravel()[absent].any()))

    def test_cycle_is_symmetric_positive_definite_and_conditions_the_system(self):
        shape = (9, 7, 6)
        w = random_affinities(shape, seed=13, dtype=torch.float64)
        seeds = point_seeds(shape, inside=[(1, 2, 1), (3, 4, 2)], outside=[(0, 0, 0), (8, 6, 5)])
        free = ~seeds.mask
        import mrfseg.rw as rw

        coarsest = rw.COARSEST
        rw.COARSEST = 8  # several levels on this small grid
        try:
            multigrid = Multigrid(system(w, seeds)[0])
        finally:
            rw.COARSEST = coarsest
        self.assertEqual([level.shape for level in multigrid.levels], [(9, 7, 6), (5, 4, 3), (3, 2, 2), (2, 1, 1)])
        n = math.prod(shape)
        M = torch.zeros(n, n, dtype=torch.float64)
        for k in range(n):
            e = torch.zeros(n, dtype=torch.float64)
            e[k] = 1
            M[:, k] = multigrid(e.reshape(shape) * free).ravel()
        f = free.ravel()
        Mf = M[f][:, f]
        torch.testing.assert_close(Mf, Mf.T, rtol=1e-10, atol=1e-10)
        self.assertGreater(float(torch.linalg.eigvalsh(Mf).min()), 0)
        Af = self.dense_of(multigrid.levels[0])[f][:, f]
        preconditioned = torch.linalg.eigvals(Mf @ Af).real
        jacobi = torch.linalg.eigvals(torch.diag(1 / torch.diagonal(Af)) @ Af).real
        self.assertLess(float(preconditioned.max() / preconditioned.min()), 0.2 * float(jacobi.max() / jacobi.min()))

    def test_long_tube_takes_few_iterations(self):
        """The reason for the multigrid: iterations do not grow with the vessel's length."""
        shape = (160, 7, 7)
        inside = torch.zeros(shape, dtype=torch.bool)
        inside[:, 2:5, 2:5] = True
        z = torch.zeros((AXES, *shape))
        for d in range(AXES):
            n = shape[d]
            one = inside.narrow(d, 0, n - 1) ^ inside.narrow(d, 1, n - 1)
            both = inside.narrow(d, 0, n - 1) & inside.narrow(d, 1, n - 1)
            z[d].narrow(d, 0, n - 1)[one] = -8.0
            z[d].narrow(d, 0, n - 1)[both] = 8.0
        w = affinities(z)
        hu = torch.where(inside, torch.tensor(350.0), torch.tensor(-100.0))
        seeds = Seeds.from_voxels([[0, 3, 3]], [1], shape, CPU).with_background(hu, 0.0, erosion=1)
        multigrid = random_walker(w, seeds, SolverOptions(tolerance=1e-5, max_iterations=2000))
        jacobi = random_walker(w, seeds, SolverOptions(tolerance=1e-6, max_iterations=5000, preconditioner="jacobi"))
        exact = random_walker(w.double(), seeds, SolverOptions(tolerance=1e-10, max_iterations=20000, preconditioner="jacobi")).probability
        self.assertTrue(multigrid.converged and jacobi.converged)
        self.assertLess(multigrid.iterations, 60)
        self.assertGreater(jacobi.iterations, 5 * multigrid.iterations)
        # The multigrid's residual measure is an estimate of the error; Jacobi's local one is
        # fooled by a long vessel, and needs a far tighter tolerance for the same accuracy.
        self.assertLess(float((multigrid.probability.double() - exact).abs().max()), 1e-4)
        self.assertLess(float((jacobi.probability.double() - exact).abs().max()), 1e-3)
        self.assertGreater(float(multigrid.probability[inside].min()), 0.95)


class GradientTest(unittest.TestCase):
    def test_implicit_gradient_matches_finite_differences(self):
        shape = (3, 4, 3)
        z = random_affinities(shape, seed=8, spread=3.0, dtype=torch.float64).log().clamp(-3, 3).requires_grad_()
        seeds = point_seeds(shape, inside=[(1, 1, 1)], outside=[(0, 3, 2), (2, 0, 0)])
        c = torch.randn(shape, dtype=torch.float64, generator=torch.Generator().manual_seed(1))

        def loss_of(z):
            p = random_walker_differentiable(affinities(z), seeds, TIGHT)
            return (c * p).sum()

        loss = loss_of(z)
        (grad,) = torch.autograd.grad(loss, z)
        eps = 1e-6
        for index in [(0, 1, 1, 1), (1, 1, 2, 0), (2, 0, 0, 0), (2, 1, 1, 1), (0, 0, 3, 2)]:
            plus, minus = z.detach().clone(), z.detach().clone()
            plus[index] += eps
            minus[index] -= eps
            numeric = float(loss_of(plus) - loss_of(minus)) / (2 * eps)
            self.assertAlmostEqual(float(grad[index]), numeric, places=5, msg=f"edge {index}")
        # Edges that do not exist (the last planes) and edges between two seeds get no gradient.
        self.assertEqual(float(grad[0, 2].abs().max()), 0.0)
        self.assertEqual(float(grad[1, :, 3].abs().max()), 0.0)

    def test_gradcheck(self):
        shape = (2, 3, 3)
        w = random_affinities(shape, seed=9, spread=2.0, dtype=torch.float64).requires_grad_()
        seeds = point_seeds(shape, inside=[(0, 1, 1)], outside=[(1, 2, 2)])
        self.assertTrue(torch.autograd.gradcheck(lambda w: RandomWalkerFunction.apply(w, seeds.mask, seeds.value, TIGHT, None), (w,), eps=1e-6, atol=1e-5))

    def test_batched_matches_items_and_reports_stats(self):
        shape = (4, 4, 4)
        w = torch.stack([random_affinities(shape, seed=10), random_affinities(shape, seed=11)]).requires_grad_()
        seeds = [point_seeds(shape, inside=[(1, 1, 1)], outside=[(3, 3, 3)]), point_seeds(shape, inside=[(2, 2, 2), (0, 0, 3)], outside=[(0, 0, 0)])]
        mask = torch.stack([s.mask for s in seeds])
        value = torch.stack([s.value for s in seeds])
        stats: list = []
        options = SolverOptions(tolerance=1e-6, max_iterations=2000)
        p = random_walker_batched(w, mask, value, options, stats)
        for i in range(2):
            torch.testing.assert_close(p[i], random_walker(w[i].detach(), seeds[i], options).probability, rtol=1e-5, atol=1e-5)
        self.assertEqual(len(stats), 2)
        self.assertTrue(all(s["converged"] for s in stats))
        p.sum().backward()
        self.assertEqual(len(stats), 4)  # the backward solves report too
        self.assertTrue(torch.isfinite(w.grad).all())


if __name__ == "__main__":
    unittest.main()
