"""The random walker: a Gaussian Markov random field on the voxel grid, and its exact posterior
given the seeds (Grady, "Random Walks for Image Segmentation", IEEE TPAMI 28(11), 2006,
doi:10.1109/TPAMI.2006.233).

The grid's voxels are the nodes i in V of a graph whose edges E join face neighbours (6 per
voxel), each edge (i, j) with an *affinity* w_ij > 0: how alike the two voxels are, how freely
the label flows between them. Where the affinities come from is the business of `basic.py`
(from the image's contrast) and `learned/` (a network); this module only knows the model and
how to solve it.

The model is the Gaussian MRF

    p(u) ~ exp(-E(u) / 2),    E(u) = sum_{(i,j) in E} w_ij (u_i - u_j)^2,

a real-valued field u that is cheap to vary where the affinity is small and expensive where it
is large. The clicks are *seeds*: observed values u_i = s_i (1 at an inside point, 0 at an
outside one) that condition the field exactly. The posterior of the free voxels F given the
seeds S is Gaussian, and its mean -- the field of least energy with the seeds held -- solves

    L_FF p_F = -L_FS s,

with L the graph Laplacian: L_ii = d_i = sum_j w_ij (the *degree*), L_ij = -w_ij for an edge,
0 otherwise. Voxel by voxel this says p_i = sum_j w_ij p_j / d_i: every free voxel is the
affinity-weighted mean of its neighbours (p is *harmonic*). p_i is also the probability that
a random walker started at i, stepping to neighbour j with probability w_ij / d_i, reaches an
inside seed before an outside one; that is what the probability map shows.

Two things follow from harmonicity, and are what this model was chosen for:

- **Maximum principle.** A harmonic p has no local extremum in F: p_i is a mean of its
  neighbours. So with all w > 0 and an outside seed somewhere, every connected component of
  {p >= t} for any t > 0 contains an inside seed -- a click reaches as far as the affinities
  carry it and nothing lights up on its own. The clicked voxel itself is at p = 1 exactly.
- **Comparison principle.** p is monotone in the seeds: adding an inside seed can only raise p
  everywhere, adding an outside one can only lower it. A correction does what it says.

Along a vessel of cross-section A (voxels) whose edges have affinity w_in, with wall edges of
affinity w_wall to a background that is seeded (`Seeds.with_background`), p decays from a
single click over about sqrt(w_in A / (P w_wall)) voxels, P the perimeter: the field leaks out
through the wall as through a lossy transmission line. The affinities are therefore learned (and
stored) as *log-affinities* z_ij = log w_ij, clamped to +-LOG_AFFINITY_RANGE: a ratio of
exp(2 * 16) between a lumen edge and a wall edge is available, and one of exp(14) already
carries a click a thousand voxels.

The system is symmetric positive definite (every voxel also has an edge of affinity SINK to a
virtual node at 0, so it is so even with no seeds) and is solved with the conjugate gradient
method on the whole grid at once, with matrix-vector products written as stencil operations
over the edges (`Level.apply`: each edge adds w_ij (v_i - v_j) to row i and the opposite to
row j, which keeps the operator exactly symmetric and positive in float32, where a summed
degree would round the weakest edges away). Plain conjugate gradients need about two iterations
per voxel of vessel length -- information crosses one edge per product -- so they are
preconditioned with a multigrid cycle (`Multigrid`): voxels are merged 2 x 2 x 2 into aggregates
whose edges are the summed affinities between them (the Galerkin coarse operator of a
piecewise-constant interpolation, which keeps the 6-stencil form, with the edges to seeds and
the sink becoming a diagonal term), down to a grid solved densely, with a red-black
Gauss-Seidel sweep before and after each coarse correction. A vessel's length is then carried
by the coarse grids and the iteration count hardly depends on it. The solution is exact up to a
tolerance on the correction the cycle still proposes (`RandomWalkerResult.residual`).

For training, `RandomWalkerFunction` differentiates the solution with respect to the affinities
*implicitly*: with g = dLoss/dp on the free voxels and L_FF lambda = g (one more solve),

    dLoss / dw_ij = -(lambda_i - lambda_j) (p_i - p_j),

with lambda = 0 and p = s on the seeds, which covers free-free and free-seed edges alike. No
iteration is unrolled and nothing but p is kept, so the gradient costs one solve, whatever the
grid.

Everything runs in torch on whatever device the tensors are on; the volumes are (z, y, x).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import torch

from .compute import Progress
from .errors import InputError

#: Edges per voxel: to its neighbour in the +z, +y and +x direction.
AXES = 3
#: Log-affinities are clamped to [-LOG_AFFINITY_RANGE, LOG_AFFINITY_RANGE] (`affinities`).
LOG_AFFINITY_RANGE = 16.0
#: Every voxel's affinity to a virtual node held at 0: the least affinity an edge can have.
#: Makes the system positive definite with no seeds, and is negligible beside any real edge.
SINK = math.exp(-LOG_AFFINITY_RANGE)
#: Bytes a voxel costs the solver, about: the affinities (the caller's and the system's), the
#: extra diagonal, the degree, the right-hand side, five conjugate-gradient vectors and three
#: multigrid temporaries in float32, the coarse grids (a seventh of the fine one), and three
#: byte masks.
BYTES_PER_VOXEL = 4 * 16 + 4 * 6 // 7 + 3
#: The most memory the solver may take: 100 M voxels at 0.5 mm is a 512 x 512 x 380 scan.
MAX_GRID_BYTES = 100_000_000 * BYTES_PER_VOXEL
#: Grids this small are solved densely at the bottom of the multigrid hierarchy.
COARSEST = 2048
#: Every this many iterations the residual is recomputed from the solution rather than updated.
RECOMPUTE = 25
PRECONDITIONERS = ("multigrid", "jacobi")


@dataclass(frozen=True)
class SolverOptions:
    tolerance: float = 1e-4
    """Stop once the preconditioner proposes no correction larger than this to any voxel's
    probability: with multigrid an estimate of the error left, with Jacobi the local step
    |r_i| / d_i, r the residual of the system."""
    max_iterations: int = 1000
    """Conjugate-gradient iterations at most; the result says whether that was enough."""
    preconditioner: str = "multigrid"
    """"multigrid" (the default: a few dozen iterations whatever the vessel's length) or
    "jacobi" (the diagonal alone: cheaper per iteration, hundreds of them along a vessel, and
    a local measure of convergence that a long vessel can fool)."""

    def __post_init__(self) -> None:
        if not (0 < self.tolerance < 1):
            raise InputError(f"tolerance must be between 0 and 1, not {self.tolerance!r}")
        if self.max_iterations < 1:
            raise InputError("max_iterations must be at least 1")
        if self.preconditioner not in PRECONDITIONERS:
            raise InputError(f"preconditioner must be one of {', '.join(map(repr, PRECONDITIONERS))}, not {self.preconditioner!r}")


def check_grid(shape: tuple[int, ...]) -> None:
    """An InputError if a grid of `shape` is too large for the solver's memory cap."""
    size = math.prod(shape) * BYTES_PER_VOXEL
    if size > MAX_GRID_BYTES:
        raise InputError(
            f"A {' x '.join(map(str, shape))} grid would take {size / (1 << 30):.1f} GB to solve on, "
            f"more than the {MAX_GRID_BYTES / (1 << 30):.0f} GB allowed: use a coarser resampling spacing"
        )


# ------------------------------------------------------------------ the graph


def check_affinities(w: torch.Tensor, batched: bool = False) -> None:
    """(3, z, y, x), or with `batched` also (B, 3, z, y, x)."""
    if w.dim() == 4 and w.shape[0] == AXES:
        return
    if batched and w.dim() == 5 and w.shape[1] == AXES:
        return
    raise ValueError(f"affinities must be ({'B, ' if batched else ''}3, z, y, x), not {tuple(w.shape)}")


def affinities(log_affinity: torch.Tensor, scale: float = 1.0) -> torch.Tensor:
    """w = exp(clamp(scale * z, -LOG_AFFINITY_RANGE, LOG_AFFINITY_RANGE)) for log-affinities z
    (3, z, y, x) -- or a batch of them, (B, 3, z, y, x) -- `affinities[d]` being the edge from a
    voxel to its neighbour one step along axis d (0: z, 1: y, 2: x); the last plane along that
    axis has no such neighbour and is zeroed. `scale` above 1 sharpens the contrast between
    strong and weak edges, below 1 blurs it. Differentiable in z (flat where the clamp acts)."""
    check_affinities(log_affinity, batched=True)
    w = torch.exp(torch.clamp(log_affinity * scale, -LOG_AFFINITY_RANGE, LOG_AFFINITY_RANGE))
    return zero_last_planes(w)


def zero_last_planes(w: torch.Tensor) -> torch.Tensor:
    """`w` (3, z, y, x), or (B, 3, z, y, x), with the plane along each axis that has no
    neighbour set to 0 (out of place)."""
    masked = w.clone()
    for d in range(AXES):
        axis = w.dim() - AXES + d  # the volume axis d, whatever leads
        n = w.shape[axis]
        masked.select(w.dim() - AXES - 1, d).narrow(axis - 1, n - 1, 1).zero_()
    return masked


def neighbour_field(w: torch.Tensor, v: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """(W v)_i = sum_{j in N(i)} w_ij v_j for `v` (z, y, x) and affinities `w` (3, z, y, x), into `out`."""
    h = torch.zeros_like(v) if out is None else out.zero_()
    for d in range(AXES):
        n = v.shape[d]
        if n < 2:
            continue
        wd = w[d].narrow(d, 0, n - 1)
        # The edge (i, i + e_d) contributes w v_{i+e_d} to h_i and w v_i to h_{i+e_d}.
        h.narrow(d, 0, n - 1).addcmul_(wd, v.narrow(d, 1, n - 1))
        h.narrow(d, 1, n - 1).addcmul_(wd, v.narrow(d, 0, n - 1))
    return h


def degree(w: torch.Tensor) -> torch.Tensor:
    """d_i = sum_j w_ij, (z, y, x): the affinities of the edges at each voxel, summed."""
    total = torch.zeros(w.shape[1:], dtype=w.dtype, device=w.device)
    for d in range(AXES):
        n = total.shape[d]
        if n < 2:
            continue
        wd = w[d].narrow(d, 0, n - 1)
        total.narrow(d, 0, n - 1).add_(wd)
        total.narrow(d, 1, n - 1).add_(wd)
    return total


def mean_log_affinity(w: torch.Tensor) -> torch.Tensor:
    """(z, y, x): the mean log-affinity of the edges at each voxel (its six, fewer at the faces
    of the grid): the map of where the label flows and where it is held back."""
    total = torch.zeros(w.shape[1:], dtype=w.dtype, device=w.device)
    count = torch.zeros_like(total)
    for d in range(AXES):
        n = total.shape[d]
        if n < 2:
            continue
        z = torch.log(w[d].narrow(d, 0, n - 1).clamp_min(SINK))
        total.narrow(d, 0, n - 1).add_(z)
        total.narrow(d, 1, n - 1).add_(z)
        count.narrow(d, 0, n - 1).add_(1)
        count.narrow(d, 1, n - 1).add_(1)
    return total / count.clamp_min_(1)


def edge_differences(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """(a_i - a_j) (b_i - b_j) on every edge (i, j = i + e_d): (3, z, y, x), the last plane
    along each axis 0. With a = b = p this is the energy of each edge per unit affinity."""
    out = torch.zeros((AXES, *a.shape), dtype=a.dtype, device=a.device)
    for d in range(AXES):
        n = a.shape[d]
        if n < 2:
            continue
        da = a.narrow(d, 0, n - 1) - a.narrow(d, 1, n - 1)
        db = b.narrow(d, 0, n - 1) - b.narrow(d, 1, n - 1)
        out[d].narrow(d, 0, n - 1).copy_(da * db)
    return out


def dirichlet_energy(w: torch.Tensor, p: torch.Tensor) -> float:
    """E(p) = sum_{(i,j)} w_ij (p_i - p_j)^2 + SINK sum_i p_i^2: what the solution minimises
    over the free voxels, given the seeds. For tests and diagnostics."""
    return float((w * edge_differences(p, p)).sum() + SINK * (p * p).sum())


# ------------------------------------------------------------------ the seeds


@dataclass
class Seeds:
    """The voxels whose value is known: the clicked ones, and the automatic background."""

    mask: torch.Tensor
    """(z, y, x) bool: True where the value is fixed."""
    value: torch.Tensor
    """(z, y, x) float32: 1 for inside, 0 for outside; only meaningful where `mask`."""
    inside: int = 0
    """How many seeds are inside points."""
    outside: int = 0
    """How many are outside points."""
    background: int = 0
    """How many are automatic background seeds (`with_background`)."""

    @classmethod
    def none(cls, shape: tuple[int, int, int], device: torch.device) -> Seeds:
        return cls(torch.zeros(shape, dtype=torch.bool, device=device), torch.zeros(shape, dtype=torch.float32, device=device))

    @classmethod
    def from_voxels(cls, voxels: np.ndarray, labels: np.ndarray, shape: tuple[int, int, int], device: torch.device) -> Seeds:
        """Seeds at (n, 3) integer (z, y, x) voxels with (n,) labels of +1 (inside) or -1
        (outside). Two clicks on one voxel must agree, or it is an InputError; duplicates are
        merged."""
        voxels = np.asarray(voxels, dtype=np.int64).reshape(-1, 3)
        labels = np.asarray(labels).reshape(-1)
        if len(voxels) != len(labels):
            raise InputError(f"{len(voxels)} voxels but {len(labels)} labels")
        if len(voxels) and ((voxels < 0).any() or (voxels >= np.array(shape)).any()):
            raise InputError("A seed lies outside the grid")
        seen: dict[tuple[int, int, int], int] = {}
        for voxel, label in zip(map(tuple, voxels), labels):
            if seen.setdefault(voxel, int(label)) != int(label):
                raise InputError("An inside and an outside point fall on the same voxel of the grid: move one of them")
        seeds = cls.none(shape, device)
        if seen:
            index = torch.tensor(list(seen.keys()), dtype=torch.int64, device=device).reshape(-1, 3)
            value = torch.tensor([1.0 if label > 0 else 0.0 for label in seen.values()], dtype=torch.float32, device=device)
            seeds.mask[index[:, 0], index[:, 1], index[:, 2]] = True
            seeds.value[index[:, 0], index[:, 1], index[:, 2]] = value
            seeds.inside = int((value > 0.5).sum())
            seeds.outside = len(seen) - seeds.inside
        return seeds

    def with_background(self, intensity: torch.Tensor, threshold: float, erosion: int = 1) -> Seeds:
        """These seeds, plus every voxel whose `intensity` (z, y, x, HU) is below `threshold`
        after `erosion` face-neighbour erosions of that set, as an outside seed -- except where a
        point already is: the user's word stands. Fat and air are below 0 HU and surround the
        coronaries within a few voxels, so this is what drains a click's probability out of
        everything its vessel's wall does not hold in."""
        if erosion < 0:
            raise InputError("seed_erosion must be at least 0")
        below = intensity < threshold
        for _ in range(erosion):
            below = eroded(below)
        new = below & ~self.mask
        seeds = Seeds(self.mask | new, torch.where(new, torch.zeros_like(self.value), self.value), self.inside, self.outside, self.background + int(new.sum()))
        return seeds

    def __len__(self) -> int:
        return self.inside + self.outside + self.background

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.mask.shape)  # type: ignore[return-value]

    def apply(self, p: torch.Tensor) -> torch.Tensor:
        """`p` with the seeded voxels set to their values, in place."""
        return torch.where(self.mask, self.value, p, out=p)

    def inside_voxels(self) -> torch.Tensor:
        """(k, 3) int64: the inside seeds' (z, y, x)."""
        return torch.nonzero(self.mask & (self.value > 0.5))

    def counts(self) -> dict[str, int]:
        return {"inside": self.inside, "outside": self.outside, "background": self.background}


def eroded(mask: torch.Tensor) -> torch.Tensor:
    """`mask` (z, y, x) bool less every True voxel with a False face neighbour; the faces of the
    grid count as whatever the voxel itself is."""
    out = mask.clone()
    for d in range(AXES):
        n = mask.shape[d]
        if n < 2:
            continue
        out.narrow(d, 0, n - 1).logical_and_(mask.narrow(d, 1, n - 1))
        out.narrow(d, 1, n - 1).logical_and_(mask.narrow(d, 0, n - 1))
    return out


# ------------------------------------------------------------------ the system, and its coarse versions


def parity_mask(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    """(z, y, x) bool: True where z + y + x is odd ("black"). No two black (or two red) voxels
    are neighbours, so each colour can be updated at once from the other."""
    z, y, x = (torch.arange(n, device=device) for n in shape)
    return ((z[:, None, None] + y[None, :, None] + x[None, None, :]) % 2).bool()


@dataclass
class Level:
    """The system on one grid: A v = sum_j w_ij (v_i - v_j) + extra_i v_i on the present nodes
    (the identity on the absent ones, where every vector is 0). Level 0 is the random walker's
    own system on the free voxels; the coarser levels are its Galerkin aggregations."""

    w: torch.Tensor
    """(3, z, y, x): the affinities between present nodes; 0 on every edge to an absent one."""
    extra: torch.Tensor
    """(z, y, x): what is on the diagonal besides the degree -- the sink and the edges to
    seeds (level 0), the sum of those over the aggregate (coarser); 0 on absent nodes."""
    free: torch.Tensor
    """(z, y, x) bool: the nodes present (not seeds; on a coarse grid, aggregates with a free voxel)."""
    black: torch.Tensor = field(init=False)
    """(z, y, x) bool: the parity of each node."""
    d: torch.Tensor = field(init=False)
    """(z, y, x): the diagonal, degree plus extra; 1 on absent nodes."""
    dense: torch.Tensor | None = None
    """The Cholesky factor of A on the coarsest grid, which is solved directly."""

    def __post_init__(self) -> None:
        self.black = parity_mask(self.shape, self.w.device)
        d = degree(self.w).add_(self.extra)
        self.d = torch.where(self.free, d, torch.ones_like(d))

    @property
    def shape(self) -> tuple[int, ...]:
        return tuple(self.extra.shape)

    def apply(self, v: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
        """A v for `v` (0 on absent nodes), into `out`, edge by edge: w_ij (v_i - v_j) to row i
        and its negative to row j, then the extra. A sum of positive terms along every edge,
        however the affinities range, so symmetric and positive definite in float32 too."""
        out = torch.mul(self.extra, v, out=out)
        for d in range(AXES):
            n = v.shape[d]
            if n < 2:
                continue
            flow = (v.narrow(d, 0, n - 1) - v.narrow(d, 1, n - 1)).mul_(self.w[d].narrow(d, 0, n - 1))
            out.narrow(d, 0, n - 1).add_(flow)
            out.narrow(d, 1, n - 1).sub_(flow)
        return out

    def smooth(self, x: torch.Tensor, r: torch.Tensor, colours: tuple[bool, ...], h: torch.Tensor) -> torch.Tensor:
        """Gauss-Seidel half-sweeps on A x = r, one per colour in the order given, in place
        (`h` is scratch): x_i <- (r_i + sum_j w_ij x_j) / d_i for the nodes of that colour."""
        for colour in colours:
            neighbour_field(self.w, x, out=h).add_(r).div_(self.d)
            torch.where((self.black == colour) & self.free, h, x, out=x)
        return x


def system(w: torch.Tensor, seeds: Seeds) -> tuple[Level, torch.Tensor]:
    """The random walker's system for affinities `w` and `seeds`: level 0 (the free voxels, with
    the sink and the edges to the seeds as the extra) and its right-hand side b_i = sum over
    seed neighbours j of w_ij s_j -- what the seeds push into the free voxels next to them."""
    free = ~seeds.mask
    seeded = seeds.mask.to(w.dtype)
    extra = neighbour_field(w, seeded).add_(SINK).mul_(free)
    s = torch.where(seeds.mask, seeds.value.to(w.dtype), torch.zeros_like(seeded))
    b = neighbour_field(w, s).mul_(free)
    masked = w.clone()
    for d in range(AXES):
        n = free.shape[d]
        if n < 2:
            continue
        both = free.narrow(d, 0, n - 1) & free.narrow(d, 1, n - 1)
        masked[d].narrow(d, 0, n - 1).mul_(both)
    return Level(masked, extra, free), b


class Multigrid:
    """Aggregation multigrid for a `Level`, as a preconditioner: `__call__(r)` returns one
    V-cycle's approximate solution of A z = r, a symmetric positive definite function of r (the
    smoother is run forwards before the coarse correction and backwards after it, and the
    coarsest grid is solved exactly).

    Each coarse grid merges 2 x 2 x 2 voxels into one node (odd sides padded by an absent
    plane). The coarse system is the Galerkin one, P^T A P, for the piecewise-constant
    interpolation P that gives every free voxel of an aggregate the aggregate's value: its edges
    are the summed affinities of the fine edges between two aggregates, and its diagonal the
    summed fine diagonals less twice the affinities inside each aggregate -- which is again the
    degree of the coarse edges plus the summed extra terms of the fine nodes. So every level has
    the form of the fine one and uses the same stencil code.
    """

    def __init__(self, level: Level) -> None:
        self.levels = [level]
        while math.prod(level.shape) > COARSEST and max(level.shape) > 2:
            level = coarsen(level)
            self.levels.append(level)
        level.dense = dense_factor(level)

    def __call__(self, r: torch.Tensor) -> torch.Tensor:
        return self._cycle(0, r)

    def _cycle(self, k: int, r: torch.Tensor) -> torch.Tensor:
        level = self.levels[k]
        if level.dense is not None:
            return dense_solve(level, r)
        h = torch.empty_like(r)
        # Pre-smooth from zero: red first (reads nothing), then black.
        x = torch.where(~level.black & level.free, r / level.d, torch.zeros_like(r))
        level.smooth(x, r, (True,), h)
        residual = level.apply(x, out=h).neg_().add_(r)
        correction = self._cycle(k + 1, restrict(residual, self.levels[k + 1].shape))
        x.add_(prolong(correction, level.shape).mul_(level.free))
        # Post-smooth in the reverse order, so the whole cycle is symmetric.
        return level.smooth(x, r, (True, False), h)


def coarsen(fine: Level) -> Level:
    """The next coarser level of `fine`, as the class docstring says."""
    shape = fine.shape
    even = tuple(n + n % 2 for n in shape)
    blocks = tuple(n // 2 for n in even)
    free = pad_to(fine.free, even).reshape(blocks[0], 2, blocks[1], 2, blocks[2], 2).any(5).any(3).any(1)
    extra = pad_to(fine.extra, even).reshape(blocks[0], 2, blocks[1], 2, blocks[2], 2).sum((1, 3, 5))
    w = torch.zeros((AXES, *blocks), dtype=fine.w.dtype, device=fine.w.device)
    for d in range(AXES):
        # The edges that leave an aggregate along axis d start at its voxels with an odd coordinate
        # there: (b0, 2, b1, 2, b2, 2) -> the odd half along that axis's pair, summed over the other pairs.
        pair = 1 + 2 * d
        wd = pad_to(fine.w[d], even).reshape(blocks[0], 2, blocks[1], 2, blocks[2], 2).select(pair, 1)
        w[d] = wd.sum(tuple(a if a < pair else a - 1 for a in (1, 3, 5) if a != pair))
    return Level(zero_last_planes(w), extra, free)


def pad_to(v: torch.Tensor, shape: tuple[int, ...]) -> torch.Tensor:
    """`v` grown with zeros (False) at the end of each axis to `shape`."""
    if tuple(v.shape) == tuple(shape):
        return v
    out = torch.zeros(shape, dtype=v.dtype, device=v.device)
    out[tuple(slice(0, n) for n in v.shape)] = v
    return out


def restrict(residual: torch.Tensor, coarse_shape: tuple[int, ...]) -> torch.Tensor:
    """P^T r: each aggregate's value is the sum over its voxels."""
    even = tuple(2 * n for n in coarse_shape)
    r = pad_to(residual, even)
    return r.reshape(coarse_shape[0], 2, coarse_shape[1], 2, coarse_shape[2], 2).sum((1, 3, 5))


def prolong(correction: torch.Tensor, fine_shape: tuple[int, ...]) -> torch.Tensor:
    """P e: every voxel takes its aggregate's value."""
    e = correction
    for axis in range(AXES):
        e = e.repeat_interleave(2, dim=axis)
    return e[tuple(slice(0, n) for n in fine_shape)].contiguous()


def dense_factor(level: Level) -> torch.Tensor:
    """The Cholesky factor of the level's A as a dense matrix over all its nodes (absent ones
    are identity rows), in float64."""
    shape = level.shape
    n = math.prod(shape)
    index = torch.arange(n, device=level.d.device).reshape(shape)
    A = torch.diag(level.d.double().ravel())
    for d in range(AXES):
        m = shape[d]
        if m < 2:
            continue
        i = index.narrow(d, 0, m - 1).ravel()
        j = index.narrow(d, 1, m - 1).ravel()
        wd = level.w[d].narrow(d, 0, m - 1).double().ravel()
        A[i, j] -= wd
        A[j, i] -= wd
    try:
        return torch.linalg.cholesky(A)
    except RuntimeError:
        # Singular to rounding (no seeds, and the sink lost below float32's resolution of the
        # degree): a touch of the identity, harmless in a preconditioner.
        return torch.linalg.cholesky(A + 1e-9 * float(torch.diagonal(A).mean()) * torch.eye(n, dtype=A.dtype, device=A.device))


def dense_solve(level: Level, r: torch.Tensor) -> torch.Tensor:
    assert level.dense is not None
    x = torch.cholesky_solve(r.double().reshape(-1, 1), level.dense)
    return x.reshape(level.shape).to(r.dtype)


# ------------------------------------------------------------------ the solver


@dataclass(frozen=True)
class RandomWalkerResult:
    probability: torch.Tensor
    """(z, y, x) float32: p_i, the posterior mean -- the probability a walker from i reaches an
    inside seed first -- with the seeds at their values."""
    iterations: int
    """Conjugate-gradient iterations made."""
    residual: float
    """The largest correction the preconditioner proposed at the end (`SolverOptions.tolerance`)."""
    converged: bool
    """Whether the residual got below the tolerance within the iterations allowed."""


def random_walker(
    w: torch.Tensor,
    seeds: Seeds,
    options: SolverOptions = SolverOptions(),
    progress: Progress | None = None,
    initial: torch.Tensor | None = None,
) -> RandomWalkerResult:
    """The posterior mean of the field with affinities `w` (3, z, y, x) given `seeds`, by the
    conjugate gradient method from `initial` (the last solution, when the affinities have not
    changed; else 0) until the tolerance or the iteration cap of `options`."""
    check_affinities(w)
    if seeds.shape != tuple(w.shape[1:]):
        raise ValueError(f"seeds are {seeds.shape}, the affinities {tuple(w.shape[1:])}")
    if not bool(seeds.mask.any()):  # nothing pushes the field anywhere: it is 0
        return RandomWalkerResult(torch.zeros(seeds.shape, dtype=w.dtype, device=w.device), 0, 0.0, True)
    level, b = system(w, seeds)
    x0 = None if initial is None else torch.where(level.free, initial.to(w.dtype), torch.zeros_like(b))
    x, iterations, residual = solve(level, b, options, x0, progress, preconditioner_for(options, level))
    x.add_(torch.where(seeds.mask, seeds.value.to(w.dtype), torch.zeros_like(b)))
    return RandomWalkerResult(x, iterations, residual, residual <= options.tolerance)


def preconditioner_for(options: SolverOptions, level: Level) -> Multigrid | None:
    """The options' preconditioner for `level`'s system; None is Jacobi's."""
    return Multigrid(level) if options.preconditioner == "multigrid" and bool(level.free.any()) else None


def solve(
    level: Level,
    b: torch.Tensor,
    options: SolverOptions,
    initial: torch.Tensor | None = None,
    progress: Progress | None = None,
    preconditioner: Multigrid | None = None,
) -> tuple[torch.Tensor, int, float]:
    """A x = b for `level`'s system (b and x are 0 on absent nodes): preconditioned conjugate
    gradients, by `preconditioner` (a multigrid cycle) or the diagonal. Returns the solution,
    the iterations made and the final residual measure."""
    x = torch.zeros_like(b) if initial is None else initial.clone()
    if not bool(level.free.any()):
        return x, 0, 0.0
    tol = options.tolerance

    def precondition(r: torch.Tensor) -> torch.Tensor:
        return preconditioner(r) if preconditioner is not None else r / level.d

    # The residual measure is the largest correction the preconditioner proposes, max |z|: with
    # Jacobi the local step r_i / d_i, with multigrid an estimate of the error itself.
    Ap = torch.empty_like(b)
    r = b - level.apply(x, Ap)
    z = precondition(r)
    residual = float(z.abs().max())
    first = max(residual, tol)
    iterations = 0
    if residual > tol:
        p = z.clone()
        rz = float(torch.dot(r.ravel(), z.ravel()))
    while residual > tol and iterations < options.max_iterations:
        level.apply(p, Ap)
        pAp = float(torch.dot(p.ravel(), Ap.ravel()))
        if not (pAp > 0):  # exact solution reached, or precision exhausted
            break
        alpha = rz / pAp
        x.add_(p, alpha=alpha)
        iterations += 1
        if iterations % RECOMPUTE == 0:  # the recurrence drifts from the true residual in float32
            r = b - level.apply(x, Ap)
        else:
            r.add_(Ap, alpha=-alpha)
        z = precondition(r)
        residual = float(z.abs().max())
        rz_new = float(torch.dot(r.ravel(), z.ravel()))
        if not (rz_new > 0):
            break
        p.mul_(rz_new / rz).add_(z)
        rz = rz_new
        if progress and (iterations % 10 == 0 or residual <= tol):
            # Residuals fall roughly geometrically: log scale from the first to the tolerance.
            progress(min(1.0, math.log(first / max(residual, tol)) / max(math.log(first / tol), 1e-12)) if first > tol else 1.0)
    return x, iterations, residual


def components(probability: torch.Tensor, threshold: float = 0.5) -> int:
    """How many face-connected components {probability >= threshold} has: by the maximum
    principle no more than the inside seeds, when the solve has converged. For diagnostics."""
    from scipy import ndimage

    labels = (probability >= threshold).cpu().numpy()
    _, count = ndimage.label(labels, structure=ndimage.generate_binary_structure(3, 1))
    return int(count)


# ------------------------------------------------------------------ training


class RandomWalkerFunction(torch.autograd.Function):
    """p = random_walker(w, seeds) as an autograd function of `w`, with the implicit gradient of
    the module docstring: one more solve in the backward pass, nothing unrolled."""

    @staticmethod
    def forward(ctx, w: torch.Tensor, seed_mask: torch.Tensor, seed_value: torch.Tensor, options: SolverOptions, stats: list | None) -> torch.Tensor:
        seeds = Seeds(seed_mask, seed_value)
        result = random_walker(w.detach(), seeds, options)
        if stats is not None:
            stats.append(dict(iterations=result.iterations, residual=result.residual, converged=result.converged))
        ctx.save_for_backward(w, result.probability, seed_mask)
        ctx.options = options
        ctx.stats = stats
        return result.probability

    @staticmethod
    def backward(ctx, grad_p: torch.Tensor):
        w, p, seed_mask = ctx.saved_tensors
        w = w.detach()
        level, _ = system(w, Seeds(seed_mask, torch.zeros(seed_mask.shape, dtype=w.dtype, device=w.device)))
        g = torch.where(level.free, grad_p.to(w.dtype), torch.zeros(seed_mask.shape, dtype=w.dtype, device=w.device))
        lam, iterations, residual = solve(level, g.contiguous(), ctx.options, preconditioner=preconditioner_for(ctx.options, level))
        if ctx.stats is not None:
            ctx.stats.append(dict(backward_iterations=iterations, backward_residual=residual))
        grad_w = edge_differences(lam, p).neg_()
        return grad_w, None, None, None, None


def random_walker_differentiable(w: torch.Tensor, seeds: Seeds, options: SolverOptions = SolverOptions(), stats: list | None = None) -> torch.Tensor:
    """The probability (z, y, x) as a differentiable function of the affinities `w`."""
    return RandomWalkerFunction.apply(w, seeds.mask, seeds.value, options, stats)


def random_walker_batched(w: torch.Tensor, seed_mask: torch.Tensor, seed_value: torch.Tensor, options: SolverOptions = SolverOptions(), stats: list | None = None) -> torch.Tensor:
    """The same on a batch: `w` (B, 3, z, y, x), `seed_mask` (B, z, y, x) bool, `seed_value`
    (B, z, y, x) -> (B, z, y, x), each item solved on its own."""
    return torch.stack([random_walker_differentiable(w[i], Seeds(seed_mask[i], seed_value[i]), options, stats) for i in range(w.shape[0])])


# ------------------------------------------------------------------ for tests


def dense_system(w: torch.Tensor, seeds: Seeds) -> tuple[torch.Tensor, torch.Tensor]:
    """The system A x = b on the free voxels of a small grid as dense float64 matrices, so a
    test can solve it directly: A (|F|, |F|) and b (|F|,), the free voxels in C order."""
    shape = tuple(w.shape[1:])
    n = math.prod(shape)
    free = (~seeds.mask).ravel().cpu().numpy()
    index = np.arange(n).reshape(shape)
    L = np.zeros((n, n), dtype=np.float64)
    wd = w.detach().cpu().double().numpy()
    for d in range(AXES):
        for i in np.ndindex(*shape):
            if i[d] + 1 >= shape[d]:
                continue
            j = list(i)
            j[d] += 1
            a, b_ = index[i], index[tuple(j)]
            L[a, b_] -= wd[(d, *i)]
            L[b_, a] -= wd[(d, *i)]
            L[a, a] += wd[(d, *i)]
            L[b_, b_] += wd[(d, *i)]
    L += SINK * np.eye(n)
    s = torch.where(seeds.mask, seeds.value, torch.zeros_like(seeds.value)).ravel().cpu().double().numpy()
    rhs = -(L[np.ix_(free, ~free)] @ s[~free])
    return torch.from_numpy(L[np.ix_(free, free)]), torch.from_numpy(rhs)
