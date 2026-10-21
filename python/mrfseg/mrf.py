"""The conditional random field (CRF) on the feature grid, and mean-field inference in it.

The grid's voxels are the sites i in V, each with a binary label y_i (1: lumen, 0: not) and the
6-connected neighbourhood (face neighbours). The model is

    p(y | I) = exp(-U(y; I)) / Z(I),
    U(y; I) = -sum_i a_i y_i + sum_{(i,j) in E} w_ij [y_i != y_j],

with a_i the *unary logit* of voxel i (positive favours lumen; with no coupling,
p(y_i = 1) = sigmoid(a_i)) and w_ij >= 0 the *coupling* of edge (i, j): the cost of giving the
two neighbours different labels. Where a_i and w_ij come from is the business of
`handcrafted.py` (from Basic features and the clicks) and `learned/` (a network): this module
only knows the model and how to infer in it.

Clicks are *clamps*: a clicked voxel's label is fixed, and it acts on the rest of the volume
through its edges (the conditional of the unclamped voxels is the same kind of model, with each
clamped neighbour k adding w_ik (2 l_k - 1) to a_i).

What is computed is the marginal p(y_i = 1 | I, clicks), approximated by mean field: the
closest product distribution q(y) = prod_i q_i(y_i), q_i(1) = m_i, in KL(q || p). Its
stationary point satisfies

    m_i = sigmoid(a_i + sum_{j in N(i)} w_ij (2 m_j - 1)),

the exact local conditional with the neighbours' labels replaced by their marginals. The grid's
two parities (checkerboard) are updated in turn: within a parity no two sites are neighbours, so
each half-sweep is an exact block coordinate minimisation of the mean-field free energy
`free_energy`, which therefore never increases, and the iteration converges. Information travels
one voxel per half-sweep, so `sweeps` bounds how far a click can reach through the coupling
alone.

Everything runs in torch on whatever device the tensors are on; the volumes are (z, y, x).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
import torch.utils.checkpoint

from .compute import Progress
from .errors import InputError

#: Edges per voxel: to its neighbour in the +z, +y and +x direction (`Potentials.coupling`).
AXES = 3
#: A sweep whose largest change in any marginal is below this counts as converged.
CONVERGED = 1e-3
#: Bytes a voxel costs the inference, about: the unary, three couplings, the marginal and three
#: temporaries in float32, and two byte masks.
BYTES_PER_VOXEL = 4 * 8 + 2
#: The most memory the inference may take: 100 M voxels at 0.5 mm is a 512 x 512 x 380 scan.
MAX_MRF_BYTES = 100_000_000 * BYTES_PER_VOXEL


@dataclass
class Potentials:
    """The model's parameters on a (z, y, x) grid, as float32 tensors on one device."""

    unary: torch.Tensor
    """(z, y, x): a_i, the unary logit of every voxel."""
    coupling: torch.Tensor
    """(3, z, y, x), non-negative: `coupling[d, ...]` is w between a voxel and its neighbour one
    step along axis d (0: z, 1: y, 2: x), so the last plane along that axis is unused and 0."""

    def __post_init__(self) -> None:
        if self.unary.dim() != 3:
            raise ValueError(f"unary must be (z, y, x), not {tuple(self.unary.shape)}")
        if self.coupling.shape != (AXES, *self.unary.shape):
            raise ValueError(f"coupling must be (3, z, y, x) = {(AXES, *self.unary.shape)}, not {tuple(self.coupling.shape)}")

    @property
    def shape(self) -> tuple[int, int, int]:
        return tuple(self.unary.shape)  # type: ignore[return-value]

    @property
    def device(self) -> torch.device:
        return self.unary.device

    def mean_coupling(self) -> torch.Tensor:
        """(z, y, x): the mean coupling of the edges at each voxel (its six, fewer at the faces of
        the grid): where the model lets the label change cheaply, and where it does not."""
        total = torch.zeros_like(self.unary)
        count = torch.zeros_like(self.unary)
        for d in range(AXES):
            n = self.unary.shape[d]
            if n < 2:
                continue
            w = self.coupling[d].narrow(d, 0, n - 1)
            total.narrow(d, 0, n - 1).add_(w)
            total.narrow(d, 1, n - 1).add_(w)
            count.narrow(d, 0, n - 1).add_(1)
            count.narrow(d, 1, n - 1).add_(1)
        return total / count.clamp_min_(1)


@dataclass
class Clamps:
    """The clicked voxels and their labels: fixed throughout the inference."""

    index: torch.Tensor
    """(n, 3) int64: the (z, y, x) index of each clamped voxel."""
    value: torch.Tensor
    """(n,) float32: 1 for inside, 0 for outside."""

    @classmethod
    def none(cls, device: torch.device) -> Clamps:
        return cls(torch.zeros((0, 3), dtype=torch.int64, device=device), torch.zeros(0, device=device))

    @classmethod
    def from_voxels(cls, voxels: np.ndarray, labels: np.ndarray, shape: tuple[int, int, int], device: torch.device) -> Clamps:
        """Clamps at (n, 3) integer (z, y, x) voxels with (n,) labels of +1 (inside) or -1
        (outside). Two clicks on one voxel must agree, or it is an InputError; duplicates are
        merged."""
        voxels = np.asarray(voxels, dtype=np.int64).reshape(-1, 3)
        labels = np.asarray(labels).reshape(-1)
        if len(voxels) != len(labels):
            raise InputError(f"{len(voxels)} voxels but {len(labels)} labels")
        if len(voxels) and ((voxels < 0).any() or (voxels >= np.array(shape)).any()):
            raise InputError("A clamped voxel lies outside the grid")
        seen: dict[tuple[int, int, int], int] = {}
        for voxel, label in zip(map(tuple, voxels), labels):
            if seen.setdefault(voxel, int(label)) != int(label):
                raise InputError("An inside and an outside point fall on the same voxel of the grid: move one of them")
        index = torch.tensor(list(seen.keys()), dtype=torch.int64, device=device).reshape(-1, 3)
        value = torch.tensor([1.0 if label > 0 else 0.0 for label in seen.values()], dtype=torch.float32, device=device)
        return cls(index, value)

    def __len__(self) -> int:
        return int(self.index.shape[0])

    def apply(self, marginal: torch.Tensor) -> torch.Tensor:
        """`marginal` with the clamped voxels set to their labels, in place."""
        if len(self):
            marginal[self.index[:, 0], self.index[:, 1], self.index[:, 2]] = self.value
        return marginal

    def inside(self) -> torch.Tensor:
        """(k, 3): the clamped voxels labelled inside."""
        return self.index[self.value > 0.5]


@dataclass(frozen=True)
class MeanFieldResult:
    marginal: torch.Tensor
    """(z, y, x) float32: m_i = q(y_i = 1), the approximate p(y_i = 1 | I, clicks)."""
    sweeps: int
    """How many full sweeps (both parities) were made."""
    delta: float
    """The largest change of any marginal in the last sweep: the convergence measure."""

    @property
    def converged(self) -> bool:
        return self.delta < CONVERGED


def check_grid(shape: tuple[int, ...]) -> None:
    """An InputError if a grid of `shape` is too large for the inference's memory cap."""
    size = math.prod(shape) * BYTES_PER_VOXEL
    if size > MAX_MRF_BYTES:
        raise InputError(
            f"A {' x '.join(map(str, shape))} grid would take {size / (1 << 30):.1f} GB to infer in, "
            f"more than the {MAX_MRF_BYTES / (1 << 30):.0f} GB allowed: use a coarser resampling spacing"
        )


def parity_mask(shape: tuple[int, int, int], device: torch.device) -> torch.Tensor:
    """(z, y, x) bool: True where z + y + x is odd. No two True (or two False) voxels are neighbours."""
    z, y, x = (torch.arange(n, device=device) for n in shape)
    return ((z[:, None, None] + y[None, :, None] + x[None, None, :]) % 2).bool()


def neighbour_field(coupling: torch.Tensor, spins: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """h_i = sum_{j in N(i)} w_ij s_j for `spins` s (z, y, x) and `coupling` (3, z, y, x), into `out`."""
    h = torch.zeros_like(spins) if out is None else out.zero_()
    for d in range(AXES):
        n = spins.shape[d]
        if n < 2:
            continue
        w = coupling[d].narrow(d, 0, n - 1)
        # The edge (i, i + e_d) contributes w s_{i+e_d} to h_i and w s_i to h_{i+e_d}.
        h.narrow(d, 0, n - 1).addcmul_(w, spins.narrow(d, 1, n - 1))
        h.narrow(d, 1, n - 1).addcmul_(w, spins.narrow(d, 0, n - 1))
    return h


def mean_field(
    potentials: Potentials,
    clamps: Clamps,
    sweeps: int,
    progress: Progress | None = None,
    initial: torch.Tensor | None = None,
) -> MeanFieldResult:
    """`sweeps` checkerboard mean-field sweeps from `initial` (sigmoid(a), unless given), with
    the clamped voxels held at their labels throughout. Each full sweep updates the even
    parity, then the odd. Stops early once a sweep changes no marginal by `CONVERGED` or more."""
    if sweeps < 0:
        raise InputError("sweeps must be at least 0")
    a, w = potentials.unary, potentials.coupling
    m = torch.sigmoid(a) if initial is None else initial.clone()
    clamps.apply(m)
    odd = parity_mask(potentials.shape, a.device)
    spins = torch.empty_like(m)
    h = torch.empty_like(m)
    delta = math.inf
    done = 0
    for sweep in range(sweeps):
        before = m.clone()
        for parity in (False, True):
            torch.mul(m, 2.0, out=spins).sub_(1.0)
            neighbour_field(w, spins, out=h)
            h.add_(a).sigmoid_()  # the new marginal of every voxel, in h
            torch.where(odd == parity, h, m, out=m)
            clamps.apply(m)
        delta = float((m - before).abs_().max()) if m.numel() else 0.0
        done = sweep + 1
        if progress:
            progress(done / sweeps)
        if delta < CONVERGED:
            break
    if sweeps == 0:
        delta = 0.0
    return MeanFieldResult(m, done, delta)


def mean_field_batched(
    unary: torch.Tensor,
    coupling: torch.Tensor,
    clamp_mask: torch.Tensor,
    clamp_value: torch.Tensor,
    sweeps: int,
    checkpoint: bool = False,
) -> torch.Tensor:
    """The same inference on a batch, written for autograd: `unary` (B, z, y, x), `coupling`
    (B, 3, z, y, x), `clamp_mask` (B, z, y, x) bool with `clamp_value` (B, z, y, x) the labels
    there, `sweeps` full sweeps with no early stop -> the marginals (B, z, y, x), through which a
    loss can be back-propagated to the potentials (and the network that made them). Nothing is
    done in place. With `checkpoint`, each sweep is recomputed in the backward pass instead of
    kept, which trades time for memory: the kept tensors of a sweep are several volumes."""
    odd = parity_mask(tuple(unary.shape[1:]), unary.device)[None]
    m = torch.where(clamp_mask, clamp_value, torch.sigmoid(unary))

    def sweep(m: torch.Tensor, unary: torch.Tensor, coupling: torch.Tensor) -> torch.Tensor:
        for parity in (False, True):
            h = _neighbour_field_batched(coupling, 2 * m - 1)
            m = torch.where(odd == parity, torch.sigmoid(unary + h), m)
            m = torch.where(clamp_mask, clamp_value, m)
        return m

    for _ in range(sweeps):
        if checkpoint and torch.is_grad_enabled():
            m = torch.utils.checkpoint.checkpoint(sweep, m, unary, coupling, use_reentrant=False)
        else:
            m = sweep(m, unary, coupling)
    return m


def _neighbour_field_batched(coupling: torch.Tensor, spins: torch.Tensor) -> torch.Tensor:
    """h (B, z, y, x) for `spins` (B, z, y, x) and `coupling` (B, 3, z, y, x), out of place."""
    parts = []
    for d in range(AXES):
        n = spins.shape[1 + d]
        if n < 2:
            continue
        w = coupling[:, d].narrow(1 + d, 0, n - 1)
        forward = w * spins.narrow(1 + d, 1, n - 1)   # at i: w s_{i+e}
        backward = w * spins.narrow(1 + d, 0, n - 1)  # at i+e: w s_i
        pad_front = [0, 0] * AXES
        pad_back = [0, 0] * AXES
        # F.pad lists the last dimension first: (x_before, x_after, y_before, y_after, z_before, z_after).
        pad_front[2 * (AXES - 1 - d) + 1] = 1
        pad_back[2 * (AXES - 1 - d)] = 1
        parts.append(F.pad(forward, pad_front) + F.pad(backward, pad_back))
    if not parts:
        return torch.zeros_like(spins)
    return sum(parts[1:], parts[0])


def free_energy(potentials: Potentials, marginal: torch.Tensor) -> float:
    """F(q) = E_q[U] - H(q) for the product distribution with marginals `marginal`: what mean
    field minimises; -log Z is a lower bound of it. For tests and diagnostics."""
    m = marginal.clamp(1e-12, 1 - 1e-12)
    energy = -(potentials.unary * m).sum()
    for d in range(AXES):
        n = m.shape[d]
        if n < 2:
            continue
        w = potentials.coupling[d].narrow(d, 0, n - 1)
        mi, mj = m.narrow(d, 0, n - 1), m.narrow(d, 1, n - 1)
        energy = energy + (w * (mi + mj - 2 * mi * mj)).sum()  # E_q[y_i != y_j] = m_i (1 - m_j) + (1 - m_i) m_j
    entropy = -(m * m.log() + (1 - m) * (1 - m).log()).sum()
    return float(energy - entropy)


def energy(potentials: Potentials, labels: torch.Tensor) -> float:
    """U(y) for one labeling `labels` (z, y, x) of 0 and 1. For tests."""
    y = labels.to(potentials.unary.dtype)
    u = -(potentials.unary * y).sum()
    for d in range(AXES):
        n = y.shape[d]
        if n < 2:
            continue
        w = potentials.coupling[d].narrow(d, 0, n - 1)
        u = u + (w * (y.narrow(d, 0, n - 1) != y.narrow(d, 1, n - 1))).sum()
    return float(u)


def resample_to(volume: torch.Tensor, dims: tuple[int, int, int], k: int) -> torch.Tensor:
    """Slice `k` (of `dims[2]`) of a (z, y, x) `volume` resampled trilinearly onto a grid of
    `dims` (x, y, z) spanning the same outermost voxel centres: (ny, nx). This is how a map on
    the feature grid is read on the CT's grid, a CT slice at a time."""
    nx, ny, nz = dims
    fz = volume.shape[0]
    z = k * ((fz - 1) / (nz - 1)) if nz > 1 else 0.0
    z0 = min(int(math.floor(z)), max(fz - 2, 0))
    t = z - z0
    plane = volume[z0]
    if t > 0 and z0 + 1 < fz:
        plane = (1 - t) * plane + t * volume[z0 + 1]
    if plane.shape != (ny, nx):
        plane = F.interpolate(plane[None, None], size=(ny, nx), mode="bilinear", align_corners=True)[0, 0]
    return plane
