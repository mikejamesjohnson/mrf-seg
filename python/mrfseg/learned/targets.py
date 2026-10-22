"""What the network is trained toward: a target per edge, and the tree a seed selects.

**Edges.** Every edge (i, j) of a patch gets a target t_ij in [0, 1] for sigmoid(z_ij), from the
binary lumen label y and the Hounsfield units I:

- where either voxel is lumen (y_i or y_j), t_ij = [y_i = y_j]: 1 between two lumen voxels,
  0 across the wall. These are the edges that decide what a click selects, and a fraction of a
  percent of all of them, so the loss weighs them as a class of their own;
- between two background voxels, Grady's contrast affinity exp(-(I_i - I_j)^2 / 2 delta^2)
  with delta = BACKGROUND_CONTRAST. The walker needs the background to be a connected conductor
  -- whatever leaks out of a vessel must reach the seeded fat and be drained, and an unclicked
  vessel must be tied to that same ground -- and the contrast affinity is a sensible one for
  that, cut only where tissues meet.

**Trees.** A click selects the lumen it is connected to: `tree_of` is the face-connected
component of the lumen that holds a voxel (the label maps are face-connected, one component per
coronary tree, so in a whole scan this is the tree; in a patch, the part of it the patch holds).
`random_seed` draws the voxel a simulated click lands on, preferring the inside of the vessel
to its partial-volume rim, where a user clicks too.
"""

from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage

from ..rw import AXES

#: HU: delta of the background edges' contrast target.
BACKGROUND_CONTRAST = 100.0
#: The lumen label's connectivity, and the walker's: face neighbours.
SIX = ndimage.generate_binary_structure(3, 1)


def edge_targets(label: torch.Tensor, hu: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """For `label` (B, z, y, x) 0/1 and `hu` (B, z, y, x): the targets (B, 3, z, y, x) float32,
    which edges touch lumen (B, 3, z, y, x) bool, and which edges exist (3, z, y, x) bool -- the
    last plane along each axis has no neighbour. Edges that do not exist have target 0."""
    lumen = label > 0
    shape = lumen.shape[1:]
    targets = torch.zeros((lumen.shape[0], AXES, *shape), dtype=torch.float32, device=label.device)
    touching = torch.zeros_like(targets, dtype=torch.bool)
    exists = torch.zeros((AXES, *shape), dtype=torch.bool, device=label.device)
    scale = -1.0 / (2 * BACKGROUND_CONTRAST**2)
    for d in range(AXES):
        n = shape[d]
        if n < 2:
            continue
        li, lj = lumen.narrow(1 + d, 0, n - 1), lumen.narrow(1 + d, 1, n - 1)
        hi, hj = hu.narrow(1 + d, 0, n - 1), hu.narrow(1 + d, 1, n - 1)
        touch = li | lj
        contrast = torch.exp((hi - hj).square() * scale)
        t = torch.where(touch, (li == lj).float(), contrast.float())
        targets[:, d].narrow(1 + d, 0, n - 1).copy_(t)
        touching[:, d].narrow(1 + d, 0, n - 1).copy_(touch)
        exists[d].narrow(d, 0, n - 1).fill_(True)
    return targets, touching, exists


def tree_of(label: np.ndarray, voxel: tuple[int, int, int]) -> np.ndarray:
    """The face-connected component of the lumen (`label > 0`) that holds `voxel`, as a bool
    volume; all False if the voxel is not lumen."""
    lumen = np.asarray(label) > 0
    if not lumen[voxel]:
        return np.zeros(lumen.shape, dtype=bool)
    components, _ = ndimage.label(lumen, structure=SIX)
    return components == components[voxel]


def random_seed(label: np.ndarray, rng: np.random.Generator) -> tuple[int, int, int] | None:
    """A lumen voxel to seed: one whose face neighbours are all lumen, when there is such a
    voxel (the inside of a vessel, where a user clicks), else any lumen voxel; None without lumen."""
    lumen = np.asarray(label) > 0
    if not lumen.any():
        return None
    deep = ndimage.binary_erosion(lumen, structure=SIX, border_value=1)
    pool = np.argwhere(deep) if deep.any() else np.argwhere(lumen)
    z, y, x = pool[rng.integers(len(pool))]
    return int(z), int(y), int(x)
