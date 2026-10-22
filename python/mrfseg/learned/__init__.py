"""The learned affinities: a 3D U-Net whose three output channels at every voxel are the
log-affinities z_ij of its edges to the neighbour along +z, +y and +x -- the graph the random
walker (`mrfseg.rw`) is solved on -- trained from scratch on CTA volumes with coronary label
maps, so that a single click in a vessel carries through the whole vessel and no further.

The network sees the image alone: one channel, the Hounsfield units on the fixed window below
(`normalize`: soft tissue near 0, contrast-filled lumen at about 0.3-1). It does not see the
clicks, and need not: a click is an exact observation on the field, not a hint to be learned
(the walker conditions on it exactly). What is learned is which voxels belong together -- the
network "learns how to learn" from a click.

Two stages train it (`train.py`), both on patches of the 0.5 mm cache `prepare.py` writes (the
same cache the GP editor's pseg3d project built: give `--cache` its absolute path), on a machine
with a GPU:

    python -m mrfseg.learned.train --cache /path/to/pseg3d/data/casx/cache --out runs/affinity
    python -m mrfseg.learned.train --cache ... --out runs/walker --init runs/affinity/best.pt --walker-weight 1 --patch 160 --batch 1

1. **Affinities.** Every edge of a patch gets a target (`targets.edge_targets`): where either
   voxel is lumen, 1 if both are and 0 if not; between two background voxels, Grady's contrast
   affinity exp(-(I_i - I_j)^2 / 2 delta^2), which keeps the background a connected conductor
   that drains leaks to the seeded fat. The loss is a binary cross-entropy on sigmoid(z), the
   lumen-touching edges and the rest weighed equally, since the former are a fraction of a
   percent of the edges.
2. **The walker.** With that loss kept, a random lumen voxel of each (larger) patch is seeded
   and the walker is solved, through the implicit gradient (`rw.RandomWalkerFunction`); the
   probability is scored against the lumen connected to the seed with cross-entropy and Dice.
   This is the objective that matters: where the affinities let the probability leak out of a
   vessel, or fail to carry it along one, the gradient says so edge by edge.

HU is an absolute scale, so nothing about the input depends on what else is in the image -- and
nothing inside the network does either: it normalises per voxel over its channels
(`model.ChannelNorm`), never over a patch, so a voxel's affinities depend on its surroundings
alone and the windows a CT is run in agree wherever they overlap (`extractor.Learned`).
"""

from __future__ import annotations

import numpy as np

#: The grid the network is trained and run on, mm, isotropic.
SPACING = 0.5
#: The HU window the network sees, and how it is scaled: (clip(HU, LOW, HIGH) - CENTRE) / SCALE.
HU_LOW, HU_HIGH, HU_CENTRE, HU_SCALE = -200.0, 1000.0, 100.0, 300.0
#: The network's input channels: the image.
IN_CHANNELS = 1


def normalize(hu: np.ndarray) -> np.ndarray:
    """HU -> the network's input scale, float32."""
    return ((np.clip(hu, HU_LOW, HU_HIGH) - HU_CENTRE) / HU_SCALE).astype(np.float32)


def network_input(hu: np.ndarray) -> np.ndarray:
    """The network's input for a (z, y, x) HU volume: (IN_CHANNELS, z, y, x) float32."""
    return normalize(hu)[None]
