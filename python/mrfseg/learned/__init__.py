"""The learned potentials: a 3D U-Net whose four output channels at every voxel are the CRF's
parameters there -- the unary logit a_i, and the couplings w to the neighbour along +z, +y and
+x -- trained from scratch on CTA volumes with coronary label maps so that mean-field inference
in the CRF it defines (`mrfseg.mrf`), with a few simulated clicks clamped, reproduces the label
map.

The extractor itself (`extractor.Learned`) is one entry in `extractors.EXTRACTORS`; the rest of
this package trains it, on the 0.5 mm cache `prepare.py` writes (the same cache the GP editor's
pseg3d project built: give `--cache` its absolute path), on a machine with a GPU:

    python -m mrfseg.learned.train --cache /path/to/pseg3d/data/casx/cache --out runs/first

`train` draws 96-voxel patches from the cache, simulates a user's clicks on each (a few inside,
in the lumen, and a few outside, mostly near the wall, as clamps), runs the network, then a
fixed number of mean-field sweeps *through which the loss is back-propagated*, and scores the
marginals against the label map (binary cross-entropy plus a soft Dice). A term on the unary's
own sigmoid keeps it a usable classifier on its own. The couplings are `softplus` of the
network's raw outputs, so they are never negative and the energy stays submodular.

The network's input is Hounsfield units on the fixed window below (`normalize`): soft tissue
near 0, contrast-filled lumen at about 0.3-1. HU is an absolute scale, so nothing about the
input depends on what else is in the image -- and nothing inside the network does either: it
normalises per voxel over its channels (`model.ChannelNorm`), never over a patch, so a voxel's
potentials depend on its surroundings alone and the windows a CT is run in agree wherever
they overlap.
"""

from __future__ import annotations

import numpy as np

#: The grid the network is trained and run on, mm, isotropic.
SPACING = 0.5
#: The HU window the network sees, and how it is scaled: (clip(HU, LOW, HIGH) - CENTRE) / SCALE.
HU_LOW, HU_HIGH, HU_CENTRE, HU_SCALE = -200.0, 1000.0, 100.0, 300.0


def normalize(hu: np.ndarray) -> np.ndarray:
    """HU -> the network's input scale, float32."""
    return ((np.clip(hu, HU_LOW, HU_HIGH) - HU_CENTRE) / HU_SCALE).astype(np.float32)
