"""Basic, the untrained affinities (`extractors.py`): Grady's own, from the image's contrast.

    z_ij = -(I_i - I_j)^2 / (2 delta^2),    w_ij = exp(z_ij),

with I in Hounsfield units and `delta` (the setting `contrast`) the step in HU at which an
edge's affinity has fallen to exp(-1/2) = 61 %. Two voxels of the same tissue, a noise's width
apart, keep most of their affinity; a step of three deltas leaves exp(-4.5) = 1 %. This is the
random walker as published (`rw.py`), and the baseline the learned affinities (`learned/`) are
measured against.

It is a weak one for a single click, and knowingly so. The walls of a contrast-filled vessel
are not steps in a CT but ramps a few voxels wide (partial volume and the scanner's blur), and
a ramp of small steps is a chain of edges each with most of its affinity. The lumen then leaks
into the wall and the wall into the seeded fat beyond it, and a point seed -- whose own
conductance is a few affinities' worth -- cannot hold even a large lumen up: the probability
falls off within a few voxels of the click (`rw.py` on the "transmission line"). Making the
affinities tell a ramp inside the lumen from a ramp across its wall, from context, is what the
network is for.

Its "features" are the image itself: one channel, the Hounsfield units on the feature grid,
which the affinities are made from (`FeatureVolume.intensity` carries the same values).
"""

from __future__ import annotations

import numpy as np
import torch

from .compute import Progress
from .errors import InputError
from .rw import AXES


class Basic:
    """A `FeatureExtractor` with the image as its one feature."""

    channels = 1
    unit_vectors = False
    #: Hounsfield units, not a min-max scaled image: the contrast scale is in HU.
    input = "hu"

    def __call__(self, volume: np.ndarray, device: torch.device, progress: Progress | None = None) -> np.ndarray:
        """(1, z, y, x) float32: the volume as it is."""
        return np.asarray(volume, dtype=np.float32)[None]


def contrast_log_affinities(intensity: torch.Tensor, contrast: float) -> torch.Tensor:
    """z on every edge of the grid from `intensity` (z, y, x) in HU: (3, z, y, x) float32,
    `z[d]` for the edge to the neighbour along axis d (see `rw.affinities`); the last plane
    along each axis, which has no such neighbour, is 0."""
    if not contrast > 0:
        raise InputError("contrast must be positive")
    z = torch.zeros((AXES, *intensity.shape), dtype=torch.float32, device=intensity.device)
    scale = -1.0 / (2 * contrast**2)
    for d in range(AXES):
        n = intensity.shape[d]
        if n < 2:
            continue
        step = intensity.narrow(d, 1, n - 1) - intensity.narrow(d, 0, n - 1)
        z[d].narrow(d, 0, n - 1).copy_(step.square_().mul_(scale))
    return z
