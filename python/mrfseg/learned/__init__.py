"""The learned potentials: a 3D U-Net whose four output channels at every voxel are the CRF's
parameters there -- the unary logit a_i, and the couplings w to the neighbour along +z, +y and
+x -- trained from scratch on CTA volumes with coronary label maps so that mean-field inference
in the CRF it defines (`mrfseg.mrf`), with the user's clicks clamped, reproduces the label map.

The network sees the clicks. Its input is three channels: the image in Hounsfield units on the
fixed window below (`normalize`: soft tissue near 0, contrast-filled lumen at about 0.3-1), and
two maps of the clicks, inside and outside, each a small Gaussian mark at every clicked voxel
(`click_maps`). What a click means is learned from how it is scored: the target of a patch with
inside clicks is the lumen connected to them, and nothing else -- an inside click *selects the
vessel it is in* -- and a structure with an outside click and no inside click is background;
with no clicks at all the target is every coronary in the patch, the automatic segmentation.
The marks are hints to the network, not potentials: how far a click reaches, and along what, is
the network's to decide, so a click in one of two vessels 2 mm apart need not touch the other.

The extractor itself (`extractor.Learned`) is one entry in `extractors.EXTRACTORS`: at session
open it runs the network once over the whole image with empty click maps, and a Generate
re-runs it only in windows centred on the points and blends the result into that (`Learned.refine`).
The rest of this package trains it, on the 0.5 mm cache `prepare.py` writes (the same cache the
GP editor's pseg3d project built: give `--cache` its absolute path), on a machine with a GPU:

    python -m mrfseg.learned.train --cache /path/to/pseg3d/data/casx/cache --out runs/first

`train` draws 96-voxel patches from the cache, simulates a user's clicks on each (a few inside,
in the lumen, and a few outside, near the wall or on a vessel that is not wanted), conditions
the target on them, runs the network, then a fixed number of mean-field sweeps *through which
the loss is back-propagated*, and scores the marginals against the target (binary cross-entropy
plus a soft Dice). A term on the unary's own sigmoid keeps it a usable classifier on its own.
The couplings are `softplus` of the network's raw outputs, so they are never negative and the
energy stays submodular.

HU is an absolute scale, so nothing about the input depends on what else is in the image -- and
nothing inside the network does either: it normalises per voxel over its channels
(`model.ChannelNorm`), never over a patch, so a voxel's potentials depend on its surroundings
alone and the windows a CT is run in agree wherever they overlap.
"""

from __future__ import annotations

import numpy as np

#: The grid the network is trained and run on, mm, isotropic.
SPACING = 0.5
#: The HU window the network sees, and how it is scaled: (clip(HU, LOW, HIGH) - CENTRE) / SCALE.
HU_LOW, HU_HIGH, HU_CENTRE, HU_SCALE = -200.0, 1000.0, 100.0, 300.0
#: The network's input channels: the image, the inside clicks, the outside clicks.
IN_CHANNELS = 3
#: The width of a click's mark in the click maps, voxels (1 mm at 0.5 mm): wide enough for a
#: convolution to notice, narrow enough to be in the vessel the click is in.
CLICK_SIGMA = 2.0
#: Beyond this many sigma a mark is 0.
CLICK_REACH = 3.0


def normalize(hu: np.ndarray) -> np.ndarray:
    """HU -> the network's input scale, float32."""
    return ((np.clip(hu, HU_LOW, HU_HIGH) - HU_CENTRE) / HU_SCALE).astype(np.float32)


def click_maps(clicks: np.ndarray, shape: tuple[int, int, int], origin: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """The two click channels of a volume of `shape` (z, y, x) whose voxel (0, 0, 0) is at
    `origin` in the clicks' grid: (2, z, y, x) float32, inside clicks in the first, outside in
    the second, each voxel the largest Gaussian mark exp(-d^2 / 2 sigma^2) of any click of that
    label. `clicks` are (n, 4) rows of (z, y, x, label), label +1 inside and -1 outside (0 is
    padding and ignored). Clicks outside the volume still mark it where their reach gets in."""
    maps = np.zeros((2, *shape), dtype=np.float32)
    clicks = np.asarray(clicks).reshape(-1, 4)
    reach = int(np.ceil(CLICK_SIGMA * CLICK_REACH))
    for z, y, x, label in clicks:
        if label == 0:
            continue
        centre = np.array([z, y, x], dtype=np.int64) - np.asarray(origin, dtype=np.int64)
        lo = np.maximum(centre - reach, 0)
        hi = np.minimum(centre + reach + 1, shape)
        if (hi <= lo).any():
            continue
        grids = np.ogrid[lo[0] - centre[0]:hi[0] - centre[0], lo[1] - centre[1]:hi[1] - centre[1], lo[2] - centre[2]:hi[2] - centre[2]]
        mark = np.exp(-(grids[0] ** 2 + grids[1] ** 2 + grids[2] ** 2) / (2 * CLICK_SIGMA**2)).astype(np.float32)
        channel = maps[0 if label > 0 else 1, lo[0]:hi[0], lo[1]:hi[1], lo[2]:hi[2]]
        np.maximum(channel, mark, out=channel)
    return maps


def network_input(hu: np.ndarray, clicks: np.ndarray | None = None, origin: tuple[int, int, int] = (0, 0, 0)) -> np.ndarray:
    """The network's input for a (z, y, x) HU volume: (IN_CHANNELS, z, y, x) float32, the
    normalised image and the click maps (empty without `clicks`)."""
    maps = click_maps(clicks, hu.shape, origin) if clicks is not None and len(clicks) else np.zeros((2, *hu.shape), dtype=np.float32)
    return np.concatenate([normalize(hu)[None], maps], axis=0)
