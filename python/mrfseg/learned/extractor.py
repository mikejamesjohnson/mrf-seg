"""Learned, the trained network as a `FeatureExtractor` for the editor, and its potentials.

Its "features" are the network's four raw output channels per voxel (`model.OUT_CHANNELS`) with
*no clicks*: the unary logit and the raw couplings of the automatic segmentation, made once per
image at session open, in windows of the training patch size that overlap by half and are
blended with Gaussian weights (`mrfseg.windows`). A Generate with points re-runs the network
only in windows centred on the points, with the click maps as input (`learned.network_input`),
and blends what it gives into a copy of those (`Learned.refine`): within a window the network
decides what the clicks select and deselect; away from every point the automatic potentials
stand. `learned_potentials` then turns the raw channels into `mrf.Potentials` (the couplings
through softplus, scaled by the settings' `coupling`; the unary shifted by `unary_shift`).

Takes the image in Hounsfield units (`input = "hu"`: `extract_features` leaves the HU grid as it
is instead of min-max scaling it) at its training spacing (`native_spacing`). Within a window
the output depends on the window's content only through each voxel's surroundings, so the
windows agree where they overlap except for a rim at their edges, which the weights outvote.

The checkpoint is `$MRFSEG_LEARNED_WEIGHTS`, else `learned.pt` in the mrfseg cache
(`$MRFSEG_CACHE`, default `~/.cache/mrfseg`); it is loaded once per process.
"""

from __future__ import annotations

import math
import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from ..compute import Progress
from ..errors import InputError
from ..features import FeatureVolume
from ..grid import Grid
from ..mrf import Potentials
from ..windows import gaussian_weights, tile_starts
from . import SPACING, network_input
from .model import OUT_CHANNELS, PotentialNet, load

if TYPE_CHECKING:
    from ..settings import MRFSettings

_shared: dict[tuple[str, float], tuple[PotentialNet, dict]] = {}     # (resolved path, mtime) -> model
_shared_lock = threading.Lock()
_run_lock = threading.Lock()


def weights_path() -> Path:
    if os.environ.get("MRFSEG_LEARNED_WEIGHTS"):
        return Path(os.environ["MRFSEG_LEARNED_WEIGHTS"])
    return Path(os.environ.get("MRFSEG_CACHE", Path.home() / ".cache" / "mrfseg")) / "learned.pt"


def load_model(path: Path | None = None) -> tuple[PotentialNet, dict]:
    """The checkpoint at `path` (default `weights_path()`), on the CPU in eval mode, loaded once
    per version of the file: a checkpoint copied in over the old one (a run still training) is
    picked up on the next use, which is also when `features_key` changes."""
    path = path or weights_path()
    if not path.is_file():
        raise InputError(f"No trained network at {path}: train one (python -m mrfseg.learned.train) and point MRFSEG_LEARNED_WEIGHTS at its best.pt")
    key = (str(path.resolve()), os.path.getmtime(path))
    with _shared_lock:
        if key not in _shared:
            for stale in [k for k in _shared if k[0] == key[0]]:    # the file this replaced
                del _shared[stale]
            _shared[key] = load(str(path), "cpu")
        return _shared[key]


class Learned:
    """A `FeatureExtractor`: the trained network's raw potentials, in overlapping windows. Cheap
    to make: the checkpoint is loaded once per process, unless a model is given (for tests)."""

    channels = OUT_CHANNELS
    unit_vectors = False
    #: Hounsfield units, not a min-max scaled image: HU is an absolute scale, and the network
    #: was trained on it (`learned.normalize`).
    input = "hu"
    #: The fraction of a window that neighbouring windows share, and the blending weights'
    #: standard deviation as a fraction of the window (MONAI's and nnU-Net's 1/8).
    overlap = 0.5
    sigma = 0.125

    def __init__(self, model: PotentialNet | None = None, extra: dict | None = None, path: Path | None = None) -> None:
        self._given = (model, extra or {}) if model is not None else None
        self._path = path

    def _loaded(self) -> tuple[PotentialNet, dict]:
        return self._given if self._given is not None else load_model(self._path)

    def model(self) -> PotentialNet:
        return self._loaded()[0]

    @property
    def native_spacing(self) -> float:
        """The grid the network was trained on, mm: the feature grid to use by default. The
        training grid when there is no checkpoint to read it from."""
        try:
            return float(self._loaded()[1].get("spacing", SPACING))
        except InputError:
            return SPACING

    @property
    def window(self) -> int:
        """Voxels a side: the training patch size."""
        return int(self._loaded()[1].get("args", {}).get("patch", 96))

    @property
    def setting_defaults(self) -> dict[str, float]:
        """The `MRFSettings` values to start from with this network: the sweeps it was trained
        through, its couplings as trained (1), no unary shift."""
        args = self._loaded()[1].get("args", {})
        return {"sweeps": int(args.get("sweeps", 20)), "coupling": 1.0, "unary_shift": 0.0}

    def __call__(self, volume: np.ndarray, device: torch.device, progress: Progress | None = None) -> np.ndarray:
        """(4, z, y, x) float32 raw potentials of a (z, y, x) HU volume, with no clicks: the
        automatic segmentation's, in overlapping windows over the whole volume."""
        shape = volume.shape
        window = self.window
        step = max(1, round(window * (1 - self.overlap)))
        tiles = [tile_starts(n, window, step) for n in shape]
        origins = [(z, y, x) for z in tiles[0] for y in tiles[1] for x in tiles[2]]
        out = np.zeros((OUT_CHANNELS, *shape), dtype=np.float32)
        weight = np.zeros(shape, dtype=np.float32)

        def take(origin, size, result):
            region = tuple(slice(o, o + s) for o, s in zip(origin, size))
            w = gaussian_weights(size, self.sigma)
            out[(slice(None), *region)] += result * w
            weight[region] += w

        self._run_windows(volume, None, origins, window, device, take, progress)
        np.divide(out, weight, out=out)
        return out

    def refine(self, volume: np.ndarray, base: np.ndarray, clicks: np.ndarray, device: torch.device, progress: Progress | None = None) -> np.ndarray:
        """`base` (4, z, y, x), the raw potentials with no clicks, re-made around the `clicks`
        ((n, 4) rows of (z, y, x, label) on the volume's grid): the network runs in windows of
        the training patch size centred on the clicks (one window covers the clicks within a
        quarter window of its centre), with the click maps as input, and each window's output
        replaces the base at its centre and fades into it toward its edges. A copy; `base` is
        left alone."""
        clicks = np.asarray(clicks, dtype=np.int64).reshape(-1, 4)
        clicks = clicks[clicks[:, 3] != 0]
        out = np.array(base, dtype=np.float32, copy=True)
        if len(clicks) == 0:
            return out
        shape = volume.shape
        window = self.window
        origins = window_origins(clicks[:, :3], shape, window)

        def take(origin, size, result):
            region = tuple(slice(o, o + s) for o, s in zip(origin, size))
            w = blend_weights(size)
            out[(slice(None), *region)] += w * (result - out[(slice(None), *region)])

        self._run_windows(volume, clicks, origins, window, device, take, progress)
        return out

    def _run_windows(self, volume, clicks, origins, window, device, take, progress) -> None:
        """Run the network on the window of `window` voxels a side at each of `origins` (clipped
        to the volume), with the `clicks` (or none) in the click maps, handing each result to
        `take(origin, size, result)`."""
        model = self.model()
        # A model given to the constructor (training, tests) runs where it is; the shared one is
        # moved to the device asked for, and back, so it is not left on a GPU between images.
        given = self._given is not None
        run_on = next(model.parameters()).device if given else device
        with _run_lock, torch.inference_mode():
            if not given:
                model.to(device)
            try:
                for done, origin in enumerate(origins, 1):
                    size = tuple(min(window, n - o) for o, n in zip(origin, volume.shape))
                    region = tuple(slice(o, o + s) for o, s in zip(origin, size))
                    block = np.asarray(volume[region], dtype=np.float32)
                    x = torch.from_numpy(network_input(block, clicks, origin))[None].to(run_on)
                    result = model(x)[0].float().cpu().numpy()
                    take(origin, size, result)
                    if progress:
                        progress(done / len(origins))
            finally:
                if not given:
                    model.to("cpu")


def window_origins(voxels: np.ndarray, shape: tuple[int, int, int], window: int) -> list[tuple[int, int, int]]:
    """Where to start windows of `window` voxels a side so that each of `voxels` ((n, 3) in a
    volume of `shape`) is within a quarter window of a window's centre -- as near the centre as
    the volume's edges allow. Greedy: a voxel already that close to a window placed before gets
    none of its own."""
    origins: list[tuple[int, int, int]] = []
    centres: list[np.ndarray] = []
    upper = np.maximum(np.asarray(shape) - window, 0)
    for voxel in np.asarray(voxels, dtype=np.int64).reshape(-1, 3):
        if any((np.abs(voxel - centre) <= window // 4).all() for centre in centres):
            continue
        origin = np.clip(voxel - window // 2, 0, upper)
        origins.append(tuple(int(o) for o in origin))
        centres.append(origin + np.minimum(window, np.asarray(shape)) // 2)
    return origins


def blend_weights(size: tuple[int, int, int]) -> np.ndarray:
    """How much a refined window replaces the base at each of its voxels: 1 within the inner
    half of the window (where the clicks are), falling linearly to 0 at its faces."""
    axes = []
    for n in size:
        x = np.abs(np.arange(n, dtype=np.float32) - (n - 1) / 2) / max(n / 2, 1)  # 0 at the middle, 1 at the faces
        axes.append(np.clip(2 - 2 * x, 0, 1))
    return axes[0][:, None, None] * axes[1][None, :, None] * axes[2][None, None, :]


def learned_potentials(
    extractor: Learned,
    features: FeatureVolume,
    grid: Grid,
    points: np.ndarray,
    labels: np.ndarray,
    settings: MRFSettings,
    device: torch.device,
    progress: Progress | None = None,
) -> Potentials:
    """`mrf.Potentials` from the network's raw output kept as features, re-made around the points
    when there are any (`Learned.refine`): see `PotentialNet.potentials`."""
    if features.channels != OUT_CHANNELS:
        raise InputError(f"The learned potentials need {OUT_CHANNELS} feature channels, not {features.channels}")
    raw = np.asarray(features.values, dtype=np.float32)
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    if len(points):
        if features.intensity is None:
            raise InputError("The features carry no intensity grid to re-run the network on")
        voxels = features.ct_to_feature_index(grid.world_to_index(points))
        clicks = np.hstack([voxels, np.asarray(labels, dtype=np.int64).reshape(-1, 1)])
        raw = extractor.refine(np.asarray(features.intensity, dtype=np.float32), raw, clicks, device, progress)
    return PotentialNet.potentials(torch.from_numpy(raw).to(device), settings.unary_shift, settings.coupling)
