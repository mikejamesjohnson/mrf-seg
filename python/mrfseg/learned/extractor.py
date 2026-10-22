"""Learned, the trained network as a `FeatureExtractor` for the editor, and its affinities.

Its "features" are the network's three output channels per voxel (`model.OUT_CHANNELS`): the
log-affinities of the voxel's edges along +z, +y and +x, made once per image at session open,
in windows of the training patch size that overlap by half and are blended with Gaussian
weights (`mrfseg.windows`). The clicks never reach the network: a Generate only seeds the
walker and solves it on this graph (`segmentation.py`), so it is quick, and a point is an exact
observation rather than something the network has to interpret. `learned_log_affinities` hands
the kept channels to the walker, scaled by the settings' `affinity_scale` (`rw.affinities`
exponentiates and clamps them).

Takes the image in Hounsfield units (`input = "hu"`: `extract_features` leaves the HU grid as it
is instead of min-max scaling it) at its training spacing (`native_spacing`). Within a window
the output depends on the window's content only through each voxel's surroundings, so the
windows agree where they overlap except for a rim at their edges, which the weights outvote.

The checkpoint is `$MRFSEG_LEARNED_WEIGHTS`, else `learned.pt` in the mrfseg cache
(`$MRFSEG_CACHE`, default `~/.cache/mrfseg`); it is loaded once per process.
"""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from ..compute import Progress
from ..errors import InputError
from ..features import FeatureVolume
from ..windows import gaussian_weights, tile_starts
from . import SPACING, network_input
from .model import OUT_CHANNELS, AffinityNet, load

if TYPE_CHECKING:
    from ..settings import MRFSettings

_shared: dict[tuple[str, float], tuple[AffinityNet, dict]] = {}     # (resolved path, mtime) -> model
_shared_lock = threading.Lock()
_run_lock = threading.Lock()


def weights_path() -> Path:
    if os.environ.get("MRFSEG_LEARNED_WEIGHTS"):
        return Path(os.environ["MRFSEG_LEARNED_WEIGHTS"])
    return Path(os.environ.get("MRFSEG_CACHE", Path.home() / ".cache" / "mrfseg")) / "learned.pt"


def load_model(path: Path | None = None) -> tuple[AffinityNet, dict]:
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
    """A `FeatureExtractor`: the trained network's log-affinities, in overlapping windows. Cheap
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

    def __init__(self, model: AffinityNet | None = None, extra: dict | None = None, path: Path | None = None) -> None:
        self._given = (model, extra or {}) if model is not None else None
        self._path = path

    def _loaded(self) -> tuple[AffinityNet, dict]:
        return self._given if self._given is not None else load_model(self._path)

    def model(self) -> AffinityNet:
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
        """Voxels a side: what the training saved as the inference window (its `--window`), else
        its patch size."""
        extra = self._loaded()[1]
        return int(extra.get("window") or extra.get("args", {}).get("patch", 96))

    @property
    def setting_defaults(self) -> dict[str, float]:
        """The `MRFSettings` values to start from with this network: its affinities as trained."""
        return {"affinity_scale": 1.0}

    def __call__(self, volume: np.ndarray, device: torch.device, progress: Progress | None = None) -> np.ndarray:
        """(3, z, y, x) float32 log-affinities of a (z, y, x) HU volume, in overlapping windows
        over the whole volume."""
        shape = volume.shape
        window = self.window
        step = max(1, round(window * (1 - self.overlap)))
        tiles = [tile_starts(n, window, step) for n in shape]
        origins = [(z, y, x) for z in tiles[0] for y in tiles[1] for x in tiles[2]]
        out = np.zeros((OUT_CHANNELS, *shape), dtype=np.float32)
        weight = np.zeros(shape, dtype=np.float32)
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
                    size = tuple(min(window, n - o) for o, n in zip(origin, shape))
                    region = tuple(slice(o, o + s) for o, s in zip(origin, size))
                    block = np.asarray(volume[region], dtype=np.float32)
                    x = torch.from_numpy(network_input(block))[None].to(run_on)
                    result = model(x)[0].float().cpu().numpy()
                    w = gaussian_weights(size, self.sigma)
                    out[(slice(None), *region)] += result * w
                    weight[region] += w
                    if progress:
                        progress(done / len(origins))
            finally:
                if not given:
                    model.to("cpu")
        np.divide(out, weight, out=out)
        return out


def learned_log_affinities(extractor: Learned, features: FeatureVolume, settings: MRFSettings, device: torch.device, progress: Progress | None = None) -> torch.Tensor:
    """The log-affinities (3, z, y, x) on `device` from the network's output kept as features,
    times the settings' `affinity_scale`."""
    if features.channels != OUT_CHANNELS:
        raise InputError(f"The learned affinities need {OUT_CHANNELS} feature channels, not {features.channels}")
    z = torch.from_numpy(np.asarray(features.values, dtype=np.float32)).to(device)
    return z * settings.affinity_scale if settings.affinity_scale != 1.0 else z
