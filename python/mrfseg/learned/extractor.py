"""Learned, the trained network as a `FeatureExtractor` for the editor, and its potentials.

Its "features" are the network's four raw output channels per voxel (`model.OUT_CHANNELS`): the
unary logit and the raw couplings, which `learned_potentials` turns into `mrf.Potentials` (the
couplings through softplus, scaled by the settings' `coupling`; the unary shifted by
`unary_shift`). So the network runs once per image, at session open, and a Generate is the
inference alone.

Takes the image in Hounsfield units (`input = "hu"`: `extract_features` leaves the HU grid as it
is instead of min-max scaling it) at its training spacing (`native_spacing`), and runs the
network in windows of the training patch size that overlap by half and are blended with
Gaussian weights (`mrfseg.windows`). Within a window the output depends on the window's content
only through each voxel's surroundings, so the windows agree where they overlap except for a
rim at their edges, which the weights outvote.

The checkpoint is `$MRFSEG_LEARNED_WEIGHTS`, else `learned.pt` in the mrfseg cache
(`$MRFSEG_CACHE`, default `~/.cache/mrfseg`); it is loaded once per process.
"""

from __future__ import annotations

import math
import os
import threading
from pathlib import Path

import numpy as np
import torch

from ..compute import Progress
from ..errors import InputError
from ..features import FeatureVolume
from ..mrf import Potentials
from ..windows import gaussian_weights, tile_starts
from . import SPACING, normalize
from .model import OUT_CHANNELS, PotentialNet, load

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
        """(4, z, y, x) float32 raw potentials of a (z, y, x) HU volume."""
        model = self.model()
        shape = volume.shape
        window = self.window
        step = max(1, round(window * (1 - self.overlap)))
        tiles = [tile_starts(n, window, step) for n in shape]
        total = math.prod(len(t) for t in tiles)
        out = np.zeros((OUT_CHANNELS, *shape), dtype=np.float32)
        weight = np.zeros(shape, dtype=np.float32)
        done = 0
        # A model given to the constructor (training, tests) runs where it is; the shared one is
        # moved to the device asked for, and back, so it is not left on a GPU between images.
        given = self._given is not None
        run_on = next(model.parameters()).device if given else device
        with _run_lock, torch.inference_mode():
            if not given:
                model.to(device)
            try:
                for z in tiles[0]:
                    for y in tiles[1]:
                        for x in tiles[2]:
                            region = np.s_[z:z + window, y:y + window, x:x + window]
                            block = normalize(np.asarray(volume[region], dtype=np.float32))
                            result = model(torch.from_numpy(block)[None, None].to(run_on))[0].float().cpu().numpy()
                            w = gaussian_weights(block.shape, self.sigma)
                            out[(slice(None), *region)] += result * w
                            weight[region] += w
                            done += 1
                            if progress:
                                progress(done / total)
            finally:
                if not given:
                    model.to("cpu")
        np.divide(out, weight, out=out)
        return out


def learned_potentials(features: FeatureVolume, unary_shift: float, coupling_scale: float, device: torch.device) -> Potentials:
    """`mrf.Potentials` from the network's raw output kept as features: see `PotentialNet.potentials`."""
    if features.channels != OUT_CHANNELS:
        raise InputError(f"The learned potentials need {OUT_CHANNELS} feature channels, not {features.channels}")
    raw = torch.from_numpy(np.asarray(features.values, dtype=np.float32)).to(device)
    return PotentialNet.potentials(raw, unary_shift, coupling_scale)
