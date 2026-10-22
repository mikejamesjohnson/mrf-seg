"""The feature extractors an `MRFSettings.feature_extractor` can name, in one table, each with
the way the walker's affinities are made from its features.

Each entry says how to make the extractor from the settings, which settings it reads (so the
features are extracted again exactly when one of those changes, `features_key`), and how to
turn its features into the log-affinities of the grid's edges (`rw.affinities` exponentiates
them). To add an extractor: write a `FeatureExtractor` (`features.py`) and an affinity
function, add them here, and offer the name in the editor (`src/lib/segmentation/settings.ts`);
any options it needs are new `MRFSettings` fields, named in its `reads`.

Two so far:

- **basic**: the image itself, with Grady's contrast affinities (`basic.py`): the classic
  random walker.
- **learned**: our own network, trained on coronary CTA (`learned/`), whose three output
  channels *are* the log-affinities of each voxel's edges. Needs its checkpoint.

Neither reads the clicks: those are the walker's seeds (`segmentation.py`).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from .basic import Basic, contrast_log_affinities
from .compute import Progress, resolve_device
from .errors import InputError
from .features import FeatureExtractor, FeatureVolume
from .learned.extractor import Learned, learned_log_affinities, weights_path

if TYPE_CHECKING:
    from .settings import MRFSettings

#: extractor, features, settings, device, progress -> the log-affinities (3, z, y, x) float32 on the device.
AffinityMaker = Callable[[FeatureExtractor, FeatureVolume, "MRFSettings", torch.device, "Progress | None"], torch.Tensor]


@dataclass(frozen=True)
class ExtractorSpec:
    make: Callable[[MRFSettings], FeatureExtractor]
    """The extractor these settings describe. Cheap: called whenever the features are extracted."""
    affinities: AffinityMaker
    """The edges' log-affinities from its features."""
    reads: tuple[str, ...]
    """The settings fields whose values change what it extracts, besides `feature_spacing`."""
    key: Callable[[MRFSettings], tuple] | None = None
    """What the features depend on, when it is not the raw values of `reads`."""
    describe: Callable[[], dict] | None = None
    """What the editor asks before choosing it (`describe_extractor`); None: always available,
    no settings of its own."""


def _basic(extractor: FeatureExtractor, features: FeatureVolume, settings: MRFSettings, device: torch.device, progress: Progress | None) -> torch.Tensor:
    if features.intensity is None:
        raise InputError("The features carry no intensity grid for the contrast affinities")
    intensity = torch.from_numpy(np.asarray(features.intensity, dtype=np.float32)).to(device)
    return contrast_log_affinities(intensity, settings.contrast)


def _learned(extractor: FeatureExtractor, features: FeatureVolume, settings: MRFSettings, device: torch.device, progress: Progress | None) -> torch.Tensor:
    if not isinstance(extractor, Learned):
        raise InputError("The learned affinities need the Learned extractor")
    return learned_log_affinities(extractor, features, settings, device, progress)


def _describe_learned() -> dict:
    """Whether the checkpoint is there, and the settings to start from with it."""
    path = weights_path()
    if not path.is_file():
        return {"available": False, "settings": None, "native_spacing": None, "note": f"No checkpoint at {path}: set MRFSEG_LEARNED_WEIGHTS, or copy a trained best.pt there"}
    try:
        extractor = Learned()
        return {"available": True, "settings": extractor.setting_defaults, "native_spacing": extractor.native_spacing, "note": str(path)}
    except Exception as error:  # a file that is not a checkpoint: say so rather than fail every request
        return {"available": False, "settings": None, "native_spacing": None, "note": f"Could not read the checkpoint at {path}: {error}"}


EXTRACTORS: dict[str, ExtractorSpec] = {
    "basic": ExtractorSpec(
        make=lambda s: Basic(),
        affinities=_basic,
        reads=(),
    ),
    "learned": ExtractorSpec(
        make=lambda s: Learned(),
        affinities=_learned,
        reads=("device",),
        # And the checkpoint: swapping the file re-extracts.
        key=lambda s: (str(resolve_device(s.device)), str(weights_path()), _mtime(weights_path())),
        describe=_describe_learned,
    ),
}


def _mtime(path) -> float | None:
    try:
        return os.path.getmtime(path)
    except OSError:
        return None


#: Their names, as the settings and the editor spell them.
FEATURE_EXTRACTORS = tuple(EXTRACTORS)


def extractor_spec(name: str) -> ExtractorSpec:
    try:
        return EXTRACTORS[name]
    except KeyError:
        raise InputError(f"feature_extractor must be one of {', '.join(map(repr, FEATURE_EXTRACTORS))}, not {name!r}") from None


def make_extractor(settings: MRFSettings) -> FeatureExtractor:
    """The extractor `settings` choose."""
    return extractor_spec(settings.feature_extractor).make(settings)


def describe_extractor(name: str) -> dict:
    """What the editor shows about `name` before choosing it: `available` (whether it can run
    here), `settings` (the `MRFSettings` values to start from with it, by their names, or None
    for the editor's own defaults), `native_spacing` (mm, or None) and a `note` (where its
    weights are, or why it is unavailable)."""
    spec = extractor_spec(name)
    if spec.describe is None:
        return {"available": True, "settings": None, "native_spacing": None, "note": None}
    return spec.describe()


def features_key(settings: MRFSettings) -> tuple:
    """What the features depend on: the extractor, its grid, and the settings it reads. The
    features are extracted again when this changes, and only then."""
    spec = extractor_spec(settings.feature_extractor)
    read = spec.key(settings) if spec.key else tuple(getattr(settings, field) for field in spec.reads)
    return (settings.feature_extractor, settings.feature_spacing, *read)
