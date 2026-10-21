"""The feature extractors an `MRFSettings.feature_extractor` can name, in one table, each with
the way the CRF's potentials are made from its features.

Each entry says how to make the extractor from the settings, which settings it reads (so the
features are extracted again exactly when one of those changes, `features_key`), and how to
turn its features and the clicks into `mrf.Potentials`. To add an extractor: write a
`FeatureExtractor` (`features.py`) and a potentials function, add them here, and offer the name
in the editor (`src/lib/segmentation/settings.ts`); any options it needs are new `MRFSettings`
fields, named in its `reads`.

Two so far:

- **basic**: blurs of the image (`basic.py`), with handcrafted potentials: kernel-density
  unaries from the clicks and contrast-sensitive couplings (`handcrafted.py`). Needs at least
  one inside click.
- **learned**: our own network, trained on coronary CTA (`learned/`), whose four output
  channels *are* the potentials: a unary logit and three couplings per voxel. It sees the
  clicks, and is re-run around them at every Generate (`learned/extractor.py`). Needs its
  checkpoint.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import torch

from .basic import Basic
from .compute import Progress, resolve_device
from .errors import InputError
from .features import FeatureExtractor, FeatureVolume
from .grid import Grid
from .learned.extractor import Learned, learned_potentials, weights_path
from .mrf import Potentials

if TYPE_CHECKING:
    from .settings import MRFSettings

#: extractor, features, grid, points (n, 3) mm, labels (n,) +1 / -1, settings, device, progress -> the potentials.
PotentialsMaker = Callable[[FeatureExtractor, FeatureVolume, Grid, np.ndarray, np.ndarray, "MRFSettings", torch.device, "Progress | None"], Potentials]


@dataclass(frozen=True)
class ExtractorSpec:
    make: Callable[[MRFSettings], FeatureExtractor]
    """The extractor these settings describe. Cheap: called whenever the features are extracted."""
    potentials: PotentialsMaker
    """The CRF's potentials from its features and the clicks."""
    reads: tuple[str, ...]
    """The settings fields whose values change what it extracts, besides `feature_spacing`."""
    needs_inside_point: bool = True
    """Whether a Generate needs at least one inside click (the handcrafted unary does; the
    network segments with none)."""
    key: Callable[[MRFSettings], tuple] | None = None
    """What the features depend on, when it is not the raw values of `reads`."""
    describe: Callable[[], dict] | None = None
    """What the editor asks before choosing it (`describe_extractor`); None: always available,
    no settings of its own."""


def _handcrafted(extractor: FeatureExtractor, features: FeatureVolume, grid: Grid, points: np.ndarray, labels: np.ndarray,
                 settings: MRFSettings, device: torch.device, progress: Progress | None) -> Potentials:
    from .handcrafted import handcrafted_potentials

    return handcrafted_potentials(features, grid, points, labels, settings.handcrafted, device, progress)


def _learned(extractor: FeatureExtractor, features: FeatureVolume, grid: Grid, points: np.ndarray, labels: np.ndarray,
             settings: MRFSettings, device: torch.device, progress: Progress | None) -> Potentials:
    if not isinstance(extractor, Learned):
        raise InputError("The learned potentials need the Learned extractor")
    return learned_potentials(extractor, features, grid, points, labels, settings, device, progress)


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
        make=lambda s: Basic(s.basic_scales, s.basic_edges, s.basic_shape),
        potentials=_handcrafted,
        reads=("basic_scales", "basic_edges", "basic_shape"),
    ),
    "learned": ExtractorSpec(
        make=lambda s: Learned(),
        potentials=_learned,
        reads=("device",),
        needs_inside_point=False,
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
