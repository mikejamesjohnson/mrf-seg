"""MRFSettings: everything about how an MRFSegmentation segments that the user may change.

Mirrored by the editor in `src/lib/segmentation/settings.ts` (field names in camelCase there,
with the same defaults and `BOUNDS`); a test on each side checks the two agree. Adding a setting
means a field here, with its validation, and one there.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from vtkmodules.vtkCommonDataModel import vtkImageData

from .errors import InputError, is_number, is_whole
from .extractors import FEATURE_EXTRACTORS, make_extractor
from .features import default_feature_spacing, extractor_native_spacing
from .rw import PRECONDITIONERS, SolverOptions

#: What each number may be, inclusive: the editor's fields allow the same.
BOUNDS: dict[str, tuple[float, float]] = {
    "feature_spacing": (0.001, 100.0),
    "contrast": (1.0, 10000.0),
    "affinity_scale": (0.1, 5.0),
    "seed_hu": (-1000.0, 1000.0),
    "seed_erosion": (0, 5),
    "tolerance": (1e-7, 0.1),
    "max_iterations": (1, 100000),
}
#: Of those, the whole numbers.
WHOLE = ("seed_erosion", "max_iterations")
#: The settings that None leaves to the image, chosen from its grid (`MRFSettings.resolved_for`).
IMAGE_DERIVED = ("feature_spacing",)
#: The settings that are true or false.
FLAGS = ("background_seeds",)
DEVICES = ("auto", "cpu")


@dataclass(frozen=True)
class MRFSettings:
    """Everything about how an MRFSegmentation segments that the user may change."""

    feature_extractor: str = "basic"
    """Which `extractors.EXTRACTORS` entry gives the graph its affinities: "basic", Grady's
    contrast affinities from the image's Hounsfield units (`basic.py`), or "learned", our own
    network's log-affinities, trained on coronary CTA (`learned/`). Changing it extracts the
    features again."""
    feature_spacing: float | None = None
    """mm: the isotropic grid the features are extracted and the walker solved on; changing it
    re-extracts the features. None: the image's finest voxel size, at least 0.5 mm (or the
    spacing the extractor was trained on), doubled until the grid fits in memory
    (`features.default_feature_spacing`)."""
    device: str = "auto"
    """Where the network and the solver run: "auto" (the GPU if there is one: CUDA, else the
    Apple GPU) or "cpu"."""
    contrast: float = 40.0
    """Basic: delta, HU. An edge whose two voxels differ by this much keeps exp(-1/2) = 61 % of
    its affinity; by three times this, 1 %."""
    affinity_scale: float = 1.0
    """Learned: what the network's log-affinities are multiplied by before the exponential.
    1 is as trained; above 1 sharpens the contrast between the edges it finds strong and weak
    (a click reaches further, and leaks less, where the network was right), below 1 softens it."""
    background_seeds: bool = True
    """Seed every voxel below `seed_hu` (eroded `seed_erosion` voxels) as outside: fat and air,
    which surround the coronaries within a few voxels and drain what a click's vessel does not
    hold in. Off, the only outside seeds are the outside points."""
    seed_hu: float = 0.0
    """HU: voxels below this are background seeds (fat is about -100, soft tissue above 0)."""
    seed_erosion: int = 1
    """Voxels: how far the background seeds keep from anything above `seed_hu`, so a partial-
    volume voxel at a vessel's edge is never seeded."""
    tolerance: float = 1e-4
    """The solver stops once its preconditioner proposes no correction larger than this to any
    voxel's probability (`rw.SolverOptions`)."""
    max_iterations: int = 1000
    """Conjugate-gradient iterations at most; the result says whether it converged."""
    preconditioner: str = "multigrid"
    """"multigrid" (a few dozen iterations whatever the vessel's length) or "jacobi" (cheaper
    iterations, hundreds of them)."""

    def __post_init__(self) -> None:
        for name, (low, high) in BOUNDS.items():
            value = getattr(self, name)
            if value is None and name in IMAGE_DERIVED:
                continue
            if name in WHOLE:
                if not is_whole(value, int(low), int(high)):
                    raise InputError(f"{name} must be a whole number from {low:g} to {high:g}, not {value!r}")
            elif not (is_number(value) and low <= value <= high):
                raise InputError(f"{name} must be a number from {low:g} to {high:g}, not {value!r}")
        for name in FLAGS:
            if not isinstance(getattr(self, name), bool):
                raise InputError(f"{name} must be true or false, not {getattr(self, name)!r}")
        if self.device not in DEVICES:
            raise InputError(f"device must be one of {', '.join(map(repr, DEVICES))}, not {self.device!r}")
        if self.feature_extractor not in FEATURE_EXTRACTORS:
            raise InputError(f"feature_extractor must be one of {', '.join(map(repr, FEATURE_EXTRACTORS))}, not {self.feature_extractor!r}")
        if self.preconditioner not in PRECONDITIONERS:
            raise InputError(f"preconditioner must be one of {', '.join(map(repr, PRECONDITIONERS))}, not {self.preconditioner!r}")

    def resolved_for(self, volume: vtkImageData | None) -> MRFSettings:
        """These settings, with what they leave to the image (`IMAGE_DERIVED`, None) chosen
        from `volume`'s grid; as they are without a volume."""
        if volume is None or self.feature_spacing is not None:
            return self
        extractor = make_extractor(self)
        spacing = default_feature_spacing(volume.GetDimensions(), volume.GetSpacing(), extractor.channels, extractor_native_spacing(extractor))
        return replace(self, feature_spacing=spacing)

    @property
    def solver(self) -> SolverOptions:
        return SolverOptions(self.tolerance, self.max_iterations, self.preconditioner)

    def affinity_key(self) -> tuple:
        """What the affinities depend on besides the features: a solve may start from the last
        probability when this and the features are unchanged."""
        return (self.contrast, self.affinity_scale)
