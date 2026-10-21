"""MRFSettings: everything about how an MRFSegmentation segments that the user may change.

Mirrored by the editor in `src/lib/segmentation/settings.ts` (field names in camelCase there,
with the same defaults and `BOUNDS`); a test on each side checks the two agree. Adding a setting
means a field here, with its validation, and one there.
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from vtkmodules.vtkCommonDataModel import vtkImageData

from .basic import MAX_SCALES
from .errors import InputError, is_number, is_whole
from .extractors import FEATURE_EXTRACTORS, make_extractor
from .features import default_feature_spacing, extractor_native_spacing
from .handcrafted import HandcraftedParameters

#: The default l_s, in the image's largest voxel size.
SPATIAL_LENGTHSCALE_VOXELS = 20

#: What each number may be, inclusive: the editor's fields allow the same.
BOUNDS: dict[str, tuple[float, float]] = {
    "feature_spacing": (0.001, 100.0),
    "basic_scales": (1, MAX_SCALES),
    "feature_lengthscale": (0.001, 10.0),
    "spatial_lengthscale": (0.1, 10000.0),
    "prior_logit": (-20.0, 20.0),
    "unary_scale": (0.0, 20.0),
    "unary_shift": (-20.0, 20.0),
    "coupling": (0.0, 100.0),
    "contrast": (1.0, 10000.0),
    "coupling_floor": (0.0, 1.0),
    "sweeps": (0, 2000),
}
#: Of those, the whole numbers.
WHOLE = ("basic_scales", "sweeps")
#: The settings that None leaves to the image, chosen from its grid (`MRFSettings.resolved_for`).
IMAGE_DERIVED = ("feature_spacing", "spatial_lengthscale")
#: The settings that are true or false.
FLAGS = ("basic_edges", "basic_shape")
DEVICES = ("auto", "cpu")


@dataclass(frozen=True)
class MRFSettings:
    """Everything about how an MRFSegmentation segments that the user may change."""

    feature_extractor: str = "basic"
    """Which `extractors.EXTRACTORS` entry gives each voxel its features, and so how the CRF's
    potentials are made: "basic", the image's brightness at several scales with handcrafted
    potentials from the clicks (`handcrafted.py`), or "learned", our own network's potentials,
    trained on coronary CTA (`learned/`). Changing it extracts the features again."""
    feature_spacing: float | None = None
    """mm: the isotropic grid the features are extracted and the CRF inferred on; changing it
    re-extracts the features. None: the image's finest voxel size, at least 0.5 mm (or the
    spacing the extractor was trained on), doubled until the grid fits in memory
    (`features.default_feature_spacing`)."""
    basic_scales: int = 4
    """Basic: how many blurs, sigma = the grid spacing and doublings of it (4: 1, 2, 4 and 8
    times it). The coarse blurs tell a thin vessel from a chamber: both are bright, but a blur
    wider than the vessel dims it and not the chamber."""
    basic_edges: bool = False
    """Basic: add each blur's edge strength, the magnitude of its gradient."""
    basic_shape: bool = False
    """Basic: add each blur's local shape, the three eigenvalues of its Hessian (a tube's are two
    large negatives and one near zero). Slower to extract."""
    device: str = "auto"
    """Where the network, the potentials and the inference run: "auto" (the GPU if there is one:
    CUDA, else the Apple GPU) or "cpu". Basic's blurs always run on the CPU."""
    feature_lengthscale: float = 0.5
    """Basic: l_f, the width of the clicks' kernels in feature space, in standard deviations of
    the image's own features (`handcrafted.py`): how far from a click's features a voxel may be
    and still count as its kind."""
    spatial_lengthscale: float | None = None
    """Basic: l_s, mm, the spatial length scale of the clicks' kernel density: how far a click's
    evidence reaches. None: SPATIAL_LENGTHSCALE_VOXELS times the image's largest voxel size."""
    prior_logit: float = -2.0
    """Basic: a_0, added to every unary logit: how much more the evidence for lumen must
    outweigh the evidence against before a voxel is inside. A voxel like no click is background
    whatever this is."""
    unary_scale: float = 1.0
    """Basic: beta, how strongly the clicks' evidence counts against the coupling."""
    unary_shift: float = 0.0
    """Learned: added to every unary logit the network gives, to move its decision boundary
    (the base rate it was trained at is in its logits)."""
    coupling: float = 1.0
    """lambda: the coupling of an edge with no contrast across it (Basic), or what the network's
    couplings are multiplied by (Learned; 1 is as trained). 0 turns the pairwise term off, and
    the probability map is then the unary's sigmoid."""
    contrast: float = 100.0
    """Basic: delta, HU. An edge whose two voxels differ by this much in HU keeps exp(-1/2) = 61 %
    of the coupling; by three times this, 1 %."""
    coupling_floor: float = 0.1
    """Basic: the share of the coupling every edge keeps whatever its contrast, for smoothness in
    flat regions."""
    sweeps: int = 50
    """Mean-field sweeps (each updates both parities of the checkerboard). Information travels one
    voxel per half-sweep, so this bounds how far a click reaches through the coupling. Stops
    early once a sweep changes no marginal by more than 0.001."""

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

    def resolved_for(self, volume: vtkImageData | None) -> MRFSettings:
        """These settings, with what they leave to the image (`IMAGE_DERIVED`, None) chosen
        from `volume`'s grid; as they are without a volume."""
        if volume is None:
            return self
        dims, spacing = volume.GetDimensions(), volume.GetSpacing()
        chosen = {}
        if self.feature_spacing is None:
            extractor = make_extractor(self)
            chosen["feature_spacing"] = default_feature_spacing(dims, spacing, extractor.channels, extractor_native_spacing(extractor))
        if self.spatial_lengthscale is None:
            chosen["spatial_lengthscale"] = SPATIAL_LENGTHSCALE_VOXELS * float(max(spacing))
        return replace(self, **chosen) if chosen else self

    @property
    def handcrafted(self) -> HandcraftedParameters:
        """The handcrafted potentials' parameters. Only for settings resolved for a volume (`resolved_for`)."""
        if self.spatial_lengthscale is None:
            raise InputError("spatial_lengthscale has not been chosen yet: resolve the settings for a volume first")
        return HandcraftedParameters(
            feature_lengthscale=self.feature_lengthscale,
            spatial_lengthscale=self.spatial_lengthscale,
            prior_logit=self.prior_logit,
            unary_scale=self.unary_scale,
            coupling=self.coupling,
            contrast=self.contrast,
            coupling_floor=self.coupling_floor,
        )
