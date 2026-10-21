"""MRFSegmentation: segmenting one image with a conditional random field and mean field."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np
import torch
from scipy import ndimage
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPolyData

from .compute import Progress, resolve_device
from .errors import InputError, is_number
from .extractors import PotentialsMaker, extractor_spec, features_key, make_extractor
from .features import FeatureExtractor, FeatureVolume, check_feature_grid, extract_features, feature_grid_dims
from .grid import Grid, image_like, voxels
from .mrf import Clamps, MeanFieldResult, Potentials, check_grid, mean_field, resample_to
from .settings import MRFSettings
from .surface import extract_surface

#: The handcrafted unary keeps a (voxels of a slice x points) kernel; past a few hundred points
#: that is gigabytes, and nobody clicks that often.
MAX_POINTS = 500
#: How far outside the outermost voxel centres a point may be, in voxels: a point clamped to
#: the edge of an oblique grid does not project back exactly onto it.
INSIDE_TOLERANCE = 1e-3
#: The labelmap's connectivity: face neighbours, as the CRF's edges.
SIX = ndimage.generate_binary_structure(3, 1)


@dataclass(frozen=True)
class InferenceSummary:
    """How the mean-field inference ended."""

    sweeps: int
    """Full sweeps made (fewer than asked when it converged early)."""
    delta: float
    """The largest change of any marginal in the last sweep."""
    converged: bool


@dataclass(frozen=True)
class Segmentation:
    """Three float32 images on the reference volume's grid, and how they were made."""

    probability: vtkImageData
    """p(y = 1 | image, clicks): the mean-field marginal, the chance a voxel is inside."""
    unary: vtkImageData
    """a: the unary logit. Positive is evidence for inside, before the coupling has its say."""
    coupling: vtkImageData
    """The mean coupling of the edges at each voxel: high where a label change is expensive."""
    inference: InferenceSummary
    seeds: np.ndarray
    """(k, 3) int64: the CT voxel (i, j, k) of each inside point, for the labelmap's connectivity."""
    seconds: dict[str, float] = field(default_factory=dict)
    """How long each step took: features (0 when they were already there), potentials, inference, maps."""


class MRFSegmentation:
    """Conditional-random-field segmentation of a reference volume.

    Keeps state between segmentations: the volume's features, extracted once per setting,
    and the last segmentation. Positions are world coordinates, LPS mm, like the volume's.

    The settings choose the feature extractor (`MRFSettings.feature_extractor`), and with it how
    the potentials are made (`extractors.EXTRACTORS`). One given here is used instead, whatever
    they say: for tests, or an extractor of your own; its potentials are then made by `potentials`
    (an `extractors.PotentialsMaker`), the handcrafted ones unless given.
    """

    def __init__(
        self,
        reference_volume: vtkImageData | None = None,
        settings: MRFSettings | None = None,
        feature_extractor: FeatureExtractor | None = None,
        potentials: PotentialsMaker | None = None,
    ) -> None:
        self._reference_volume: vtkImageData | None = None
        self._given_extractor = feature_extractor
        self._given_potentials = potentials
        self.settings = settings or MRFSettings()
        self.reference_volume = reference_volume

    @property
    def settings(self) -> MRFSettings:
        """The settings in use: as given, with what they leave to the image chosen from the
        reference volume (`MRFSettings.resolved_for`)."""
        return self._settings

    @settings.setter
    def settings(self, settings: MRFSettings) -> None:
        self._requested = settings
        self._settings = settings.resolved_for(self._reference_volume)

    @property
    def reference_volume(self) -> vtkImageData | None:
        """The image to segment, or None. Setting it forgets the features and the last segmentation."""
        return self._reference_volume

    @reference_volume.setter
    def reference_volume(self, volume: vtkImageData | None) -> None:
        if volume is not None and not isinstance(volume, vtkImageData):
            raise TypeError(f"reference_volume must be a vtkImageData or None, not {type(volume).__name__}")
        self._reference_volume = volume
        self._settings = self._requested.resolved_for(volume)
        self._features: FeatureVolume | None = None
        self._features_for: tuple | None = None
        self.last_segmentation: Segmentation | None = None

    @property
    def features(self) -> FeatureVolume | None:
        """The volume's features, once extracted."""
        return self._features

    def has_features(self) -> bool:
        """Whether the features for the current settings are already there."""
        return self._features is not None and self._features_for == features_key(self.settings)

    def feature_extractor(self) -> FeatureExtractor:
        """The extractor the settings choose, or the one this was given."""
        return self._given_extractor if self._given_extractor is not None else make_extractor(self.settings)

    def needs_inside_point(self) -> bool:
        """Whether a Generate needs an inside point: the handcrafted unary does, the network does not."""
        if self._given_extractor is not None:
            return self._given_potentials is None
        return extractor_spec(self.settings.feature_extractor).needs_inside_point

    def check_settings(self, settings: MRFSettings | None = None) -> None:
        """An InputError if `settings` (the current ones, if not given) cannot be used on the
        reference volume: a feature grid too large for memory, or for the inference."""
        volume = self._require_volume()
        settings = (settings or self._requested).resolved_for(volume)
        channels = (self._given_extractor or make_extractor(settings)).channels
        check_grids(volume.GetDimensions(), volume.GetSpacing(), settings.feature_spacing, channels)

    def extract_features(self, progress: Progress | None = None) -> FeatureVolume:
        """Extract the features for the current settings, unless they already are."""
        volume = self._require_volume()
        if not self.has_features():
            self.check_settings()
            self._features = None  # let the old ones go before making new ones
            device = resolve_device(self.settings.device)
            self._features = extract_features(volume, self.settings.feature_spacing, self.feature_extractor(), device, progress)
            self._features_for = features_key(self.settings)
        return self._features

    def check_points(self, points: np.ndarray, labels: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """`points` as (n, 3) floats and `labels` as (n,) ints, or an InputError saying what is wrong."""
        grid = Grid(self._require_volume())
        points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
        labels = np.asarray(labels).reshape(-1)
        if len(points) != len(labels):
            raise InputError(f"{len(points)} points but {len(labels)} labels")
        if len(points) > MAX_POINTS:
            raise InputError(f"{len(points)} points: at most {MAX_POINTS}")
        if len(labels) and not set(np.unique(labels)) <= {-1, 1}:
            raise InputError("labels must be +1 (inside) or -1 (outside)")
        if self.needs_inside_point() and not (labels > 0).any():
            raise InputError("Add at least one inside (+1) point")
        if len(points) and not np.isfinite(points).all():
            raise InputError("Point positions must be finite")
        outside = ~grid.contains(grid.world_to_index(points), INSIDE_TOLERANCE)
        if outside.any():
            raise InputError(f"{int(outside.sum())} point(s) lie outside the image")
        return points, labels.astype(np.int64)

    def generate_segmentation(
        self,
        points: np.ndarray,
        labels: np.ndarray,
        progress: Progress | None = None,
        stage: Callable[[str], None] | None = None,
    ) -> Segmentation:
        """Build the CRF's potentials from the features and the labelled points, infer the
        marginals with the points clamped, and read the maps onto the reference volume's grid.

        `points` are (n, 3) world positions inside the image, `labels` +1 inside and -1
        outside. `stage(name)` and `progress(fraction)` report where it has got to. The last
        segmentation stays until the new one is about to be made, so a step that fails leaves it
        in place.
        """
        points, labels = self.check_points(points, labels)
        volume = self._require_volume()
        grid = Grid(volume)
        report = stage or (lambda _: None)
        seconds: dict[str, float] = {}

        report("features")
        started = time.perf_counter()
        had_features = self.has_features()
        features = self.extract_features(progress)
        seconds["features"] = 0.0 if had_features else time.perf_counter() - started

        device = resolve_device(self.settings.device)
        report("potentials")
        started = time.perf_counter()
        potentials = self._potentials(features, grid, points, labels, device, progress)
        clamps = Clamps.from_voxels(features.ct_to_feature_index(grid.world_to_index(points)), labels, features.shape, device)
        seconds["potentials"] = time.perf_counter() - started

        report("inference")
        started = time.perf_counter()
        result = mean_field(potentials, clamps, self.settings.sweeps, progress)
        seconds["inference"] = time.perf_counter() - started

        report("maps")
        started = time.perf_counter()
        self.last_segmentation = None  # its memory is free for the new maps
        seeds = np.rint(grid.world_to_index(points[labels > 0])).astype(np.int64) if (labels > 0).any() else np.zeros((0, 3), dtype=np.int64)
        segmentation = self._maps(volume, potentials, result, seeds, progress)
        seconds["maps"] = time.perf_counter() - started
        self.last_segmentation = replace(segmentation, seconds=seconds)
        return self.last_segmentation

    def _potentials(self, features: FeatureVolume, grid: Grid, points: np.ndarray, labels: np.ndarray, device: torch.device, progress: Progress | None) -> Potentials:
        """The potentials the extractor's spec makes from its features; for an extractor given to
        the constructor, the maker given with it, or the handcrafted ones."""
        if self._given_extractor is not None:
            if self._given_potentials is not None:
                return self._given_potentials(features, grid, points, labels, self.settings, device, progress)
            from .handcrafted import handcrafted_potentials

            return handcrafted_potentials(features, grid, points, labels, self.settings.handcrafted, device, progress)
        return extractor_spec(self.settings.feature_extractor).potentials(features, grid, points, labels, self.settings, device, progress)

    def extract_surface(self, source: str = "probability", level: float | None = None) -> vtkPolyData:
        """A triangle mesh of the last segmentation's boundary: the probability's `level` set
        (0.5 unless given), or the unary logit's (0 unless given), the evidence alone."""
        segmentation = self._require_segmentation()
        if source == "probability":
            return extract_surface(segmentation.probability, _probability_level(0.5 if level is None else level))
        if source == "unary":
            if level is not None and not is_number(level):
                raise InputError(f"A unary level must be a finite number, not {level!r}")
            return extract_surface(segmentation.unary, 0.0 if level is None else level)
        raise InputError(f"Unknown surface source {source!r}: 'probability' or 'unary'")

    def labelmap(self, threshold: float = 0.5, connected: bool = False) -> vtkImageData:
        """The last segmentation's voxels with p(y = 1) >= `threshold`, as a uint8 image on the
        reference grid: 1 inside, 0 outside. With `connected`, only the face-connected
        components holding an inside point are kept (the largest one, when there were no inside
        points): the CRF cannot promise connectivity, so this is where islands go."""
        segmentation = self._require_segmentation()
        level = _probability_level(threshold)
        image, labels = image_like(segmentation.probability, "label", np.uint8)
        probability = voxels(segmentation.probability)
        for k in range(len(labels)):  # a slice at a time: no float volume the size of the CT
            labels[k] = probability[k] >= level
        if connected:
            _keep_connected(labels, segmentation.seeds)
        return image

    def __repr__(self) -> str:
        volume = self._reference_volume
        described = "None" if volume is None else "x".join(map(str, volume.GetDimensions())) + " vtkImageData"
        return f"MRFSegmentation(reference_volume={described})"

    def _require_volume(self) -> vtkImageData:
        if self._reference_volume is None:
            raise InputError("MRFSegmentation has no reference volume to segment")
        return self._reference_volume

    def _require_segmentation(self) -> Segmentation:
        if self.last_segmentation is None:
            raise InputError("No segmentation yet: generate one first")
        return self.last_segmentation

    def _maps(self, volume: vtkImageData, potentials: Potentials, result: MeanFieldResult, seeds: np.ndarray, progress: Progress | None) -> Segmentation:
        """The marginal, the unary and the mean coupling read onto the CT's grid, a slice at a time."""
        dims = volume.GetDimensions()
        probability_image, probability = image_like(volume, "probability")
        unary_image, unary = image_like(volume, "unary")
        coupling_image, coupling = image_like(volume, "coupling")
        sources = {"probability": result.marginal, "unary": potentials.unary, "coupling": potentials.mean_coupling()}
        targets = {"probability": probability, "unary": unary, "coupling": coupling}
        nz = dims[2]
        for k in range(nz):
            for name, source in sources.items():
                targets[name][k] = resample_to(source, dims, k).cpu().numpy()
            if progress:
                progress((k + 1) / nz)
        inference = InferenceSummary(result.sweeps, result.delta, result.converged)
        return Segmentation(probability_image, unary_image, coupling_image, inference, seeds)


def check_grids(ct_dims: tuple[int, int, int], ct_spacing: tuple[float, float, float], spacing: float, channels: int) -> None:
    """An InputError if the feature grid of `spacing` mm is too large for the features' memory
    (`features.check_feature_grid`) or to infer in (`mrf.check_grid`), saying what spacing would
    satisfy both."""

    def check(candidate: float) -> None:
        check_feature_grid(ct_dims, ct_spacing, candidate, channels)
        check_grid(feature_grid_dims(ct_dims, ct_spacing, candidate)[::-1])

    try:
        check(spacing)
    except InputError as error:
        fits = spacing
        while True:
            fits *= 2
            try:
                check(fits)
                break
            except InputError:
                continue
        reason = str(error).split("; use at least")[0]
        raise InputError(f"{reason}; use at least {fits:g} mm") from None


def _probability_level(threshold: float) -> float:
    if not (is_number(threshold) and 0 < threshold < 1):
        raise InputError(f"A probability threshold must be between 0 and 1, not {threshold!r}")
    return float(threshold)


def _keep_connected(labels: np.ndarray, seeds: np.ndarray) -> None:
    """In place: of the face-connected components of `labels` (z, y, x), keep those holding a
    seed ((k, 3) CT indices (i, j, k)), or the largest if there are no seeds."""
    components, count = ndimage.label(labels, structure=SIX)
    if count <= 1:
        return
    keep = np.zeros(0, dtype=np.int64)
    if len(seeds):
        nz, ny, nx = labels.shape
        seeds = np.clip(seeds, 0, [nx - 1, ny - 1, nz - 1])
        keep = np.unique(components[seeds[:, 2], seeds[:, 1], seeds[:, 0]])
        keep = keep[keep > 0]
    if len(keep) == 0:  # no seeds, or none on a labelled voxel: the largest component
        sizes = np.bincount(components.ravel())
        sizes[0] = 0
        keep = np.array([int(sizes.argmax())])
    wanted = np.zeros(count + 1, dtype=bool)
    wanted[keep] = True
    for k in range(len(labels)):
        labels[k] = wanted[components[k]]
