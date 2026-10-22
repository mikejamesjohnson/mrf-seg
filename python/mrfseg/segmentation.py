"""MRFSegmentation: segmenting one image with the random walker on learned (or contrast) affinities."""

from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass, field, replace

import numpy as np
import torch
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPolyData

from .compute import Progress, resolve_device
from .errors import InputError, is_number
from .extractors import AffinityMaker, extractor_spec, features_key, make_extractor
from .features import FeatureExtractor, FeatureVolume, check_feature_grid, extract_features, feature_grid_dims
from .grid import Grid, image_like, resample_to, voxels
from .rw import RandomWalkerResult, Seeds, affinities, check_grid, components, mean_log_affinity, random_walker
from .settings import MRFSettings
from .surface import extract_surface

#: More points than anyone clicks; the seeds are dense anyway, so this only bounds a request.
MAX_POINTS = 500
#: How far outside the outermost voxel centres a point may be, in voxels: a point clamped to
#: the edge of an oblique grid does not project back exactly onto it.
INSIDE_TOLERANCE = 1e-3


@dataclass(frozen=True)
class InferenceSummary:
    """How the walker's solve ended."""

    iterations: int
    """Conjugate-gradient iterations made."""
    residual: float
    """The largest correction the preconditioner still proposed (`rw.SolverOptions.tolerance`)."""
    converged: bool
    components: int
    """How many face-connected pieces {p >= 0.5} has. By the maximum principle each holds an
    inside point, so more pieces than inside points means the solve did not converge."""
    seeds: dict[str, int] = field(default_factory=dict)
    """How many seeds of each kind: "inside" and "outside" points, "background"."""


@dataclass(frozen=True)
class Segmentation:
    """Two float32 images on the reference volume's grid, and how they were made."""

    probability: vtkImageData
    """p(y = 1 | image, points): the walker's probability of reaching an inside point first."""
    affinity: vtkImageData
    """The mean log-affinity of the edges at each voxel: positive where the label flows freely,
    negative where it is held back (a wall)."""
    inference: InferenceSummary
    seconds: dict[str, float] = field(default_factory=dict)
    """How long each step took: features (0 when they were already there), affinities, seeds, solve, maps."""


class MRFSegmentation:
    """Random-walker segmentation of a reference volume.

    Keeps state between segmentations: the volume's features, extracted once per setting, the
    last segmentation, and the last probability on the solver's grid, which the next solve
    starts from when the affinities have not changed (a new point changes the solution near
    itself most). Positions are world coordinates, LPS mm, like the volume's.

    The settings choose the feature extractor (`MRFSettings.feature_extractor`), and with it how
    the affinities are made (`extractors.EXTRACTORS`). One given here is used instead, whatever
    they say: for tests, or an extractor of your own; its affinities are then made by
    `affinities` (an `extractors.AffinityMaker`), the contrast ones unless given.
    """

    def __init__(
        self,
        reference_volume: vtkImageData | None = None,
        settings: MRFSettings | None = None,
        feature_extractor: FeatureExtractor | None = None,
        affinities: AffinityMaker | None = None,
    ) -> None:
        self._reference_volume: vtkImageData | None = None
        self._given_extractor = feature_extractor
        self._given_affinities = affinities
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
        self._last_probability: tuple[tuple, torch.Tensor] | None = None
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
        """Whether a Generate needs an inside point: always, for the walker -- without one
        nothing is selected."""
        return True

    def check_settings(self, settings: MRFSettings | None = None) -> None:
        """An InputError if `settings` (the current ones, if not given) cannot be used on the
        reference volume: a feature grid too large for memory, or for the solver."""
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
            self._last_probability = None
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
        """Build the graph's affinities from the features, seed it with the labelled points (and
        the background), solve the walker, and read the maps onto the reference volume's grid.

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
        report("affinities")
        started = time.perf_counter()
        w = affinities(self._log_affinities(features, device, progress))
        seconds["affinities"] = time.perf_counter() - started

        report("seeds")
        started = time.perf_counter()
        seeds = self._seeds(features, grid, points, labels, device)
        seconds["seeds"] = time.perf_counter() - started

        report("solve")
        started = time.perf_counter()
        key = (self._features_for, self.settings.affinity_key())
        initial = self._last_probability[1] if self._last_probability is not None and self._last_probability[0] == key else None
        result = random_walker(w, seeds, self.settings.solver, progress, initial)
        self._last_probability = (key, result.probability)
        pieces = components(result.probability)
        seconds["solve"] = time.perf_counter() - started

        report("maps")
        started = time.perf_counter()
        self.last_segmentation = None  # its memory is free for the new maps
        summary = InferenceSummary(result.iterations, result.residual, result.converged, pieces, seeds.counts())
        segmentation = self._maps(volume, w, result, summary, progress)
        seconds["maps"] = time.perf_counter() - started
        self.last_segmentation = replace(segmentation, seconds=seconds)
        return self.last_segmentation

    def _log_affinities(self, features: FeatureVolume, device: torch.device, progress: Progress | None) -> torch.Tensor:
        """The log-affinities the extractor's spec makes from its features; for an extractor
        given to the constructor, the maker given with it, or the contrast ones."""
        if self._given_extractor is not None:
            if self._given_affinities is not None:
                return self._given_affinities(self._given_extractor, features, self.settings, device, progress)
            from .extractors import _basic

            return _basic(self._given_extractor, features, self.settings, device, progress)
        spec = extractor_spec(self.settings.feature_extractor)
        return spec.affinities(self.feature_extractor(), features, self.settings, device, progress)

    def _seeds(self, features: FeatureVolume, grid: Grid, points: np.ndarray, labels: np.ndarray, device: torch.device) -> Seeds:
        """The points as seeds on the feature grid (the nearest voxel to each), and the
        background below `seed_hu` if the settings ask for it."""
        seeds = Seeds.from_voxels(features.ct_to_feature_index(grid.world_to_index(points)), labels, features.shape, device)
        if self.settings.background_seeds:
            if features.intensity is None:
                raise InputError("The features carry no intensity grid for the background seeds")
            intensity = torch.from_numpy(np.asarray(features.intensity, dtype=np.float32)).to(device)
            seeds = seeds.with_background(intensity, self.settings.seed_hu, self.settings.seed_erosion)
        return seeds

    def extract_surface(self, source: str = "probability", level: float | None = None) -> vtkPolyData:
        """A triangle mesh of the last segmentation's boundary: the probability's `level` set
        (0.5 unless given)."""
        segmentation = self._require_segmentation()
        if source != "probability":
            raise InputError(f"Unknown surface source {source!r}: 'probability'")
        return extract_surface(segmentation.probability, _probability_level(0.5 if level is None else level))

    def labelmap(self, threshold: float = 0.5) -> vtkImageData:
        """The last segmentation's voxels with p(y = 1) >= `threshold`, as a uint8 image on the
        reference grid: 1 inside, 0 outside. Every piece of it holds an inside point (the
        maximum principle), so the labelmap and the surface at one threshold agree."""
        segmentation = self._require_segmentation()
        level = _probability_level(threshold)
        image, labels = image_like(segmentation.probability, "label", np.uint8)
        probability = voxels(segmentation.probability)
        for k in range(len(labels)):  # a slice at a time: no float volume the size of the CT
            labels[k] = probability[k] >= level
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

    def _maps(self, volume: vtkImageData, w: torch.Tensor, result: RandomWalkerResult, summary: InferenceSummary, progress: Progress | None) -> Segmentation:
        """The probability and the mean log-affinity read onto the CT's grid, a slice at a time."""
        dims = volume.GetDimensions()
        probability_image, probability = image_like(volume, "probability")
        affinity_image, affinity = image_like(volume, "affinity")
        sources = {"probability": result.probability, "affinity": mean_log_affinity(w)}
        targets = {"probability": probability, "affinity": affinity}
        nz = dims[2]
        for k in range(nz):
            for name, source in sources.items():
                targets[name][k] = resample_to(source, dims, k).cpu().numpy()
            if progress:
                progress((k + 1) / nz)
        return Segmentation(probability_image, affinity_image, summary)


def check_grids(ct_dims: tuple[int, int, int], ct_spacing: tuple[float, float, float], spacing: float, channels: int) -> None:
    """An InputError if the feature grid of `spacing` mm is too large for the features' memory
    (`features.check_feature_grid`) or for the solver (`rw.check_grid`), saying what spacing
    would satisfy both."""

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
