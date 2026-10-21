"""The backend's state: a Session per image the editor has open, and its background work."""

from __future__ import annotations

import dataclasses
import threading
import time
import traceback
import uuid
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np
from vtkmodules.vtkCommonDataModel import vtkImageData, vtkPolyData

from ..errors import InputError
from ..features import FeatureVolume
from ..grid import voxels
from ..segmentation import InferenceSummary, MRFSegmentation, MRFSettings, Segmentation
from .encoding import MAPS, compact, compact_scale


class Cancelled(Exception):
    """The session was closed: its background work stops at its next progress report."""


class NotReady(Exception):
    """A request the session cannot serve yet (a 409): no segmentation, or one being generated."""


class Task:
    """A background step's progress, read without a lock by whoever polls."""

    def __init__(self, state: str = "idle", stage: str | None = None, cancelled: threading.Event | None = None) -> None:
        self.state = state  # idle | running | done | error
        self.stage = stage
        self.progress = 0.0
        self.message: str | None = None
        self.result: dict | None = None
        self._cancelled = cancelled

    def report(self) -> dict:
        report = {"state": self.state, "stage": self.stage, "progress": round(self.progress, 4)}
        if self.message:
            report["message"] = self.message
        if self.result is not None:
            report["result"] = self.result
        return report

    def advance(self, fraction: float) -> None:
        """The progress callback the core reports to: where a closed session's work stops."""
        if self._cancelled is not None and self._cancelled.is_set():
            raise Cancelled()
        self.progress = fraction

    def begin(self, stage: str) -> None:
        self.advance(0.0)
        self.stage = stage

    def done(self, result: dict) -> None:
        # The result before the state: a poll that sees "done" must find it.
        self.result, self.message = result, None
        self.state = "done"

    def fail(self, message: str) -> None:
        self.state, self.message = "error", message


@dataclass(frozen=True)
class Result:
    """A segmentation and the scales its maps are sent with: replaced as one, so a reader
    without the lock never pairs one Generate's maps with another's scales."""

    segmentation: Segmentation
    scales: dict[str, float]


class Session:
    """One open image and its MRFSegmentation. `lock` serialises the work on the segmentation;
    the `Task`s, `result` and `products` are what a request reads meanwhile.

    The last result, and the products built from it, stay until the next Generate replaces
    them (or the session closes): a Generate that fails leaves them as they were.
    """

    def __init__(self, segmentation: MRFSegmentation) -> None:
        self.id = uuid.uuid4().hex[:12]
        self.segmentation = segmentation
        self.lock = threading.Lock()
        self.cancelled = threading.Event()
        self._starting = threading.Lock()
        self.features = Task("running", "features", self.cancelled)
        self.job = Task()
        self.result: Result | None = None
        #: Built from `result` on request, by name: "surface" (a vtkPolyData), "labelmap" (a vtkImageData).
        self.products: dict[str, vtkPolyData | vtkImageData] = {}

    def close(self) -> None:
        """Stop the background work at its next progress report, and let the results go."""
        self.cancelled.set()
        self.result = None
        self.products.clear()

    def start_features(self) -> None:
        threading.Thread(target=self._extract_features, name=f"features-{self.id}", daemon=True).start()

    def start_job(self, points: np.ndarray, labels: np.ndarray, settings: MRFSettings) -> Task | None:
        """Generate in the background, or None if a Generate is already running. `points` and
        `settings` have been checked (`MRFSegmentation.check_points`, `check_settings`)."""
        with self._starting:
            if self.busy:
                return None
            self.job = job = Task("running", "queued", self.cancelled)
        threading.Thread(target=self._segment, args=(job, points, labels, settings), name=f"segment-{self.id}", daemon=True).start()
        return job

    @property
    def busy(self) -> bool:
        """Whether a Generate is running."""
        return self.job.state == "running"

    def map_values(self, name: str, compact_encoding: bool) -> np.ndarray | None:
        """One of the last result's maps, flat: float32, or compact. None if there is no result."""
        result = self.result
        if result is None:
            return None
        values = voxels(getattr(result.segmentation, name)).ravel()
        return compact(name, values, result.scales[name]) if compact_encoding else values

    def build_surface(self, source: str, level: float | None) -> vtkPolyData:
        return self._build("surface", lambda: self.segmentation.extract_surface(source, level))

    def build_labelmap(self, threshold: float, connected: bool) -> vtkImageData:
        return self._build("labelmap", lambda: self.segmentation.labelmap(threshold, connected))

    def _build(self, name: str, make):
        """A product of the last result, kept under `name`; a NotReady if there is no result or
        the segmentation is busy (a Generate, or the features at session open)."""
        if self.result is None:
            raise NotReady("No segmentation yet: generate one first")
        if not self.lock.acquire(blocking=False):
            raise NotReady("A segmentation is being generated: build once it is done")
        try:
            self.products[name] = product = make()
            return product
        finally:
            self.lock.release()

    def _extract_features(self) -> None:
        """At session open: the features for the opening settings, in the background."""
        try:
            with self.lock:
                self._extract(None)
        except Cancelled:
            self.features.fail("Cancelled")
        except InputError as error:
            self.features.fail(str(error))
        except Exception as error:  # reported to the editor rather than lost
            traceback.print_exc()
            self.features.fail(f"Feature extraction failed: {error}")

    def _extract(self, job: Task | None) -> float:
        """Extract the features for the current settings, unless they already are, keeping the
        `features` Task true to them throughout; the seconds it took, 0 if they were there.
        Progress goes to `job` too, when there is one. Under the lock."""
        if self.segmentation.has_features():
            return 0.0
        task = self.features = Task("running", "features", self.cancelled)
        started = time.perf_counter()

        def progress(fraction: float) -> None:
            task.advance(fraction)
            if job is not None:
                job.advance(fraction)

        try:
            features = self.segmentation.extract_features(progress=progress)
        except BaseException as error:
            task.fail("Cancelled" if isinstance(error, Cancelled) else str(error) if isinstance(error, InputError) else f"Feature extraction failed: {error}")
            raise
        seconds = time.perf_counter() - started
        task.done(_features_summary(features, self.segmentation.settings, seconds))
        return seconds

    def _segment(self, job: Task, points: np.ndarray, labels: np.ndarray, settings: MRFSettings) -> None:
        try:
            with self.lock:
                self.segmentation.settings = settings
                job.begin("features")
                extracting = self._extract(job)
                segmentation = self.segmentation.generate_segmentation(points, labels, progress=job.advance, stage=job.begin)
                segmentation = dataclasses.replace(segmentation, seconds={**segmentation.seconds, "features": extracting})
                ranges = {name: _finite_range(name, voxels(getattr(segmentation, name))) for name in MAPS}
                scales = {name: compact_scale(name, *ranges[name]) for name in MAPS}
                summary = _job_summary(segmentation, len(labels), ranges, scales)
                self.result = Result(segmentation, scales)
                job.done(summary)
        except Cancelled:
            job.fail("Cancelled")
        except InputError as error:  # the message is for the user
            job.fail(str(error))
        except Exception as error:
            traceback.print_exc()
            job.fail(f"{type(error).__name__}: {error}")


class Sessions:
    """The live sessions, least recently used first; only the newest `capacity` are kept, and
    one that is dropped is closed."""

    def __init__(self, capacity: int) -> None:
        self._capacity = capacity
        self._sessions: OrderedDict[str, Session] = OrderedDict()
        self._lock = threading.Lock()

    def add(self, session: Session) -> None:
        with self._lock:
            self._sessions[session.id] = session
            while len(self._sessions) > self._capacity:
                self._sessions.popitem(last=False)[1].close()

    def get(self, session_id: str) -> Session | None:
        with self._lock:
            session = self._sessions.get(session_id)
            if session is not None:
                self._sessions.move_to_end(session_id)
            return session

    def remove(self, session_id: str) -> bool:
        with self._lock:
            session = self._sessions.pop(session_id, None)
        if session is None:
            return False
        session.close()
        return True


def _features_summary(features: FeatureVolume, settings: MRFSettings, seconds: float) -> dict:
    """What the editor shows about the features: `FeaturesSummary` in `src/lib/segmentation/state.ts`."""
    return {
        "extractor": settings.feature_extractor,
        "dims": list(features.dims),
        "spacing": settings.feature_spacing,
        "channels": features.channels,
        "seconds": seconds,
    }


def _finite_range(name: str, values: np.ndarray) -> tuple[float, float]:
    low, high = float(values.min()), float(values.max())
    if not (np.isfinite(low) and np.isfinite(high)):
        raise RuntimeError(f"The {name} map is not finite everywhere")
    return low, high


def _inference_summary(inference: InferenceSummary) -> dict:
    """`InferenceSummary` in `src/lib/segmentation/state.ts`: how the mean field ended."""
    return {"sweeps": inference.sweeps, "delta": inference.delta, "converged": inference.converged}


def _job_summary(segmentation: Segmentation, points: int, ranges: dict[str, tuple[float, float]], scales: dict[str, float]) -> dict:
    """What the editor shows about a Generate: `SegmentationResult` in `src/lib/segmentation/backend.ts`."""
    return {
        "points": points,
        "seconds": segmentation.seconds,
        "inference": _inference_summary(segmentation.inference),
        "maps": {name: {"range": list(ranges[name]), "scale": scales[name]} for name in MAPS},
    }
