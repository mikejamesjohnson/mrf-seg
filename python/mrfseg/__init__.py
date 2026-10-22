"""MRFSeg: random-walker segmentation of CT images on learned affinities -- a Gaussian Markov
random field conditioned exactly on the user's points.

The core, in plain Python (numpy, VTK, PyTorch) with no user interface. The editor reaches it
through the local backend in `mrfseg.server`.
"""

import os

# Before torch is imported: an operation the Apple GPU lacks then runs on the CPU instead of
# failing. Set here because `device="auto"` is the core's choice, not just the server's.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from .basic import Basic, contrast_log_affinities  # noqa: E402
from .errors import InputError  # noqa: E402
from .extractors import EXTRACTORS, FEATURE_EXTRACTORS  # noqa: E402
from .features import FeatureExtractor, FeatureVolume  # noqa: E402
from .image_io import READABLE_SUFFIXES, ImageReadError, read_image  # noqa: E402
from .learned.extractor import Learned  # noqa: E402
from .rw import RandomWalkerResult, Seeds, SolverOptions, affinities, random_walker  # noqa: E402
from .segmentation import InferenceSummary, MRFSegmentation, Segmentation  # noqa: E402
from .settings import MRFSettings  # noqa: E402
from .surface import extract_surface  # noqa: E402

__all__ = [
    "Basic",
    "EXTRACTORS",
    "FEATURE_EXTRACTORS",
    "FeatureExtractor",
    "FeatureVolume",
    "ImageReadError",
    "InferenceSummary",
    "InputError",
    "Learned",
    "MRFSegmentation",
    "MRFSettings",
    "READABLE_SUFFIXES",
    "RandomWalkerResult",
    "Seeds",
    "Segmentation",
    "SolverOptions",
    "affinities",
    "contrast_log_affinities",
    "extract_surface",
    "random_walker",
    "read_image",
]
