"""MRFSeg: conditional-random-field segmentation of CT images, with mean-field inference.

The core, in plain Python (numpy, VTK, PyTorch) with no user interface. The editor reaches it
through the local backend in `mrfseg.server`.
"""

import os

# Before torch is imported: an operation the Apple GPU lacks then runs on the CPU instead of
# failing. Set here because `device="auto"` is the core's choice, not just the server's.
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

from .basic import Basic  # noqa: E402
from .errors import InputError  # noqa: E402
from .extractors import EXTRACTORS, FEATURE_EXTRACTORS  # noqa: E402
from .features import FeatureExtractor, FeatureVolume  # noqa: E402
from .handcrafted import HandcraftedParameters  # noqa: E402
from .image_io import READABLE_SUFFIXES, ImageReadError, read_image  # noqa: E402
from .learned.extractor import Learned  # noqa: E402
from .mrf import Clamps, MeanFieldResult, Potentials, free_energy, mean_field  # noqa: E402
from .segmentation import InferenceSummary, MRFSegmentation, Segmentation  # noqa: E402
from .settings import MRFSettings  # noqa: E402
from .surface import extract_surface  # noqa: E402

__all__ = [
    "Basic",
    "Clamps",
    "EXTRACTORS",
    "FEATURE_EXTRACTORS",
    "FeatureExtractor",
    "FeatureVolume",
    "HandcraftedParameters",
    "ImageReadError",
    "InferenceSummary",
    "InputError",
    "Learned",
    "MRFSegmentation",
    "MRFSettings",
    "MeanFieldResult",
    "Potentials",
    "READABLE_SUFFIXES",
    "Segmentation",
    "extract_surface",
    "free_energy",
    "mean_field",
    "read_image",
]
