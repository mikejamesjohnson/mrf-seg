"""Potentials by hand, from Basic features and the clicks: the classic interactive CRF, with
nothing trained. This is what the editor uses until the learned network (`learned/`) is there,
and the baseline it is measured against.

**Unary.** The clicks are a labelled sample of the image's voxels, and their features say what
lumen looks like *in this image*; the image itself says what background looks like, since
lumen is a fraction of a percent of it. Both are densities in feature space, compared voxel by
voxel as a likelihood ratio:

    p(phi | in)  = mean_{k in inside clicks}  N(phi; phi_k, kappa Sigma) * exp(-|p - p_k|^2 / 2 l_s^2)
    p(phi | bg)  = N(phi; mu, Sigma)                        -- fitted to a random sample of the image
    p(phi | out) = p(phi | bg), or its equal mixture with the outside clicks' kernels, placed like the inside ones

    a_i = a_0 + beta * [log p(phi_i | in) - log p(phi_i | out)]

Sigma is the covariance of the image's features (shrunk a little toward its diagonal), so the
kernels are Mahalanobis: a click reaches voxels that differ from it only in directions along
which the image varies anyway, and not in those where the click itself is unusual -- the
coarse-scale channels, where a thin vessel is dim and a chamber is not. The kernel width kappa
is l_f^2, in units of the image's own spread. A voxel like no click gets a very negative logit
(p(phi | in) falls to a floor, p(phi | bg) does not), so the unexplored image is background, as
a constant prior mean below zero made it in the GP editor; a_0 moves every voxel's logit, and
beta scales the evidence against the coupling. The spatial length scale l_s makes a click
local: with l_s large the unary is feature-only and a click anywhere speaks for every voxel
like it.

**Coupling.** Contrast-sensitive, in Hounsfield units, as Boykov & Jolly's: a label may change
cheaply where the image does,

    w_ij = lambda * (floor + (1 - floor) * exp(-(I_i - I_j)^2 / 2 delta^2)),

with `floor` in [0, 1] a share of the coupling that stays whatever the contrast, for smoothness
in flat regions. The grid is isotropic, so every edge has the same length and no distance factor
is needed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from .compute import Progress
from .errors import InputError
from .features import FeatureVolume
from .grid import Grid
from .mrf import AXES, Potentials

#: What the densities are floored at, so a voxel like no click has a finite logit.
DENSITY_FLOOR = 1e-6
#: Voxels of the feature grid the background Gaussian is fitted to.
BACKGROUND_SAMPLE = 4096
#: How far the background covariance is shrunk toward its diagonal.
SHRINKAGE = 0.1


@dataclass(frozen=True)
class HandcraftedParameters:
    """The numbers above, as the settings hold them."""

    feature_lengthscale: float
    """l_f: the clicks' kernel width, in standard deviations of the image's features (kappa = l_f^2)."""
    spatial_lengthscale: float
    """l_s, mm."""
    prior_logit: float
    """a_0: added to every unary logit; negative asks for more evidence before a voxel is inside."""
    unary_scale: float
    """beta: how strongly the clicks' evidence counts, against the coupling."""
    coupling: float
    """lambda: the coupling of an edge with no contrast across it."""
    contrast: float
    """delta, HU: how large an intensity step makes an edge cheap to cut."""
    coupling_floor: float
    """The share of lambda every edge keeps whatever its contrast, in [0, 1]."""


@dataclass
class Background:
    """The image's feature distribution, as a Gaussian: what everything is until a click says otherwise."""

    mean: torch.Tensor
    """(C,) float32."""
    whitening: torch.Tensor
    """(C, C) float32: L^-1 for Sigma = L L^T, so that |L^-1 (phi - mean)|^2 is the Mahalanobis distance."""

    @classmethod
    def of(cls, features: FeatureVolume, device: torch.device, seed: int = 0) -> Background:
        """Fitted to `BACKGROUND_SAMPLE` voxels of the feature grid, drawn the same way every time."""
        rng = np.random.default_rng(seed)
        flat = features.values.reshape(features.channels, -1)
        sample = torch.from_numpy(flat[:, rng.integers(0, flat.shape[1], BACKGROUND_SAMPLE)].T.astype(np.float64))
        if features.unit:
            sample = sample / sample.norm(dim=1, keepdim=True).clamp_min(1e-12)
        mean = sample.mean(0)
        covariance = torch.cov(sample.T) if sample.shape[0] > 1 else torch.eye(features.channels, dtype=torch.float64)
        covariance = (1 - SHRINKAGE) * covariance + SHRINKAGE * torch.diag(torch.diag(covariance))
        covariance += 1e-6 * torch.eye(features.channels, dtype=torch.float64)
        whitening = torch.linalg.inv(torch.linalg.cholesky(covariance))
        return cls(mean.float().to(device), whitening.float().to(device))

    def whiten(self, phi: torch.Tensor) -> torch.Tensor:
        """(n, C) features -> (n, C) whitened: zero mean, unit covariance under the background."""
        return (phi - self.mean) @ self.whitening.T


def kde_unary(
    features: FeatureVolume,
    grid: Grid,
    points: np.ndarray,
    labels: np.ndarray,
    parameters: HandcraftedParameters,
    device: torch.device,
    progress: Progress | None = None,
) -> torch.Tensor:
    """a_i at every voxel of the feature grid, (z, y, x) float32 on `device`, from `points`
    (n, 3) in world mm with `labels` +1 / -1. A feature slice at a time: the (voxels x clicks)
    kernel is the only large temporary."""
    p = parameters
    points = np.asarray(points, dtype=np.float64).reshape(-1, 3)
    labels = np.asarray(labels).reshape(-1)
    inside = torch.from_numpy(labels > 0).to(device)
    outside = ~inside
    if not bool(inside.any()):
        raise InputError("The handcrafted potentials need at least one inside point")
    background = Background.of(features, device)
    click_features = background.whiten(torch.from_numpy(features.at_ct_indices(grid.world_to_index(points))).to(device))
    click_positions = torch.from_numpy((points - grid.center).astype(np.float32)).to(device)
    channels = features.channels
    kappa = p.feature_lengthscale**2
    fz, fy, fx = features.shape
    unary = torch.empty((fz, fy, fx), dtype=torch.float32, device=device)
    positions_of = feature_slice_positions(features, grid, device)
    log_floor = math.log(DENSITY_FLOOR)
    n_inside, n_outside = int(inside.sum()), int(outside.sum())
    for k in range(fz):
        z = background.whiten(features.feature_slice(k, device).reshape(channels, -1).T)  # (fy * fx, C)
        log_background = -0.5 * (z * z).sum(1)  # log N(phi; mu, Sigma), less the constant every density shares
        # log N(phi; phi_k, kappa Sigma) times the spatial kernel, per click
        log_kernel = (
            -0.5 * _squared_distance(z, click_features) / kappa
            - 0.5 * channels * math.log(kappa)
            - 0.5 * _squared_distance(positions_of(k), click_positions) / p.spatial_lengthscale**2
        )
        log_in = torch.logsumexp(log_kernel[:, inside], 1) - math.log(n_inside)
        log_in = torch.logaddexp(log_in, torch.full_like(log_in, log_floor))
        if n_outside:
            log_out_clicks = torch.logsumexp(log_kernel[:, outside], 1) - math.log(n_outside)
            log_out = torch.logaddexp(log_background, log_out_clicks) - math.log(2)
        else:
            log_out = log_background
        log_out = torch.logaddexp(log_out, torch.full_like(log_out, log_floor))
        unary[k] = (p.prior_logit + p.unary_scale * (log_in - log_out)).reshape(fy, fx)
        if progress:
            progress((k + 1) / fz)
    return unary


def contrast_coupling(intensity: torch.Tensor, parameters: HandcraftedParameters) -> torch.Tensor:
    """w on every edge of the grid from `intensity` (z, y, x) in HU: (3, z, y, x) float32,
    `coupling[d]` for the edge to the neighbour along axis d (see `mrf.Potentials`)."""
    p = parameters
    if not 0 <= p.coupling_floor <= 1:
        raise InputError("coupling_floor must be between 0 and 1")
    coupling = torch.zeros((AXES, *intensity.shape), dtype=torch.float32, device=intensity.device)
    scale = -1.0 / (2 * p.contrast**2)
    for d in range(AXES):
        n = intensity.shape[d]
        if n < 2:
            continue
        step = intensity.narrow(d, 1, n - 1) - intensity.narrow(d, 0, n - 1)
        w = torch.exp(step.square_() * scale).mul_(1 - p.coupling_floor).add_(p.coupling_floor).mul_(p.coupling)
        coupling[d].narrow(d, 0, n - 1).copy_(w)
    return coupling


def handcrafted_potentials(
    features: FeatureVolume,
    grid: Grid,
    points: np.ndarray,
    labels: np.ndarray,
    parameters: HandcraftedParameters,
    device: torch.device,
    progress: Progress | None = None,
) -> Potentials:
    """The CRF's potentials from the features (and the intensity they were made from), the
    clicks and the parameters."""
    if features.intensity is None:
        raise InputError("The features carry no intensity grid for the contrast-sensitive coupling")
    unary = kde_unary(features, grid, points, labels, parameters, device, progress)
    intensity = torch.from_numpy(np.asarray(features.intensity, dtype=np.float32)).to(device)
    return Potentials(unary, contrast_coupling(intensity, parameters))


def feature_slice_positions(features: FeatureVolume, grid: Grid, device: torch.device):
    """k -> the world position of every voxel of feature slice k, relative to the grid's centre:
    (fy * fx, 3) float32 on `device`. The feature grid spans the CT's outermost voxel centres, so
    feature index f along an axis is CT index f / scale."""
    fx, fy, _ = features.dims
    per_ct = features.grid_scale()  # feature indices per CT index, (x, y, z)
    ct_per_feature = np.where(per_ct > 0, 1 / np.where(per_ct > 0, per_ct, 1), 0.0)
    j, i = torch.meshgrid(torch.arange(fy, dtype=torch.float64), torch.arange(fx, dtype=torch.float64), indexing="ij")
    steps = torch.from_numpy(grid.axes * grid.spacing * ct_per_feature)  # column a: one feature step along axis a, in mm
    base = torch.from_numpy(grid.origin - grid.center)
    in_plane = i.reshape(-1, 1) * steps[:, 0] + j.reshape(-1, 1) * steps[:, 1] + base
    return lambda k: (in_plane + k * steps[:, 2]).to(device=device, dtype=torch.float32)


def _squared_distance(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """|a_i - b_j|^2 for (m, d) and (n, d): (m, n)."""
    return ((a * a).sum(1, keepdim=True) + (b * b).sum(1)[None, :] - 2 * a @ b.T).clamp_min(0)
