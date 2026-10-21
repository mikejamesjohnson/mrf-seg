"""Synthetic images to segment, with the answer known.

`make_cpr_phantom` makes what a curved planar reformat (CPR) of a CT shows of one vessel: the
vessel runs along z, from z = 0 to its length, roughly but not exactly along the axis, as a
reformat about an imperfect centreline leaves it, with x and y across it. Side branches leave it
at an angle and run out of the volume, and it can have a stenosis, a soft plaque and a
calcification. Everything is in Hounsfield units and world mm (LPS), and every random choice
comes from `CPRPhantomConfig.seed`.

How the image is formed, in order:

1. **Partial volume.** Each voxel is the mean of `supersample`^3 points inside it, each given
   the HU of the material there: fat, myocardium, vessel wall, plaque, contrast-filled lumen or
   calcium, in increasing precedence.
2. **Texture.** Slowly varying HU, as fat and myocardium are not uniform, everywhere but the
   lumen and the calcium.
3. **Blur.** A Gaussian point spread function, `blur_fwhm` across, which is also what makes a
   calcification bloom.
4. **Noise.** Gaussian, with a CT noise power spectrum: NPS(f) ~ f exp(-(f / f0)^2), the shape
   filtered back projection gives it, peaking at `noise_peak` cycles/mm. Its standard deviation
   is `noise_std`.
5. **Quantisation** to whole HU, as int16.

The label is the lumen, the main vessel's and the branches': a voxel more than half inside it
is 1. Plaque, wall and calcium are not lumen.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter, gaussian_filter1d

from .image_io import write_metaimage

Range = tuple[float, float]


@dataclass(frozen=True)
class CPRPhantomConfig:
    """Everything that decides a synthetic CPR volume.

    A `Range` is (low, high), drawn from uniformly for each volume; give (v, v) for a fixed
    value. Lengths are mm, angles degrees, intensities HU.
    """

    seed: int | None = 0
    """Every random choice follows from this; None draws a new volume each time. Geometry,
    texture and noise each have their own stream, so changing the noise keeps the vessel."""

    # The grid ------------------------------------------------------------------------------
    length: float = 10.0
    """Along z, the vessel's axis: voxel centres run from z = 0 to the one nearest z = length."""
    spacing: float | tuple[float, float, float] = 0.1
    """The voxel size: one for every axis, or (x, y, z)."""
    width_factor: float = 3.0
    """Half the width across the vessel, as a multiple of its largest lumen radius: x and y
    run over [-width_factor * r_max, width_factor * r_max]."""
    half_width: float | None = None
    """Half the width across the vessel in mm instead, the same for every volume."""

    # The vessel ----------------------------------------------------------------------------
    radius: Range = (0.5, 1.5)
    """The healthy lumen radius stays within this. It tapers from one end to the other, the
    start drawn from the upper part of the range and the end from below the start."""
    radius_wobble: float = 0.08
    """The radius' smooth variation about its taper, as a fraction of it (standard deviation)."""
    radius_wobble_length: float = 2.0
    """How quickly that variation changes along the vessel."""
    centerline_offset: float = 0.25
    """How far the vessel's centre wanders from the axis (standard deviation, per axis), as a
    reformat's imperfect centreline leaves it."""
    centerline_wander_length: float = 3.0
    """How quickly that wander changes along the vessel."""
    centerline_reach: float | None = None
    """Instead of `centerline_offset`: the farthest the centre gets from the axis, as a fraction
    of the vessel's mean healthy lumen radius. The wander is the same smooth path, moved to be
    about the axis on average and scaled so its farthest point is exactly there, so scrolling
    along the vessel shows its cross-section drift from one side of the axis to the other."""
    wall_thickness: float = 0.25
    """The vessel wall around the lumen; a branch's is 0.6 of this."""

    # Side branches -------------------------------------------------------------------------
    branches: tuple[int, int] = (1, 2)
    """How many, from low to high inclusive."""
    branch_radius: Range = (0.3, 0.6)
    """A branch's lumen radius, as a fraction of the vessel's where it leaves."""
    branch_angle: Range = (35.0, 75.0)
    """Between a branch and the vessel's axis, leaving towards larger z."""

    # Lesions, each at its own place along the vessel --------------------------------------
    stenosis: bool = True
    stenosis_severity: Range = (0.4, 0.6)
    """The fraction of the lumen radius it takes at its narrowest, all round (concentric)."""
    stenosis_length: Range = (1.0, 2.0)
    soft_plaque: bool = True
    soft_plaque_thickness: Range = (0.3, 0.5)
    """How far it pushes the lumen in at its thickest, as a fraction of the lumen radius."""
    soft_plaque_length: Range = (1.5, 2.5)
    soft_plaque_arc: Range = (90.0, 180.0)
    """How much of the circumference it covers."""
    calcification: bool = True
    calcification_size: Range = (0.3, 0.5)
    """Its radial half-thickness, as a fraction of the lumen radius there; it is 1.2 times that
    around the vessel and 1.5-2.5 times along it."""
    calcification_reach: Range = (0.0, 0.5)
    """How far into the lumen it reaches, as a fraction of its half-thickness."""

    # Intensities ---------------------------------------------------------------------------
    lumen_hu: Range = (350.0, 450.0)
    """Contrast-filled lumen, at z = 0."""
    lumen_falloff: float = 0.1
    """The fraction of that lost by z = length: contrast fades along a vessel."""
    wall_hu: float = 40.0
    fibrous_hu: float = 90.0
    """The plaque that makes the stenosis."""
    soft_plaque_hu: float = 30.0
    calcification_hu: Range = (600.0, 1000.0)
    fat_hu: float = -90.0
    myocardium: bool = True
    """A wall of myocardium along one side, as beside a coronary artery."""
    myocardium_hu: float = 50.0
    myocardium_gap: Range = (0.6, 1.2)
    """Between the vessel wall, where it comes closest, and the myocardium."""
    texture: float = 12.0
    """The HU standard deviation of the slow variation in fat and myocardium."""
    texture_length: float = 0.8
    """Its length scale."""

    # Image formation -----------------------------------------------------------------------
    supersample: int = 3
    """Points per voxel along each axis, for partial volume."""
    blur_fwhm: float = 0.5
    """The point spread function's full width at half maximum; 0 for none."""
    noise_std: float = 30.0
    """0 for none."""
    noise_peak: float = 0.5
    """Cycles per mm at which the noise power spectrum peaks: lower is coarser noise."""

    @property
    def voxel_size(self) -> tuple[float, float, float]:
        """`spacing` as (x, y, z)."""
        s = self.spacing
        return (float(s),) * 3 if isinstance(s, (int, float)) else tuple(float(v) for v in s)

    def __post_init__(self) -> None:
        positive = ("length", "width_factor", "radius_wobble_length", "centerline_wander_length", "texture_length")
        for name in positive:
            if not getattr(self, name) > 0:
                raise ValueError(f"{name} must be positive, not {getattr(self, name)!r}")
        if isinstance(self.spacing, (int, float)) or len(self.spacing) == 3:
            if not all(v > 0 for v in self.voxel_size):
                raise ValueError(f"spacing must be positive, not {self.spacing!r}")
        else:
            raise ValueError(f"spacing must be one number or (x, y, z), not {self.spacing!r}")
        if self.half_width is not None and not self.half_width > 0:
            raise ValueError(f"half_width must be positive or None, not {self.half_width!r}")
        if self.centerline_reach is not None and not self.centerline_reach >= 0:
            raise ValueError(f"centerline_reach must not be negative, not {self.centerline_reach!r}")
        for name in ("radius_wobble", "centerline_offset", "wall_thickness", "texture", "blur_fwhm", "noise_std", "lumen_falloff"):
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must not be negative, not {getattr(self, name)!r}")
        if self.noise_peak <= 0:
            raise ValueError(f"noise_peak must be positive, not {self.noise_peak!r}")
        if not (isinstance(self.supersample, int) and self.supersample >= 1):
            raise ValueError(f"supersample must be a whole number of at least 1, not {self.supersample!r}")
        if not (0 < self.radius[0] <= self.radius[1]):
            raise ValueError(f"radius must be (low, high) with 0 < low <= high, not {self.radius!r}")
        if not (0 <= self.branches[0] <= self.branches[1]):
            raise ValueError(f"branches must be (low, high) with 0 <= low <= high, not {self.branches!r}")
        for name in ("stenosis_severity", "soft_plaque_thickness"):
            low, high = getattr(self, name)
            if not (0 <= low <= high < 1):
                raise ValueError(f"{name} must be fractions (low, high) below 1, not {(low, high)!r}")


@dataclass
class Branch:
    z: float
    """Where it leaves the vessel's centreline."""
    origin: np.ndarray
    """That point, (x, y, z)."""
    direction: np.ndarray
    """Unit (x, y, z)."""
    radius: float
    angle: float
    """From the vessel's axis, degrees."""


@dataclass
class Lesion:
    kind: str
    """"stenosis", "soft plaque" or "calcification"."""
    z: float
    """Its centre along the vessel."""
    length: float
    """Along the vessel."""
    details: dict[str, float] = field(default_factory=dict)


@dataclass
class CPRPhantom:
    """A synthetic CPR volume, its label, and the geometry they were made from."""

    config: CPRPhantomConfig
    image: np.ndarray
    """(z, y, x) int16 HU."""
    label: np.ndarray
    """(z, y, x) uint8: 1 where more than half the voxel is lumen."""
    lumen_fraction: np.ndarray
    """(z, y, x) float32: how much of each voxel is lumen."""
    spacing: tuple[float, float, float]
    """(x, y, z) mm."""
    origin: tuple[float, float, float]
    """Voxel (0, 0, 0)'s centre, (x, y, z) mm."""
    z: np.ndarray
    """The voxel centres along the vessel."""
    centerline: np.ndarray
    """(nz, 2): the lumen's centre, (x, y), at each of them."""
    radius: np.ndarray
    """(nz,): the healthy lumen radius at each of them; lesions narrow it."""
    branches: list[Branch]
    lesions: list[Lesion]

    @property
    def dimensions(self) -> tuple[int, int, int]:
        """(x, y, z) voxels, as VTK and MetaImage count them."""
        nz, ny, nx = self.image.shape
        return nx, ny, nz

    def summary(self) -> str:
        c, (nx, ny, nz) = self.config, self.dimensions
        size = f"{self.spacing[0]:g}" if len(set(self.spacing)) == 1 else " x ".join(f"{v:g}" for v in self.spacing)
        lines = [
            f"{nx} x {ny} x {nz} voxels of {size} mm: x in +-{-self.origin[0]:.2f}, y in +-{-self.origin[1]:.2f}, "
            f"z 0-{self.z[-1]:g} mm; seed {c.seed}",
            f"lumen radius {self.radius[0]:.2f} -> {self.radius[-1]:.2f} mm (largest {self.radius.max():.2f}); "
            f"centre within {np.hypot(*self.centerline.T).max():.2f} mm of the axis",
        ]
        for branch in self.branches:
            lines.append(f"branch at z = {branch.z:.1f} mm: radius {branch.radius:.2f} mm, {branch.angle:.0f} deg from the axis")
        for lesion in self.lesions:
            details = ", ".join(f"{key} {value:.3g}" for key, value in lesion.details.items())
            lines.append(f"{lesion.kind} at z = {lesion.z:.1f} mm, {lesion.length:.1f} mm long: {details}")
        lines.append(f"{int(self.label.sum())} lumen voxels; HU {int(self.image.min())} to {int(self.image.max())}")
        return "\n".join(lines)


def make_cpr_phantom(config: CPRPhantomConfig = CPRPhantomConfig()) -> CPRPhantom:
    """A synthetic CPR volume of one vessel, its lumen label, and its geometry."""
    geometry_rng, texture_rng, noise_rng = (np.random.default_rng(s) for s in np.random.SeedSequence(config.seed).spawn(3))
    c, m = config, config.supersample
    sx, sy, sz = c.voxel_size
    inside = ((np.arange(m) + 0.5) / m - 0.5)  # where a voxel's points are, in voxels from its centre

    # Along the vessel: voxel centres, and the planes of points inside them.
    nz = int(round(c.length / sz)) + 1
    z = np.arange(nz) * sz
    z_fine = (z[:, None] + sz * inside[None, :]).ravel()

    vessel = _Vessel.draw(c, geometry_rng, z_fine)
    half = c.half_width if c.half_width is not None else c.width_factor * float(vessel.radius.max())
    x_axis = np.arange(-max(1, round(half / sx)), max(1, round(half / sx)) + 1) * sx
    y_axis = np.arange(-max(1, round(half / sy)), max(1, round(half / sy)) + 1) * sy
    myocardium = _draw_myocardium(c, geometry_rng, vessel)
    branches = _draw_branches(c, geometry_rng, vessel, myocardium)

    # Partial volume: each voxel the mean of its points' materials.
    hu = np.zeros((nz, len(y_axis), len(x_axis)), dtype=np.float64)
    lumen = np.zeros_like(hu)
    still = np.zeros_like(hu)  # lumen or calcium: no texture there
    x, y = np.meshgrid((x_axis[:, None] + sx * inside).ravel(), (y_axis[:, None] + sy * inside).ravel(), indexing="xy")  # (y, x)
    for plane, z_at in enumerate(z_fine):
        plane_hu, plane_lumen, plane_still = _materials(c, vessel, branches, myocardium, plane, z_at, x, y)
        k = plane // m
        hu[k] += _block_mean(plane_hu, m)
        lumen[k] += _block_mean(plane_lumen, m)
        still[k] += _block_mean(plane_still, m)
    hu /= m
    lumen /= m
    still /= m

    per_axis = (sz, sy, sx)  # in the arrays' (z, y, x) order
    if c.texture > 0:
        hu += c.texture * _smooth_field(texture_rng, hu.shape, tuple(c.texture_length / v for v in per_axis)) * (1 - still)
    if c.blur_fwhm > 0:
        sigma = c.blur_fwhm / (2 * math.sqrt(2 * math.log(2)))
        hu = gaussian_filter(hu, sigma=tuple(sigma / v for v in per_axis), mode="nearest")
    if c.noise_std > 0:
        hu += c.noise_std * _ct_noise(noise_rng, hu.shape, per_axis, c.noise_peak)
    image = np.clip(np.rint(hu), -1024, 3071).astype(np.int16)

    voxel = slice(m // 2, None, m)  # the plane through each voxel's centre (odd m), or next to it
    return CPRPhantom(
        config=config,
        image=image,
        label=(lumen > 0.5).astype(np.uint8),
        lumen_fraction=lumen.astype(np.float32),
        spacing=(sx, sy, sz),
        origin=(float(x_axis[0]), float(y_axis[0]), 0.0),
        z=z,
        centerline=vessel.centre[voxel],
        radius=vessel.radius[voxel],
        branches=branches,
        lesions=vessel.lesions,
    )


def write_cpr_phantom(phantom: CPRPhantom, stem: str | Path) -> tuple[Path, Path]:
    """`<stem>.mha`, the image, and `<stem>_label.mha`, its label; their paths."""
    stem = Path(stem)
    image = write_metaimage(stem.with_name(stem.name + ".mha"), phantom.image, phantom.origin, phantom.spacing)
    label = write_metaimage(stem.with_name(stem.name + "_label.mha"), phantom.label, phantom.origin, phantom.spacing)
    return image, label


# --------------------------------------------------------------------------------------------- #
# Geometry
# --------------------------------------------------------------------------------------------- #


def _draw(rng: np.random.Generator, span: Range) -> float:
    low, high = span
    return float(low if low == high else rng.uniform(low, high))


def _smooth(rng: np.random.Generator, x: np.ndarray, length: float, span: float) -> np.ndarray:
    """A smooth random function of position along [0, `span`], at `x`: zero mean, unit standard
    deviation, varying over about `length`. Drawn on a grid of its own, so the same seed gives
    the same function whatever the voxels it is sampled at."""
    step = length / 20
    grid = np.arange(-4 * length - 1, span + 4 * length + 1 + step, step)
    sigma = length / step
    smooth = gaussian_filter1d(rng.standard_normal(len(grid)), sigma, mode="wrap")
    smooth *= math.sqrt(2 * math.sqrt(math.pi) * sigma)  # the filter's own loss of variance
    return np.interp(x, grid, smooth)


def _bump(z: np.ndarray, centre: float, length: float) -> np.ndarray:
    """1 at `centre`, falling smoothly to 0 at `length` / 2 either side."""
    u = (z - centre) / length
    return np.where(np.abs(u) < 0.5, np.cos(math.pi * u) ** 2, 0.0)


@dataclass
class _Vessel:
    """The main vessel at every plane of points: centre, healthy radius, and its lesions."""

    z: np.ndarray  # (n,): the planes
    centre: np.ndarray  # (n, 2)
    radius: np.ndarray  # (n,)
    stenosed: np.ndarray  # (n,): the radius the stenosis leaves
    plaque: np.ndarray  # (n,): how far the soft plaque pushes in, at its thickest
    plaque_angle: float
    plaque_arc: float
    calcium: tuple[np.ndarray, np.ndarray, np.ndarray] | None  # centre, axes (3 x 3, rows), half-sizes
    calcium_hu: float
    lumen_hu: np.ndarray  # (n,)
    lesions: list[Lesion]

    @classmethod
    def draw(cls, c: CPRPhantomConfig, rng: np.random.Generator, z: np.ndarray) -> _Vessel:
        low, high = c.radius
        start = _draw(rng, (low + 0.4 * (high - low), high))
        end = _draw(rng, (low, start))
        taper = start + (end - start) * np.clip(z / c.length, 0, 1)
        radius = np.clip(taper * (1 + c.radius_wobble * _smooth(rng, z, c.radius_wobble_length, c.length)), low, high)
        wander = np.stack([_smooth(rng, z, c.centerline_wander_length, c.length) for _ in range(2)], axis=1)
        if c.centerline_reach is None:
            centre = c.centerline_offset * wander
        else:  # the same path, scaled: no further random draws, so the rest of the volume is unchanged
            within = (z >= 0) & (z <= c.length)
            wander = wander - wander[within].mean(axis=0)  # about the axis: the centreline is right on average
            farthest = float(np.hypot(*wander[within].T).max())
            centre = wander * (c.centerline_reach * float(radius[within].mean()) / max(farthest, 1e-12))
        lumen_hu = _draw(rng, c.lumen_hu) * (1 - c.lumen_falloff * np.clip(z / c.length, 0, 1))

        # Where each lesion goes: apart, and clear of the ends.
        wanted = []
        if c.stenosis:
            wanted.append(("stenosis", _draw(rng, c.stenosis_length)))
        if c.soft_plaque:
            wanted.append(("soft plaque", _draw(rng, c.soft_plaque_length)))
        calcium_fraction = _draw(rng, c.calcification_size)
        calcium_stretch = _draw(rng, (1.5, 2.5))
        if c.calcification:
            # Room for it at the largest radius allowed: where it goes, and so its size, is not known yet.
            wanted.append(("calcification", 2 * calcium_stretch * calcium_fraction * high))
        places = _place(rng, [length for _, length in wanted], c.length)

        stenosed, plaque = radius.copy(), np.zeros_like(radius)
        plaque_angle = plaque_arc = 0.0
        calcium, calcium_hu = None, 0.0
        lesions = []
        for (kind, length), at in zip(wanted, places):
            index = int(np.argmin(np.abs(z - at)))
            if kind == "stenosis":
                severity = _draw(rng, c.stenosis_severity)
                stenosed = radius * (1 - severity * _bump(z, at, length))
                lesions.append(Lesion(kind, at, length, {"severity": severity, "fibrous HU": c.fibrous_hu}))
            elif kind == "soft plaque":
                thickness = _draw(rng, c.soft_plaque_thickness)
                plaque = thickness * radius * _bump(z, at, length)
                plaque_angle, plaque_arc = rng.uniform(-math.pi, math.pi), math.radians(_draw(rng, c.soft_plaque_arc))
                lesions.append(Lesion(kind, at, length, {"thickness": thickness, "arc deg": math.degrees(plaque_arc), "HU": c.soft_plaque_hu}))
            else:
                calcium_size = calcium_fraction * float(radius[index])
                length = 2 * calcium_stretch * calcium_size
                angle = rng.uniform(-math.pi, math.pi)
                outward = np.array([math.cos(angle), math.sin(angle), 0.0])
                around = np.array([-math.sin(angle), math.cos(angle), 0.0])
                reach = _draw(rng, c.calcification_reach) * calcium_size
                middle = np.array([*centre[index], at]) + outward * (radius[index] + calcium_size - reach)
                sizes = np.array([calcium_size, 1.2 * calcium_size, length / 2])
                calcium = (middle, np.stack([outward, around, [0.0, 0.0, 1.0]]), sizes)
                calcium_hu = _draw(rng, c.calcification_hu)
                lesions.append(Lesion(kind, at, length, {"half-thickness mm": calcium_size, "HU": calcium_hu}))
        return cls(z, centre, radius, stenosed, plaque, plaque_angle, plaque_arc, calcium, calcium_hu, lumen_hu, lesions)


def _place(rng: np.random.Generator, lengths: list[float], span: float, margin: float = 0.5, gap: float = 0.3) -> list[float]:
    """Centres for things of `lengths` along [0, span], none overlapping."""
    for _ in range(2000):
        centres = [rng.uniform(margin + length / 2, max(margin + length / 2, span - margin - length / 2)) for length in lengths]
        if all(
            abs(centres[i] - centres[j]) >= (lengths[i] + lengths[j]) / 2 + gap for i in range(len(lengths)) for j in range(i)
        ):
            return centres
    raise ValueError(f"Lesions {', '.join(f'{v:.1f}' for v in lengths)} mm long do not fit apart along {span:g} mm: shorten them or turn some off")


def _draw_branches(c: CPRPhantomConfig, rng: np.random.Generator, vessel: _Vessel, myocardium: _Myocardium | None) -> list[Branch]:
    count = int(rng.integers(c.branches[0], c.branches[1] + 1))
    lesions = [(lesion.z, lesion.length) for lesion in vessel.lesions]
    branches = []
    # Away from the myocardium: at least 60 degrees from heading into it.
    away = math.atan2(-myocardium.normal[1], -myocardium.normal[0]) if myocardium is not None else 0.0
    spread = math.radians(120) if myocardium is not None else math.pi
    for _ in range(count):
        # Clear of the lesions and of the other branches, if there is room; anywhere otherwise.
        for _ in range(200):
            at = rng.uniform(1.0, max(1.0, c.length - 1.0))
            if all(abs(at - z_l) > length / 2 + 0.3 for z_l, length in lesions) and all(abs(at - b.z) > 1.5 for b in branches):
                break
        index = int(np.argmin(np.abs(vessel.z - at)))
        angle = _draw(rng, c.branch_angle)
        heading = away + rng.uniform(-spread, spread)
        tilt = math.radians(angle)
        direction = np.array([math.sin(tilt) * math.cos(heading), math.sin(tilt) * math.sin(heading), math.cos(tilt)])
        origin = np.array([*vessel.centre[index], at])
        radius = _draw(rng, c.branch_radius) * float(vessel.radius[index])
        branches.append(Branch(at, origin, direction, radius, angle))
    return branches


@dataclass
class _Myocardium:
    normal: np.ndarray  # (2,) towards it
    distance: np.ndarray  # (n,) from the axis, at each plane


def _draw_myocardium(c: CPRPhantomConfig, rng: np.random.Generator, vessel: _Vessel) -> _Myocardium | None:
    if not c.myocardium:
        return None
    heading = rng.uniform(-math.pi, math.pi)
    normal = np.array([math.cos(heading), math.sin(heading)])
    outer = vessel.centre @ normal + vessel.radius + c.wall_thickness
    # A gently undulating surface, never closer than the gap.
    distance = float(outer.max()) + _draw(rng, c.myocardium_gap) + 0.1 * np.abs(_smooth(rng, vessel.z, 2.0, c.length))
    return _Myocardium(normal, distance)


# --------------------------------------------------------------------------------------------- #
# Image formation
# --------------------------------------------------------------------------------------------- #


def _materials(c, vessel: _Vessel, branches: list[Branch], myocardium: _Myocardium | None, plane: int, z: float, x, y):
    """HU, lumen and lumen-or-calcium at a plane of points (y, x) at height `z`."""
    hu = np.full(x.shape, c.fat_hu)
    if myocardium is not None:
        hu[x * myocardium.normal[0] + y * myocardium.normal[1] > myocardium.distance[plane]] = c.myocardium_hu

    dx, dy = x - vessel.centre[plane, 0], y - vessel.centre[plane, 1]
    rho = np.hypot(dx, dy)
    healthy, stenosed = vessel.radius[plane], vessel.stenosed[plane]
    inner = np.full(x.shape, stenosed)
    if vessel.plaque[plane] > 0:
        turn = np.angle(np.exp(1j * (np.arctan2(dy, dx) - vessel.plaque_angle)))
        crescent = np.where(np.abs(turn) < vessel.plaque_arc / 2, np.cos(math.pi * turn / vessel.plaque_arc) ** 2, 0.0)
        inner = stenosed - vessel.plaque[plane] * crescent

    wall = rho < healthy + c.wall_thickness
    lumen = rho < inner
    for branch in branches:
        reach = _segment_distance(x, y, z, branch)
        if reach is not None:
            wall |= reach < branch.radius + 0.6 * c.wall_thickness
            lumen |= reach < branch.radius

    hu[wall] = c.wall_hu
    hu[(rho < healthy) & (rho >= stenosed)] = c.fibrous_hu
    hu[(rho < stenosed) & (rho >= inner)] = c.soft_plaque_hu
    still = lumen.copy()
    if vessel.calcium is not None:
        middle, axes, sizes = vessel.calcium
        if abs(z - middle[2]) < sizes[2]:
            d = np.stack([x - middle[0], y - middle[1], np.full(x.shape, z - middle[2])], axis=-1)
            calcium = (((d @ axes.T) / sizes) ** 2).sum(-1) <= 1
            lumen &= ~calcium
            still |= calcium
            hu[calcium] = vessel.calcium_hu
    hu[lumen] = vessel.lumen_hu[plane]
    return hu, lumen.astype(np.float64), still.astype(np.float64)


def _segment_distance(x, y, z: float, branch: Branch) -> np.ndarray | None:
    """Distance from each point of a plane to the branch's axis, from where it leaves the vessel
    on; None if the branch does not reach this plane."""
    if z < branch.origin[2] - branch.radius - 1.0:
        return None
    d = branch.direction
    px, py, pz = x - branch.origin[0], y - branch.origin[1], z - branch.origin[2]
    t = np.clip(px * d[0] + py * d[1] + pz * d[2], 0, None)
    return np.sqrt((px - t * d[0]) ** 2 + (py - t * d[1]) ** 2 + (pz - t * d[2]) ** 2)


def _block_mean(values: np.ndarray, m: int) -> np.ndarray:
    """The mean of each m x m block: a plane of points down to a plane of voxels."""
    ny, nx = values.shape
    return values.reshape(ny // m, m, nx // m, m).mean(axis=(1, 3))


def _smooth_field(rng: np.random.Generator, shape: tuple[int, ...], sigma: tuple[float, ...]) -> np.ndarray:
    """A smooth random field, zero mean and unit standard deviation, varying over `sigma` voxels."""
    field = gaussian_filter(rng.standard_normal(shape), sigma=sigma, mode="wrap")
    return (field - field.mean()) / (field.std() or 1.0)


def _ct_noise(rng: np.random.Generator, shape: tuple[int, ...], spacing: tuple[float, ...], peak: float) -> np.ndarray:
    """Unit-variance Gaussian noise with a CT-like power spectrum, f exp(-(f / f0)^2), peaking at
    `peak` cycles/mm (f0 = sqrt(2) peak): no power at zero frequency, as filtered back
    projection leaves none. `shape` and `spacing` are (z, y, x)."""
    nz, ny, nx = shape
    fz = np.fft.fftfreq(nz, d=spacing[0])[:, None, None]
    fy = np.fft.fftfreq(ny, d=spacing[1])[None, :, None]
    fx = np.fft.rfftfreq(nx, d=spacing[2])[None, None, :]
    f = np.sqrt(fx * fx + fy * fy + fz * fz)
    f0 = math.sqrt(2) * peak
    spectrum = np.fft.rfftn(rng.standard_normal(shape)) * np.sqrt(f * np.exp(-((f / f0) ** 2)))
    noise = np.fft.irfftn(spectrum, s=shape, axes=(0, 1, 2))
    return noise / (noise.std() or 1.0)
