"""Training items from the cache: a patch of a scan, its lumen label, and a few simulated clicks.

A patch is `PatchConfig.size` voxels a side, cut from a scan in the cache (`prepare.py`). Most
patches are centred near a labelled voxel, offset by up to `jitter` so the vessel is not always
in the middle; the rest fall anywhere in a box around the labelled voxels, for background. The
target is the binary lumen label on the patch (every segment label collapsed to 1), which is
what the editor segments.

The clicks are what a user might have placed on the patch by the time the network is asked
(`clicks.random_clicks`): a few inside, spread over the lumen's connected components so a small
branch gets its share, and a few outside -- some on a vessel that got no inside click, where a
user clicks to drop what is not wanted, the rest mostly within `near_wall` mm of the lumen,
where a user's outside clicks go. How many of each is drawn per item, from none at all (the
network must segment with no help, as the editor lets it) to `max_inside` and `max_outside`.
The network sees the clicks as two input channels (`learned.click_maps`), and the target is
conditioned on them (`clicks.conditioned_target`): with inside clicks, the lumen connected to
them and nothing else; without, all of it; less a vessel with an outside click and no inside
one. The clicks are clamped during the mean-field inference the loss is taken through as well.

Augmentation: the 48 symmetries of the cube (flips and axis permutations; the grid is isotropic
and anatomy's orientation means nothing to the potentials) and a little HU scale and offset
jitter. The clicks are drawn after the symmetry, on the label as augmented. Without
augmentation, item i is the same patch and clicks every time.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from . import network_input
from .clicks import conditioned_target, random_clicks
from .prepare import read_index, read_splits

SIX = ndimage.generate_binary_structure(3, 1)  # 6-connectivity, as the CRF's edges
#: Clicks are padded to this many rows per item, label 0 marking the padding.
MAX_CLICKS = 8


@dataclass(frozen=True)
class PatchConfig:
    size: int = 96                      # voxels a side; a multiple of 16
    max_inside: int = 3                 # inside clicks per patch, at most (drawn uniformly from 0)
    max_outside: int = 4                # outside clicks per patch, at most (drawn uniformly from 0)
    no_click_share: float = 0.2         # of patches given no clicks at all, whatever the draw
    near_wall: float = 2.0              # mm: how close to the lumen an outside click near the wall is
    near_wall_share: float = 2.0 / 3.0  # of the outside clicks not on a vessel, near the wall
    on_lumen_share: float = 0.3         # of the outside clicks, on a vessel that got no inside click
    margin: int = 4                     # voxels a click keeps from the patch's faces
    vessel_share: float = 0.8           # of patches placed near a labelled voxel
    jitter: int = 32                    # voxels: how far that voxel may be from the patch's centre
    heart_margin: int = 48              # voxels around the labelled voxels' box the other patches may reach

    def __post_init__(self) -> None:
        if self.max_inside + self.max_outside > MAX_CLICKS:
            raise ValueError(f"at most {MAX_CLICKS} clicks per patch")


class Case:
    """One cached scan, read lazily and memory-mapped, so a data-loader worker holds no copy.

    The maps are opened for each access and dropped with the patch cut from them (`cut` copies):
    a memory map holds a file descriptor for as long as it lives, and 560 scans times two maps
    in each of several workers is far more than a process is allowed open. The page cache makes
    opening a file again cheap.
    """

    def __init__(self, cache: Path, meta: dict) -> None:
        self.cache, self.meta = Path(cache), meta
        self.id: str = meta["id"]
        self.shape = tuple(meta["shape"])
        self.spacing = float(meta["spacing"])
        self._lumen = None

    @property
    def image(self) -> np.ndarray:
        """(z, y, x) int16 HU, memory-mapped: read what you need and let it go."""
        return np.load(self.cache / f"{self.id}.image.npy", mmap_mode="r")

    @property
    def label(self) -> np.ndarray:
        """(z, y, x) uint8, memory-mapped, likewise."""
        return np.load(self.cache / f"{self.id}.label.npy", mmap_mode="r")

    @property
    def lumen(self) -> np.ndarray:
        """(n, 3) int32: the (z, y, x) indices of the labelled voxels."""
        if self._lumen is None:
            self._lumen = np.load(self.cache / f"{self.id}.lumen.npy")
        return self._lumen

    def world_to_index(self, points: np.ndarray) -> np.ndarray:
        """Continuous (z, y, x) indices of (n, 3) LPS mm points (x, y, z)."""
        origin = np.asarray(self.meta["origin"], dtype=float)
        direction = np.asarray(self.meta["direction"], dtype=float).reshape(3, 3)
        xyz = ((np.asarray(points, dtype=float) - origin) @ direction) / self.spacing
        return xyz[:, ::-1]


def load_cases(cache: str | Path, split: str | None = None, limit: int | None = None) -> list[Case]:
    """The cache's scans, in `split` (train / val / test) or all of them."""
    cache = Path(cache)
    index = read_index(cache)
    ids = read_splits(cache)[split] if split else sorted(index["cases"])
    return [Case(cache, index["cases"][i]) for i in ids if i in index["cases"]][:limit]


# ------------------------------------------------------------------ one patch

def patch_origin(case: Case, cfg: PatchConfig, rng: np.random.Generator) -> np.ndarray:
    """Where a patch starts (z, y, x): near a labelled voxel, or anywhere in the box around them."""
    shape, size = np.array(case.shape), cfg.size
    upper = np.maximum(shape - size, 0)
    if len(case.lumen) and rng.random() < cfg.vessel_share:
        centre = case.lumen[rng.integers(len(case.lumen))] + rng.integers(-cfg.jitter, cfg.jitter + 1, size=3)
        origin = centre - size // 2
    elif len(case.lumen):
        low = np.maximum(case.lumen.min(0) - cfg.heart_margin, 0)
        high = np.minimum(case.lumen.max(0) + cfg.heart_margin, shape - 1)
        origin = np.array([rng.integers(lo, max(lo, hi - size) + 1) for lo, hi in zip(low, high)])
    else:
        origin = np.array([rng.integers(0, u + 1) for u in upper])
    return np.clip(origin, 0, upper)


def cut(volume: np.ndarray, origin: np.ndarray, size: int, fill) -> np.ndarray:
    """`volume[origin : origin + size]` per axis, padded with `fill` where the volume ends. A
    copy, so a memory-mapped `volume` can be closed once this returns."""
    stop = origin + size
    region = tuple(slice(int(o), int(min(s, n))) for o, s, n in zip(origin, stop, volume.shape))
    block = np.array(volume[region])
    pad = [(0, size - (sl.stop - sl.start)) for sl in region]
    return np.pad(block, pad, mode="constant", constant_values=fill) if any(p[1] for p in pad) else block


def symmetry(arrays: list[np.ndarray], rng: np.random.Generator) -> list[np.ndarray]:
    """The same random flips and axis permutation for every array: the cube's 48 symmetries."""
    flips = [slice(None, None, -1) if rng.random() < 0.5 else slice(None) for _ in range(3)]
    order = rng.permutation(3)
    return [np.ascontiguousarray(a[tuple(flips)].transpose(order)) for a in arrays]


def make_item(case: Case, cfg: PatchConfig, rng: np.random.Generator, augment: bool) -> dict:
    """One training item from `case`: the network's input (IN_CHANNELS, s, s, s) float32 (the
    normalised patch and the click maps), the target (s, s, s) uint8 (the lumen as the clicks
    condition it, `clicks.conditioned_target`), the clicks (MAX_CLICKS, 4) int64 rows of
    (z, y, x, label) padded with label 0, and where the patch came from."""
    origin = patch_origin(case, cfg, rng)
    hu = cut(case.image, origin, cfg.size, fill=int(case.meta.get("hu_range", [-1024])[0])).astype(np.float32)
    label = cut(case.label, origin, cfg.size, fill=0)
    if augment:
        hu, label = symmetry([hu, label], rng)
        hu = hu * rng.uniform(0.95, 1.05) + rng.uniform(-10.0, 10.0)
    lumen = label > 0
    if rng.random() < cfg.no_click_share:
        n_inside = n_outside = 0
    else:
        n_inside, n_outside = int(rng.integers(0, cfg.max_inside + 1)), int(rng.integers(0, cfg.max_outside + 1))
    clicks = random_clicks(lumen, rng, n_inside, n_outside, case.spacing, cfg.near_wall, cfg.near_wall_share, cfg.margin, cfg.on_lumen_share)
    padded = np.zeros((MAX_CLICKS, 4), dtype=np.int64)
    padded[: len(clicks)] = clicks[:MAX_CLICKS]
    target = conditioned_target(lumen, padded)
    return dict(image=network_input(hu, padded), label=target.astype(np.uint8), clicks=padded, case=case.id, origin=origin)


class PatchDataset(torch.utils.data.Dataset):
    """One item per draw: a patch of a random position in scan `index`. With `augment`, every
    draw is new (a counter per worker goes into the seed); without, item i is the same patch and
    clicks every time, which is what validation wants."""

    def __init__(self, cache: str | Path, split: str, cfg: PatchConfig = PatchConfig(), augment: bool = True,
                 seed: int = 0, limit: int | None = None) -> None:
        self.cases = load_cases(cache, split, limit)
        if not self.cases:
            raise ValueError(f"no {split} scans in {cache}")
        self.cfg, self.augment, self.seed = cfg, augment, seed
        self._draws = 0

    def __len__(self) -> int:
        return len(self.cases)

    def __getitem__(self, index: int) -> dict:
        if self.augment:
            self._draws += 1
            rng = np.random.default_rng([torch.initial_seed() % 2**32, index, self._draws])
        else:
            rng = np.random.default_rng([self.seed, index])
        item = make_item(self.cases[index], self.cfg, rng, self.augment)
        return dict(image=torch.from_numpy(item["image"]), label=torch.from_numpy(item["label"]),
                    clicks=torch.from_numpy(item["clicks"]), spacing=torch.tensor(self.cases[index].spacing))


def describe(cache: str | Path) -> str:
    """A line per split: scans, voxels, labelled voxels."""
    cache = Path(cache)
    index, splits = read_index(cache), read_splits(cache)
    lines = [f"{cache}: {len(index['cases'])} scans at {index['spacing']} mm"]
    for split, ids in splits.items():
        metas = [index["cases"][i] for i in ids if i in index["cases"]]
        voxels = sum(math.prod(m["shape"]) for m in metas)
        lumen = sum(m["lumen_voxels"] for m in metas)
        lines.append(f"  {split}: {len(metas)} scans, {voxels / 1e6:.0f} M voxels, {lumen / 1e6:.2f} M labelled ({100 * lumen / max(voxels, 1):.2f} %)")
    return "\n".join(lines)


if __name__ == "__main__":
    import sys

    print(describe(sys.argv[1]))
    print(json.dumps(PatchConfig().__dict__, indent=1))
