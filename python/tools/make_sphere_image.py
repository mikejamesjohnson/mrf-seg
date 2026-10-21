"""Write a synthetic test image: a bright sphere in a noisy background, as a MetaImage.

    python/.venv/bin/python python/tools/make_sphere_image.py            # -> data/sphere.mha

The grid is deliberately not a cube and not isotropic (96 x 96 x 80 voxels of 1 x 1 x 1.25 mm),
so a mixed-up axis shows. The sphere is centred, 25 mm in radius, with a slightly blurred
edge; intensities are CT-like (0 outside, 250 inside) with noise.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy.ndimage import gaussian_filter

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mrfseg.image_io import write_metaimage  # noqa: E402

DIMS = (96, 96, 80)  # x, y, z
SPACING = (1.0, 1.0, 1.25)
RADIUS = 25.0
INSIDE, OUTSIDE, NOISE = 250.0, 0.0, 25.0


def sphere_image(seed: int = 0) -> tuple[np.ndarray, tuple[float, float, float]]:
    """(z, y, x) int16 voxels, and the origin that puts the sphere's centre at the world origin."""
    nx, ny, nz = DIMS
    sx, sy, sz = SPACING
    origin = (-(nx - 1) / 2 * sx, -(ny - 1) / 2 * sy, -(nz - 1) / 2 * sz)
    z, y, x = np.meshgrid(
        origin[2] + np.arange(nz) * sz, origin[1] + np.arange(ny) * sy, origin[0] + np.arange(nx) * sx, indexing="ij"
    )
    inside = np.sqrt(x * x + y * y + z * z) <= RADIUS
    values = np.where(inside, INSIDE, OUTSIDE).astype(np.float32)
    values = gaussian_filter(values, sigma=1.0)
    values += np.random.default_rng(seed).normal(0, NOISE, values.shape)
    return np.round(values).astype(np.int16), origin


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parents[2] / "data" / "sphere.mha")
    out = parser.parse_args().out
    out.parent.mkdir(parents=True, exist_ok=True)
    voxels, origin = sphere_image()
    write_metaimage(out, voxels, origin, SPACING)
    print(f"wrote {out}: {DIMS[0]} x {DIMS[1]} x {DIMS[2]} voxels, sphere of {RADIUS:g} mm at the origin")


if __name__ == "__main__":
    main()
