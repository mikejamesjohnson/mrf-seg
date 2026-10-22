"""How arrays go to the editor: the maps in a compact encoding, and the surface as one buffer.
The editor's side is `src/lib/segmentation/backend.ts`."""

from __future__ import annotations

import struct
import sys
from dataclasses import dataclass

import numpy as np
from vtkmodules.util.numpy_support import vtk_to_numpy
from vtkmodules.vtkCommonDataModel import vtkPolyData


@dataclass(frozen=True)
class MapEncoding:
    """How a map is sent with `?encoding=compact`, which is what the editor draws: a CT-sized
    float32 map is 378 MB, and each of its panes holds a copy on the GPU."""

    dtype: type
    scale: float | None
    """What a compact value is multiplied by to give the real one; None scales the map's largest
    magnitude to the type's largest value, which keeps zero exact."""


#: The two maps of a `Segmentation`, by the name of its field, in the order the editor lists
#: them. A probability needs no finer than 1/255; the log-affinity fills int16.
MAPS: dict[str, MapEncoding] = {
    "affinity": MapEncoding(np.int16, None),
    "probability": MapEncoding(np.uint8, 1 / 255),
}


def compact_scale(name: str, low: float, high: float) -> float:
    """What a compact value of map `name`, whose values run from `low` to `high`, is multiplied by."""
    encoding = MAPS[name]
    if encoding.scale is not None:
        return encoding.scale
    peak = max(-low, high)
    return peak / np.iinfo(encoding.dtype).max if peak > 0 else 1.0


def compact(name: str, values: np.ndarray, scale: float) -> np.ndarray:
    """`values / scale`, rounded and flat, a block at a time to keep the temporaries small."""
    values = values.reshape(-1)
    out = np.empty(values.shape, dtype=MAPS[name].dtype)
    block = 1 << 22
    for start in range(0, values.size, block):
        out[start : start + block] = np.rint(values[start : start + block] / scale)
    return out


def little_endian(values: np.ndarray) -> np.ndarray:
    values = np.ascontiguousarray(values)
    return values.byteswap() if sys.byteorder != "little" else values


def mesh_bytes(surface: vtkPolyData) -> bytes:
    """uint32 point and triangle counts, then float32 points (x, y, z), float32 normals and
    uint32 triangles (three point indices each), all little-endian."""
    if surface.GetNumberOfPoints() == 0:
        return struct.pack("<II", 0, 0)
    points = vtk_to_numpy(surface.GetPoints().GetData()).astype("<f4")
    normals = vtk_to_numpy(surface.GetPointData().GetNormals()).astype("<f4")
    triangles = vtk_to_numpy(surface.GetPolys().GetConnectivityArray()).astype("<u4")
    header = struct.pack("<II", len(points), len(triangles) // 3)
    return header + points.tobytes() + normals.tobytes() + triangles.tobytes()
