"""Reading an image file into a `vtkImageData`, in the editor's world space: LPS millimetres,
and writing one as MetaImage.

The editor reads the same files in the browser (`src/lib/io/`) and draws what is computed
here on its own copy, so **the two readers must agree** on every file; the editor checks
the geometry the backend read against its own.

- `.vti`: VTK's reader, which reads `Direction` the way VTK wrote it (vtk.js is the one
  that needs a transpose). An extent that does not start at 0 is folded into the origin, so
  voxel (0, 0, 0) is always at the origin, as the rest of the code assumes.
- `.mha`, `.mhd` (+ `.raw` / `.zraw`): MetaImage, read here rather than with
  `vtkMetaImageReader`, which as of VTK 9.7 loads `TransformMatrix` transposed. ITK writes
  row n as the direction of voxel axis n; VTK's matrix has the axes as columns.
"""

from __future__ import annotations

import zlib
from pathlib import Path

import numpy as np
from vtkmodules.util.numpy_support import numpy_to_vtk
from vtkmodules.vtkCommonDataModel import vtkImageData
from vtkmodules.vtkIOXML import vtkXMLImageDataReader

from .grid import axes

READABLE_SUFFIXES = (".vti", ".mha", ".mhd")

#: What `write_metaimage` writes each numpy type as.
_WRITABLE = {"int8": "MET_CHAR", "uint8": "MET_UCHAR", "int16": "MET_SHORT", "uint16": "MET_USHORT", "float32": "MET_FLOAT"}


class ImageReadError(Exception):
    """The file could not be read as an image."""


def read_image(path: str | Path) -> vtkImageData:
    """The image at `path`, with its origin, spacing and orientation, in LPS mm."""
    path = Path(path)
    suffix = path.suffix.lower()
    if suffix not in READABLE_SUFFIXES:
        raise ImageReadError(f"Can't read {path.name}. Supported: {', '.join(READABLE_SUFFIXES)}")
    if not path.is_file():
        raise ImageReadError(f"File not found: {path}")

    image = _read_vti(path) if suffix == ".vti" else _read_metaimage(path)
    scalars = image.GetPointData().GetScalars()
    if scalars is None:
        raise ImageReadError(f"{path.name}: the image has no voxel values")
    if scalars.GetNumberOfComponents() != 1:
        raise ImageReadError(f"{path.name}: images with several values per voxel are not supported yet")
    rotation = axes(image)
    if np.abs(rotation.T @ rotation - np.eye(3)).max() > 1e-3:
        raise ImageReadError(f"{path.name}: its direction matrix is not a rotation (sheared or scaled axes are not supported)")
    return image


def _read_vti(path: Path) -> vtkImageData:
    reader = vtkXMLImageDataReader()
    reader.SetFileName(str(path))
    reader.Update()
    output = reader.GetOutput()
    if output is None or output.GetNumberOfPoints() == 0:
        raise ImageReadError(f"{path.name}: no ImageData in file")
    image = vtkImageData()
    image.ShallowCopy(output)
    point_data = image.GetPointData()
    if point_data.GetScalars() is None and point_data.GetNumberOfArrays() > 0:
        # The file did not flag an active scalar array; take the first one.
        point_data.SetActiveScalars(point_data.GetArrayName(0))
    first = np.array(image.GetExtent()[::2])
    if first.any():
        nx, ny, nz = image.GetDimensions()
        image.SetOrigin(*(np.array(image.GetOrigin()) + axes(image) @ (first * np.array(image.GetSpacing()))))
        image.SetExtent(0, nx - 1, 0, ny - 1, 0, nz - 1)
    return image


# --------------------------------------------------------------------------- #
# MetaImage
# --------------------------------------------------------------------------- #

#: MetaIO element types. `MET_LONG` is 32-bit in MetaIO; 64-bit integers are not supported.
_ELEMENT_TYPES = {
    "MET_CHAR": "i1",
    "MET_UCHAR": "u1",
    "MET_SHORT": "i2",
    "MET_USHORT": "u2",
    "MET_INT": "i4",
    "MET_UINT": "u4",
    "MET_LONG": "i4",
    "MET_ULONG": "u4",
    "MET_FLOAT": "f4",
    "MET_DOUBLE": "f8",
}


def write_metaimage(
    path: str | Path,
    voxels: np.ndarray,
    origin: tuple[float, float, float],
    spacing: tuple[float, float, float],
    direction: np.ndarray | None = None,
) -> Path:
    """`voxels` (z, y, x) as a compressed `.mha`, the way ITK writes one: `Offset` is voxel
    (0, 0, 0) in world mm, and row n of `TransformMatrix` is the direction of voxel axis n
    (`direction`'s column n; the identity by default). Read back by `read_image`."""
    path = Path(path)
    voxels = np.asarray(voxels)
    if voxels.ndim != 3:
        raise ValueError(f"Expected (z, y, x) voxels, got shape {voxels.shape}")
    if voxels.dtype.name not in _WRITABLE:
        raise ValueError(f"Can't write {voxels.dtype} voxels; use one of {', '.join(_WRITABLE)}")
    axes = np.eye(3) if direction is None else np.asarray(direction, dtype=float)
    nz, ny, nx = voxels.shape
    data = zlib.compress(np.ascontiguousarray(voxels, dtype=voxels.dtype.newbyteorder("<")).tobytes(), 6)
    header = "\n".join(
        [
            "ObjectType = Image",
            "NDims = 3",
            "BinaryData = True",
            "BinaryDataByteOrderMSB = False",
            "CompressedData = True",
            f"CompressedDataSize = {len(data)}",
            "TransformMatrix = " + " ".join(f"{v:.17g}" for v in axes.T.ravel()),
            "Offset = " + " ".join(f"{v:.17g}" for v in origin),
            "ElementSpacing = " + " ".join(f"{v:.17g}" for v in spacing),
            f"DimSize = {nx} {ny} {nz}",
            f"ElementType = {_WRITABLE[voxels.dtype.name]}",
            "ElementDataFile = LOCAL",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(header.encode("latin-1") + b"\n" + data)
    return path


def _read_header(data: bytes) -> tuple[dict[str, str], int]:
    """`Key = Value` lines up to and including `ElementDataFile`, and where they end."""
    fields: dict[str, str] = {}
    offset = 0
    limit = min(len(data), 1 << 16)
    while offset < limit:
        newline = data.find(b"\n", offset)
        end = len(data) if newline < 0 else newline + 1
        key, equals, value = data[offset:end].decode("latin-1").rstrip("\r\n").partition("=")
        offset = end
        if equals and key.strip():
            name = key.strip().lower()
            fields[name] = value.strip()
            if name == "elementdatafile":
                return fields, offset
    raise ImageReadError("Not a MetaImage file: no ElementDataFile line in the header")


def _integer(fields: dict[str, str], key: str, default: int) -> int:
    try:
        return int(fields.get(key, str(default)))
    except ValueError:
        raise ImageReadError(f"MetaImage header: {key} should be a whole number, got {fields[key]!r}") from None


def _numbers(fields: dict[str, str], keys: tuple[str, ...], count: int) -> list[float] | None:
    for key in keys:
        if key in fields:
            try:
                values = [float(v) for v in fields[key].split()]
            except ValueError:
                values = []
            if len(values) < count:
                raise ImageReadError(f"MetaImage header: {key} should be {count} numbers, got {fields[key]!r}")
            return values[:count]
    return None


def _read_metaimage(path: Path) -> vtkImageData:
    data = path.read_bytes()
    fields, data_offset = _read_header(data)

    if fields.get("objecttype", "image").lower() != "image":
        raise ImageReadError(f"MetaImage ObjectType is {fields['objecttype']}, not Image")
    if fields.get("binarydata", "true").lower() != "true":
        raise ImageReadError("MetaImage with ASCII (BinaryData = False) voxels is not supported")
    n_dims = _integer(fields, "ndims", 3)
    if n_dims not in (2, 3):
        raise ImageReadError(f"MetaImage with NDims = {n_dims} is not supported (2D or 3D only)")

    def padded(values: list[float] | None, fill: float) -> list[float]:
        if values is None:
            return [fill, fill, fill]
        return list(values) + [fill] * (3 - n_dims)

    sizes = _numbers(fields, ("dimsize",), n_dims)
    if sizes is None or any(s < 1 or s != int(s) for s in sizes):
        raise ImageReadError(f"MetaImage header: invalid DimSize {fields.get('dimsize', '')!r}")
    dimensions = [int(s) for s in padded(sizes, 1)]
    spacing = padded(_numbers(fields, ("elementspacing", "elementsize"), n_dims), 1.0)
    origin = padded(_numbers(fields, ("offset", "origin", "position"), n_dims), 0.0)

    channels = _integer(fields, "elementnumberofchannels", 1)
    if channels != 1:
        raise ImageReadError(f"MetaImage with {channels} channels per voxel is not supported yet")
    element_type = fields.get("elementtype", "").upper()
    if element_type not in _ELEMENT_TYPES:
        raise ImageReadError(f"MetaImage ElementType {fields.get('elementtype', '')!r} is not supported")

    # Row n is the direction of voxel axis n (ITK's convention); VTK wants them as columns.
    axes = np.eye(3)
    matrix = _numbers(fields, ("transformmatrix", "rotation", "orientation"), n_dims * n_dims)
    if matrix is not None:
        axes[:n_dims, :n_dims] = np.reshape(matrix, (n_dims, n_dims))

    compressed = fields.get("compresseddata", "false").lower() == "true"
    big_endian = fields.get("binarydatabyteordermsb", fields.get("elementbyteordermsb", "false")).lower() == "true"
    data_file = fields["elementdatafile"]
    dtype = np.dtype(_ELEMENT_TYPES[element_type]).newbyteorder(">" if big_endian else "<")
    count = dimensions[0] * dimensions[1] * dimensions[2]
    byte_length = count * dtype.itemsize

    if data_file.upper() == "LOCAL":
        raw = data[data_offset:]
    else:
        if any(c in data_file for c in "*%") or data_file.upper() == "LIST":
            raise ImageReadError(f"MetaImage with one file per slice ({data_file}) is not supported")
        data_path = Path(data_file) if Path(data_file).is_absolute() else path.parent / data_file
        if not data_path.is_file():
            raise ImageReadError(f"MetaImage data file not found: {data_path}")
        raw = data_path.read_bytes()
    if compressed:
        try:
            raw = zlib.decompress(raw)
        except zlib.error as error:
            raise ImageReadError(f"{path.name}: could not inflate the voxel data ({error})") from None

    # A header size only applies to an uncompressed, separate data file.
    start = 0
    if not compressed and data_file.upper() != "LOCAL":
        header_size = _integer(fields, "headersize", 0)
        start = len(raw) - byte_length if header_size < 0 else header_size
    if start < 0 or len(raw) - start < byte_length:
        raise ImageReadError(
            f"MetaImage voxel data is {len(raw) - max(start, 0)} bytes; "
            f"{' × '.join(map(str, dimensions))} {element_type} needs {byte_length}"
        )
    voxels = np.frombuffer(raw, dtype=dtype, count=count, offset=start).astype(dtype.newbyteorder("="))

    image = vtkImageData()
    image.SetDimensions(*dimensions)
    image.SetSpacing(*spacing)
    image.SetOrigin(*origin)
    image.SetDirectionMatrix(*axes.T.ravel())
    scalars = numpy_to_vtk(voxels, deep=True)
    scalars.SetName("Scalars")
    image.GetPointData().SetScalars(scalars)
    return image
