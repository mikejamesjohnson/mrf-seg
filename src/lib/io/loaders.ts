/**
 * A file path to an `ImageDataset` + `vtkImageData`: the only layer that knows file formats,
 * and the only one allowed to correct geometry into LPS mm. The bytes come from the local
 * file route (`app/api/file`); parsing happens here, in the browser.
 *
 * - `.vti`: vtk.js's own reader, with the direction matrix transposed (`transposeMat3`), and
 *   an extent that does not start at 0 folded into the origin: everything else takes voxel
 *   (0, 0, 0) to be at the origin.
 * - `.mha`, `.mhd` (+ its data file): `metaImage.ts`, since vtk.js has no reader for it.
 *
 * The backend reads the same files (`python/mrfseg/image_io.py`); the two must agree.
 */

import vtkDataArray from "@kitware/vtk.js/Common/Core/DataArray";
import vtkImageData from "@kitware/vtk.js/Common/DataModel/ImageData";
import vtkXMLImageDataReader from "@kitware/vtk.js/IO/XML/XMLImageDataReader";

import type { ImageDataset, ImageFormat } from "@/lib/scene/types";
import { add, dot, isIdentityMat3, scale, transposeMat3, voxelAxis, type Bounds, type GridGeometry, type Mat3, type Vec3 } from "@/lib/world";
import { decodeVoxels, inflate, LOCAL_DATA, readMetaImageHeader, type ScalarArray } from "./metaImage";

class ImageLoadError extends Error {}

const FORMATS: Record<string, ImageFormat> = { vti: "vti", mha: "mha", mhd: "mhd" };

/** Every extension the loader reads, for messages and placeholders. */
export const SUPPORTED_EXTENSIONS = Object.keys(FORMATS).map((ext) => `.${ext}`);

export function fileName(path: string): string {
  return path.slice(path.lastIndexOf("/") + 1);
}

/** A file's bytes through the file route; `data` asks for the data file a `.mhd` header names. */
async function fetchFile(path: string, part?: "data"): Promise<Response> {
  const response = await fetch(`/api/file?path=${encodeURIComponent(path)}${part ? `&part=${part}` : ""}`);
  if (!response.ok) {
    const reason = (await response.text().catch(() => "")) || response.statusText;
    throw new ImageLoadError(reason || `Could not read ${path} (HTTP ${response.status})`);
  }
  return response;
}

async function readVti(path: string): Promise<vtkImageData> {
  const reader = vtkXMLImageDataReader.newInstance();
  reader.parseAsArrayBuffer(await (await fetchFile(path)).arrayBuffer());
  const image = reader.getOutputData(0) as vtkImageData | undefined;
  if (!image) throw new ImageLoadError(`${fileName(path)}: no ImageData in file`);
  const asRead = [...image.getDirection()] as Mat3;
  if (!isIdentityMat3(asRead)) image.setDirection(...transposeMat3(asRead));
  const [x0, , y0, , z0] = image.getExtent();
  if (x0 || y0 || z0) {
    const [nx, ny, nz] = image.getDimensions();
    const grid = { gridOrigin: [...image.getOrigin()] as Vec3, spacing: [...image.getSpacing()] as Vec3, dimensions: [nx, ny, nz] as Vec3, direction: [...image.getDirection()] as Mat3 };
    const first = ([x0, y0, z0] as const).reduce<Vec3>((p, steps, a) => add(p, scale(voxelAxis(grid, a as 0 | 1 | 2), steps * grid.spacing[a])), grid.gridOrigin);
    image.setOrigin(first);
    image.setExtent(0, nx - 1, 0, ny - 1, 0, nz - 1);
  }
  const pointData = image.getPointData();
  // The file did not flag an active scalar array: take the first one.
  if (!pointData.getScalars() && pointData.getNumberOfArrays() > 0) pointData.setActiveAttributeByIndex(0, "Scalars");
  return image;
}

async function readMetaImage(path: string): Promise<vtkImageData> {
  const fileBytes = new Uint8Array(await (await fetchFile(path)).arrayBuffer());
  const { header, dataOffset } = readMetaImageHeader(fileBytes);
  let raw: Uint8Array | ReadableStream<Uint8Array>;
  if (header.dataFile.toUpperCase() === LOCAL_DATA) {
    raw = fileBytes.subarray(dataOffset);
  } else {
    const response = await fetchFile(path, "data");
    // Compressed data streams straight into the decompressor, never held whole.
    raw = header.compressed && response.body ? response.body : new Uint8Array(await response.arrayBuffer());
  }
  const voxels = decodeVoxels(header, header.compressed ? await inflate(raw) : (raw as Uint8Array));
  // The direction is already [i axis, j axis, k axis].
  return imageOnGrid({ ...header, gridOrigin: header.origin }, "Scalars", voxels);
}

/** `values` (x fastest) as a vtkImageData on `grid`. */
export function imageOnGrid(grid: GridGeometry, name: string, values: ScalarArray): vtkImageData {
  const image = vtkImageData.newInstance();
  image.setDimensions(grid.dimensions);
  image.setSpacing(grid.spacing);
  image.setOrigin(grid.gridOrigin);
  image.setDirection(...grid.direction);
  image.getPointData().setScalars(vtkDataArray.newInstance({ name, numberOfComponents: 1, values }));
  return image;
}

/** Unit, perpendicular voxel axes, to within what a DICOM-derived matrix's rounding leaves. */
function isRotation(direction: Mat3): boolean {
  const grid = { direction } as GridGeometry;
  const axes = ([0, 1, 2] as const).map((a) => voxelAxis(grid, a));
  return axes.every((u, i) => axes.every((v, j) => Math.abs(dot(u, v) - (i === j ? 1 : 0)) < 1e-3));
}

let loadCounter = 0;

export async function loadImageFromPath(path: string): Promise<{ dataset: ImageDataset; image: vtkImageData }> {
  const format = FORMATS[path.split(".").pop()?.toLowerCase() ?? ""];
  if (!format) throw new ImageLoadError(`Can't read ${fileName(path)}. Supported: ${SUPPORTED_EXTENSIONS.join(", ")}`);

  const image = format === "vti" ? await readVti(path) : await readMetaImage(path);
  const scalars = image.getPointData().getScalars();
  if (!scalars) throw new ImageLoadError(`${fileName(path)}: the image has no voxel values`);
  if (scalars.getNumberOfComponents() !== 1) {
    throw new ImageLoadError(`${fileName(path)}: images with several values per voxel are not supported yet`);
  }
  const bounds = [...image.getBounds()] as Bounds;
  if (!bounds.every(Number.isFinite)) throw new ImageLoadError(`${fileName(path)}: parsed with non-finite bounds`);
  const direction = [...image.getDirection()] as Mat3;
  if (!isRotation(direction)) {
    throw new ImageLoadError(`${fileName(path)}: its direction matrix is not a rotation (sheared or scaled axes are not supported)`);
  }

  const range = scalars.getRange();
  loadCounter += 1;
  const dataset: ImageDataset = {
    id: `image-${loadCounter}`,
    kind: "image",
    label: fileName(path).replace(/\.[^.]+$/, ""),
    path,
    format,
    version: 1,
    bounds,
    dimensions: [...image.getDimensions()] as Vec3,
    spacing: [...image.getSpacing()] as Vec3,
    gridOrigin: [...image.getOrigin()] as Vec3,
    direction,
    arrayName: scalars.getName() || "Scalars",
    scalarType: scalars.getDataType(),
    scalarRange: [range[0], range[1]],
  };
  return { dataset, image };
}
