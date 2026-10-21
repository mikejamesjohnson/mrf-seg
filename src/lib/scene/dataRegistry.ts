/**
 * The vtk.js data objects, keyed by dataset id, outside any reactive store: they are large,
 * mutable and have identity, and nothing in the UI reads voxels. The store holds the plain
 * `Dataset` descriptions; the renderer looks the buffers up here.
 *
 * A replaced object (a regenerated map) is not deleted on `put`: a pane may draw it once more
 * before its next sync re-reads by `id:version`, so it is left to the garbage collector.
 */

import type vtkImageData from "@kitware/vtk.js/Common/DataModel/ImageData";
import type vtkPolyData from "@kitware/vtk.js/Common/DataModel/PolyData";

import type { DatasetId } from "./types";

const images = new Map<DatasetId, vtkImageData>();
const meshes = new Map<DatasetId, vtkPolyData>();

export function putImage(id: DatasetId, image: vtkImageData): void {
  images.set(id, image);
}

export function getImage(id: DatasetId): vtkImageData | undefined {
  return images.get(id);
}

/** Called once nothing draws it any more. */
export function dropImage(id: DatasetId): void {
  images.get(id)?.delete();
  images.delete(id);
}

export function putMesh(id: DatasetId, mesh: vtkPolyData): void {
  meshes.set(id, mesh);
}

export function getMesh(id: DatasetId): vtkPolyData | undefined {
  return meshes.get(id);
}

export function dropMesh(id: DatasetId): void {
  meshes.get(id)?.delete();
  meshes.delete(id);
}
