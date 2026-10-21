/**
 * What is built from the last segmentation on request: the labelmap and the surface. Each is
 * built in the backend, fetched, stored as a dataset, and reported in its own slice of state
 * (`Product`); `build` is what they share, and `PRODUCTS` is the one list of them, which auto
 * update and the panels loop over. A new product is an entry there with a build function, its
 * state in `state.ts`, its endpoints, and a panel.
 */

import vtkCellArray from "@kitware/vtk.js/Common/Core/CellArray";
import vtkDataArray from "@kitware/vtk.js/Common/Core/DataArray";
import vtkPolyData from "@kitware/vtk.js/Common/DataModel/PolyData";

import { imageOnGrid } from "@/lib/io/loaders";
import { currentImage, onGridOf, voxelCount } from "@/lib/scene/datasets";
import type { SceneState } from "@/lib/scene/store";
import type { ImageDataset, LabelmapDataset, MeshDataset } from "@/lib/scene/types";
import type { Bounds } from "@/lib/world";
import { buildLabelmap, buildSurface, fetchLabelmap, fetchSurface, type Mesh } from "./backend";
import { isCurrent, messageOf, store } from "./flow";
import { productStale, surfaceLevel, type ProductKey } from "./state";

type Built<K extends ProductKey> = Omit<NonNullable<SceneState[K]["built"]>, "generation">;

/**
 * Build `key` from the last segmentation: `work` asks the backend and returns what was built,
 * and how to store it, which happens only if its image is still the one open. Without a
 * session or a segmentation to build from, the product reports why instead.
 */
async function build<K extends ProductKey>(
  key: K,
  message: string,
  work: (sessionId: string, image: ImageDataset) => Promise<{ built: Built<K>; commit: () => void }>,
): Promise<void> {
  const state = store();
  const image = currentImage(state);
  if (!image) return;
  const report = (patch: Partial<SceneState[K]>) => store().setProduct(key, patch);
  const { sessionId, generated, generation } = state.segmentation;
  if (!generated) return void report({ status: { state: "error", message: "No segmentation yet: generate one first." } } as Partial<SceneState[K]>);
  if (!sessionId || sessionId !== generated.sessionId) {
    return void report({ status: { state: "error", message: "The backend session is new and has no segmentation yet: generate again." } } as Partial<SceneState[K]>);
  }
  report({ status: { state: "running", message } } as Partial<SceneState[K]>);
  try {
    const { built, commit } = await work(sessionId, image);
    if (!isCurrent(image.id)) return;
    commit();
    report({ built: { ...built, generation }, status: { state: "idle" } } as Partial<SceneState[K]>);
  } catch (error) {
    if (isCurrent(image.id)) report({ status: { state: "error", message: messageOf(error) } } as Partial<SceneState[K]>);
  }
}

/** The labelmap at the panel's threshold: uint8 on the image's grid, drawn over it. */
export function buildSegmentationLabelmap(): Promise<void> {
  const { threshold } = store().labelmap;
  return build("labelmap", "Building labelmap…", async (sessionId, image) => {
    const summary = await buildLabelmap(sessionId, threshold);
    const labels = await fetchLabelmap(sessionId, voxelCount(image));
    const dataset: LabelmapDataset = {
      ...onGridOf(image, "labelmap"),
      kind: "labelmap",
      label: "Labelmap",
      threshold,
      voxels: summary.voxels,
      arrayName: "label",
      scalarType: "Uint8Array",
      scalarRange: [0, 1],
    };
    return {
      built: { threshold, voxels: summary.voxels, seconds: summary.seconds },
      commit: () => store().putMap(dataset, imageOnGrid(image, "label", labels), {}),
    };
  });
}

/** The surface the panel asks for: marching cubes on a probability, or on the unary logit's zero level. */
export function buildSegmentationSurface(): Promise<void> {
  const { source } = store().surface;
  const level = surfaceLevel(store().surface);
  return build("surface", "Building surface…", async (sessionId, image) => {
    const summary = await buildSurface(sessionId, source, level);
    const mesh = await fetchSurface(sessionId);
    const dataset: MeshDataset = {
      id: `${image.id}:surface`,
      kind: "mesh",
      label: "Surface",
      sourceId: image.id,
      source,
      level,
      points: summary.points,
      triangles: summary.triangles,
      version: 1,
      bounds: meshBounds(mesh.points) ?? image.bounds,
    };
    return {
      built: { source, level, triangles: summary.triangles, seconds: summary.seconds },
      commit: () => store().putMesh(dataset, toPolyData(mesh), {}),
    };
  });
}

export interface ProductSpec {
  key: ProductKey;
  /** What the panel calls building it. */
  label: string;
  build: () => Promise<void>;
  /** Whether building it now would give something other than what is there. */
  stale: (state: SceneState) => boolean;
  /** Its own settings: a change to them asks auto update for a run. */
  inputs: (state: SceneState) => unknown[];
}

/** Every product, in the order they are built. */
export const PRODUCTS: readonly ProductSpec[] = [
  {
    key: "labelmap",
    label: "Build labelmap",
    build: buildSegmentationLabelmap,
    stale: (state) => productStale(state, "labelmap"),
    inputs: (state) => [state.labelmap.threshold],
  },
  {
    key: "surface",
    label: "Build surface",
    build: buildSegmentationSurface,
    stale: (state) => productStale(state, "surface"),
    inputs: (state) => [state.surface.source, state.surface.threshold],
  },
];

function toPolyData(mesh: Mesh): vtkPolyData {
  const poly = vtkPolyData.newInstance();
  poly.getPoints().setData(mesh.points, 3);
  const triangles = mesh.triangles.length / 3;
  const cells = new Uint32Array(triangles * 4);
  for (let t = 0; t < triangles; t += 1) {
    cells[t * 4] = 3;
    cells.set(mesh.triangles.subarray(t * 3, t * 3 + 3), t * 4 + 1);
  }
  poly.setPolys(vtkCellArray.newInstance({ values: cells }));
  poly.getPointData().setNormals(vtkDataArray.newInstance({ name: "Normals", numberOfComponents: 3, values: mesh.normals }));
  return poly;
}

function meshBounds(points: Float32Array): Bounds | null {
  if (points.length === 0) return null;
  const bounds: Bounds = [Infinity, -Infinity, Infinity, -Infinity, Infinity, -Infinity];
  for (let i = 0; i < points.length; i += 3) {
    for (let a = 0; a < 3; a += 1) {
      bounds[a * 2] = Math.min(bounds[a * 2], points[i + a]);
      bounds[a * 2 + 1] = Math.max(bounds[a * 2 + 1], points[i + a]);
    }
  }
  return bounds;
}
