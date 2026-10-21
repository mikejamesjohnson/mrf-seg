/** What is loaded (datasets) and how each is drawn (nodes), and the editor's load status. */

import type vtkImageData from "@kitware/vtk.js/Common/DataModel/ImageData";
import type vtkPolyData from "@kitware/vtk.js/Common/DataModel/PolyData";

import { unionBounds, type Bounds } from "@/lib/world";
import { dropImage, dropMesh, putImage, putMesh } from "./dataRegistry";
import { dropHistogram, histogramOf } from "./histogram";
import type { SceneState, StoreSlice } from "./store";
import {
  NODE_TYPE_LABELS,
  NODE_TYPES_FOR_KIND,
  type Dataset,
  type DatasetId,
  type GridDataset,
  type ImageDataset,
  type LabelmapDataset,
  type LabelmapNode,
  type MapDataset,
  type MeshDataset,
  type NodeId,
  type OverlayNode,
  type SceneNode,
  type SceneNodeType,
  type SurfaceNode,
  type TaskStatus,
} from "./types";
import { autoWindowLevel } from "./windowLevel";

export interface DatasetsSlice {
  datasets: Record<DatasetId, Dataset>;
  datasetOrder: DatasetId[];
  nodes: Record<NodeId, SceneNode>;
  /** Drawing order: an overlay's nodes come after its image's. */
  nodeOrder: NodeId[];
  /** The union of the images' bounds. */
  worldBounds: Bounds | null;
  status: TaskStatus;

  addDataset: (dataset: ImageDataset, image: vtkImageData) => void;
  /** Add a map or a label map, or replace one in place (same id, next version), merging `node` into its node. */
  putMap: (dataset: MapDataset | LabelmapDataset, image: vtkImageData, node: Partial<OverlayNode> | Partial<LabelmapNode>) => void;
  /** Add the surface mesh, or replace it in place, merging `surface` into its node. */
  putMesh: (dataset: MeshDataset, mesh: vtkPolyData, surface: Partial<SurfaceNode>) => void;
  removeDataset: (id: DatasetId) => void;
  updateNode: (id: NodeId, patch: Partial<SceneNode>) => void;
  setStatus: (status: TaskStatus) => void;
}

export function images(state: SceneState): ImageDataset[] {
  return state.datasetOrder.map((id) => state.datasets[id]).filter((d): d is ImageDataset => d.kind === "image");
}

/** The image the views show: the first one loaded. */
export function currentImage(state: SceneState): ImageDataset | undefined {
  return images(state)[0];
}

const BYTES_PER_SCALAR: Record<string, number> = { Int8Array: 1, Uint8Array: 1, Int16Array: 2, Uint16Array: 2, Float64Array: 8 };

export function voxelCount(dataset: GridDataset): number {
  const [nx, ny, nz] = dataset.dimensions;
  return nx * ny * nz;
}

/** How much memory a dataset's voxels take, from its description alone. */
export function datasetBytes(dataset: Dataset): number {
  if (dataset.kind === "mesh") return dataset.points * 24 + dataset.triangles * 16;
  return voxelCount(dataset) * (BYTES_PER_SCALAR[dataset.scalarType] ?? 4);
}

/** The grid a dataset computed on `image`'s grid shares with it, with the id such datasets take. */
export function onGridOf(image: ImageDataset, suffix: string): Pick<GridDataset, "bounds" | "dimensions" | "spacing" | "gridOrigin" | "direction"> & { id: DatasetId; sourceId: DatasetId; version: 1 } {
  const { bounds, dimensions, spacing, gridOrigin, direction } = image;
  return { id: `${image.id}:${suffix}`, sourceId: image.id, version: 1, bounds, dimensions, spacing, gridOrigin, direction };
}

let nodeCounter = 0;

/** A dataset's view of one type, as it starts out. */
function defaultNode(dataset: Dataset, type: SceneNodeType): SceneNode {
  nodeCounter += 1;
  const base = { id: `${type}-${nodeCounter}`, datasetId: dataset.id, name: NODE_TYPE_LABELS[type], visible: true };
  switch (type) {
    case "imageSlice": {
      const { colorWindow, colorLevel } = autoWindowLevel(dataset as ImageDataset, histogramOf(dataset as ImageDataset));
      return { ...base, type, colorWindow, colorLevel, opacity: 1, interpolation: "trilinear" };
    }
    case "outline":
      return { ...base, type, color: [0.55, 0.6, 0.68] };
    case "overlay":
      return { ...base, type, colorMap: "jet", range: [0, 1], opacity: 0.5, interpolation: "trilinear" };
    case "labelmap":
      // A label is a set of voxels: by default drawn as the voxels it is.
      return { ...base, type, color: [1, 0, 0], opacity: 0.5, interpolation: "nearest" };
    case "surface":
      return { ...base, type, color: [0.95, 0.75, 0.3], opacity: 1, wireframe: false, onSlices: true, lineWidth: 2 };
  }
}

/** `dataset` added with its nodes (each with `patch`), or replaced in place keeping them. */
function withDataset(state: SceneState, dataset: Dataset, patch: Partial<SceneNode> = {}): Partial<SceneState> {
  const existing = state.datasets[dataset.id];
  if (existing) {
    // Same id, next version: whatever draws it re-reads the data and keeps its settings.
    const node = Object.values(state.nodes).find((n) => n.datasetId === dataset.id);
    return {
      datasets: { ...state.datasets, [dataset.id]: { ...dataset, version: existing.version + 1 } },
      nodes: node ? { ...state.nodes, [node.id]: { ...node, ...patch } as SceneNode } : state.nodes,
    };
  }
  const nodes = NODE_TYPES_FOR_KIND[dataset.kind].map((type) => ({ ...defaultNode(dataset, type), ...patch }) as SceneNode);
  return {
    datasets: { ...state.datasets, [dataset.id]: dataset },
    datasetOrder: [...state.datasetOrder, dataset.id],
    nodes: { ...state.nodes, ...Object.fromEntries(nodes.map((n) => [n.id, n])) },
    nodeOrder: [...state.nodeOrder, ...nodes.map((n) => n.id)],
  };
}

export const datasetsSlice: StoreSlice<DatasetsSlice> = (set) => ({
  datasets: {},
  datasetOrder: [],
  nodes: {},
  nodeOrder: [],
  worldBounds: null,
  status: { state: "idle" },

  addDataset: (dataset, image) => {
    putImage(dataset.id, image);
    set((state) => ({ ...withDataset(state, dataset), worldBounds: unionBounds(state.worldBounds, dataset.bounds) }));
  },

  putMap: (dataset, image, node) => {
    putImage(dataset.id, image);
    set((state) => withDataset(state, dataset, node));
  },

  putMesh: (dataset, mesh, surface) => {
    putMesh(dataset.id, mesh);
    set((state) => withDataset(state, dataset, surface));
  },

  removeDataset: (id) => {
    set((state) => {
      const nodeOrder = state.nodeOrder.filter((nid) => state.nodes[nid]?.datasetId !== id);
      const { [id]: _removed, ...datasets } = state.datasets;
      const datasetOrder = state.datasetOrder.filter((d) => d !== id);
      return {
        datasets,
        datasetOrder,
        nodes: Object.fromEntries(nodeOrder.map((nid) => [nid, state.nodes[nid]])),
        nodeOrder,
        // Recomputed rather than subtracted: bounds are not invertible.
        worldBounds: datasetOrder.reduce<Bounds | null>(
          (bounds, did) => (datasets[did].kind === "image" ? unionBounds(bounds, datasets[did].bounds) : bounds),
          null,
        ),
      };
    });
    // Not now: the panes sync on the next animation frame, and until then whatever draws
    // this dataset still holds its data. Two frames later every pane has let go of it.
    const release = () => {
      dropHistogram(id);
      dropImage(id);
      dropMesh(id);
    };
    if (typeof requestAnimationFrame === "function") requestAnimationFrame(() => requestAnimationFrame(release));
    else release();
  },

  updateNode: (id, patch) =>
    set((state) => {
      const existing = state.nodes[id];
      return existing ? { nodes: { ...state.nodes, [id]: { ...existing, ...patch } as SceneNode } } : state;
    }),

  setStatus: (status) => set({ status }),
});
