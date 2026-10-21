/**
 * The scene's data model: plain, serialisable data only -- no vtk.js objects, class instances
 * or functions. The vtk.js data lives in `dataRegistry.ts`, keyed by `DatasetId`.
 *
 * A `Dataset` is *what you have*; a `SceneNode` is *how it is drawn*. A dataset's nodes are
 * fixed (`NODE_TYPES_FOR_KIND`), and the pane decides where a slice cuts (`views.ts`), so one
 * slice node serves every pane.
 */

import type { Rgb } from "@/lib/color";
import type { Bounds, GridGeometry, Mat3 } from "@/lib/world";

export type DatasetId = string;
export type NodeId = string;

// --------------------------------------------------------------------------- //
// Datasets
// --------------------------------------------------------------------------- //

/** File formats the loaders read. */
export type ImageFormat = "vti" | "mha" | "mhd";

/** A regular sampling grid, as the loader reports it. */
interface GridInfo extends GridGeometry {
  /** World directions of the voxel axes, **[i axis, j axis, k axis]** (vtk.js's layout). */
  direction: Mat3;
  arrayName: string;
  /** Typed-array name of the voxels, e.g. `Int16Array`. */
  scalarType: string;
  scalarRange: [number, number];
}

interface DatasetCommon {
  id: DatasetId;
  label: string;
  /** Bumped on every content change, so caches keyed on `id:version` are invalidated. */
  version: number;
  /** World-space bounds, LPS mm. */
  bounds: Bounds;
}

export interface ImageDataset extends DatasetCommon, GridInfo {
  kind: "image";
  /** Where it was loaded from. */
  path: string;
  format: ImageFormat;
}

/** What a segmentation produces on its image's grid; each is described in `segmentation/maps.ts`. */
export type MapQuantity = "unary" | "coupling" | "probability";

/**
 * A map computed on an image's grid by the backend, replaced in place (same id, next
 * `version`) on every Generate so whatever draws it keeps its settings. Stored compact --
 * int16 or uint8 times `valueScale` -- because every pane holds a copy on the GPU.
 */
export interface MapDataset extends DatasetCommon, GridInfo {
  kind: "map";
  quantity: MapQuantity;
  /** The image whose grid it shares. */
  sourceId: DatasetId;
  /** What a stored value is multiplied by to give the real one; `scalarRange` is real. */
  valueScale: number;
}

/**
 * A binary label on an image's grid: 1 where p(y = 1) >= `threshold`, 0 elsewhere, as uint8.
 * Thresholded in the backend on the exact maps, like the surface, so the two agree.
 */
export interface LabelmapDataset extends DatasetCommon, GridInfo {
  kind: "labelmap";
  sourceId: DatasetId;
  threshold: number;
  /** How many voxels are labelled 1. */
  voxels: number;
}

/** Which map a surface is an isosurface of. */
export type SurfaceSource = "probability" | "unary";

/** A segmentation's boundary as a triangle mesh, built by the backend. */
export interface MeshDataset extends DatasetCommon {
  kind: "mesh";
  sourceId: DatasetId;
  source: SurfaceSource;
  /** The isovalue: the threshold for the probability, 0 for the unary logit. */
  level: number;
  points: number;
  triangles: number;
}

export type Dataset = ImageDataset | MapDataset | LabelmapDataset | MeshDataset;

/** A dataset with voxels. */
export type GridDataset = ImageDataset | MapDataset | LabelmapDataset;

// --------------------------------------------------------------------------- //
// Scene nodes
// --------------------------------------------------------------------------- //

/** How a volume is sampled where a plane cuts it: blended between voxels, or the nearest one. */
export type Interpolation = "trilinear" | "nearest";

interface NodeCommon {
  id: NodeId;
  name: string;
  datasetId: DatasetId;
  visible: boolean;
}

/** Slices through a volume. No orientation: each pane's plane decides where it cuts. */
export interface ImageSliceNode extends NodeCommon {
  type: "imageSlice";
  colorWindow: number;
  colorLevel: number;
  opacity: number;
  interpolation: Interpolation;
}

/** Wireframe box around a volume, in the 3D pane. */
export interface OutlineNode extends NodeCommon {
  type: "outline";
  color: Rgb;
}

/** Colour maps an overlay can use (`viewer/colorRamp.ts`). */
export type OverlayColorMap = "jet" | "diverging" | "sequential";

/** A map drawn over the image, cut wherever the image is. */
export interface OverlayNode extends NodeCommon {
  type: "overlay";
  colorMap: OverlayColorMap;
  /** The values the colour map spans, low to high, in real units. */
  range: [number, number];
  opacity: number;
  interpolation: Interpolation;
}

/** A label map drawn over the image in one colour, cut wherever the image is. */
export interface LabelmapNode extends NodeCommon {
  type: "labelmap";
  color: Rgb;
  opacity: number;
  interpolation: Interpolation;
}

/**
 * A surface mesh: in the 3D pane (`visible`), and in each slice view as the line where the
 * view's plane cuts it (`onSlices`).
 */
export interface SurfaceNode extends NodeCommon {
  type: "surface";
  color: Rgb;
  opacity: number;
  wireframe: boolean;
  onSlices: boolean;
  /** The slice outline's width, in screen pixels. */
  lineWidth: number;
}

export type SceneNode = ImageSliceNode | OutlineNode | OverlayNode | LabelmapNode | SurfaceNode;

export type SceneNodeType = SceneNode["type"];

/** The views every dataset of a kind gets, created with it. */
export const NODE_TYPES_FOR_KIND: Record<Dataset["kind"], SceneNodeType[]> = {
  image: ["imageSlice", "outline"],
  map: ["overlay"],
  labelmap: ["labelmap"],
  mesh: ["surface"],
};

export const NODE_TYPE_LABELS: Record<SceneNodeType, string> = {
  imageSlice: "Slices",
  outline: "Bounding box",
  overlay: "Overlay",
  labelmap: "Labelmap",
  surface: "Surface",
};

/** An operation the editor reports on: loading, connecting, building. */
export type TaskStatus = { state: "idle" } | { state: "running"; message: string } | { state: "error"; message: string };
