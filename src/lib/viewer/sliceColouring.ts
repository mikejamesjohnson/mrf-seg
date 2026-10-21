/** How a slice node's values become colours: the ramp, and the window it spans in the data's own units. */

import type { Dataset, ImageSliceNode, LabelmapNode, OverlayNode } from "@/lib/scene/types";
import { GRAYSCALE, labelRamp, OVERLAY_RAMPS, type ColorRamp } from "@/lib/viewer/colorRamp";

export type SliceNode = ImageSliceNode | OverlayNode | LabelmapNode;

export interface Colouring {
  ramp: ColorRamp;
  /** The window and level the ramp spans, in *stored* values. */
  window: number;
  level: number;
  /** Changes when any of the above does: what a representation compares to know whether to rebuild. */
  key: string;
}

/**
 * The ramp and, in the map's *stored* values, the window it spans: an overlay's `range` is in
 * real units, but a map's voxels are compact (`MapDataset.valueScale`), and the mapper reads them.
 */
export function colouringOf(node: SliceNode, dataset: Dataset | undefined): Colouring {
  if (node.type === "imageSlice") {
    return { ramp: GRAYSCALE, window: node.colorWindow, level: node.colorLevel, key: `grey:${node.colorWindow}:${node.colorLevel}` };
  }
  if (node.type === "labelmap") return { ramp: labelRamp(node.color), window: 1, level: 0.5, key: `label:${node.color.join()}` };
  const valueScale = dataset?.kind === "map" ? dataset.valueScale : 1;
  const [low, high] = [node.range[0] / valueScale, node.range[1] / valueScale];
  const [window, level] = [Math.max(high - low, 1e-6), (low + high) / 2];
  return { ramp: OVERLAY_RAMPS[node.colorMap], window, level, key: `${node.colorMap}:${window}:${level}` };
}
