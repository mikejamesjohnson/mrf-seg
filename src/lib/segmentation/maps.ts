/**
 * The maps a Generate returns, one entry each: what the editor calls it, how it comes over the
 * wire, and how it is drawn. Adding a map means an entry here and, in Python, a field of
 * `Segmentation` filled in `MRFSegmentation._maps` and an entry in `encoding.MAPS`.
 */

import type { MapQuantity, OverlayNode } from "@/lib/scene/types";

export interface MapDisplay {
  label: string;
  /** What it is, in a few symbols, for the panel; null if the label says it all. */
  meaning: string | null;
  /** How it comes with `?encoding=compact` (`mrfseg/server/encoding.py`). */
  compact: Int16ArrayConstructor | Uint8ArrayConstructor;
  /** How it is drawn when it first appears; after that, the user's settings stay. */
  firstOverlay: Pick<OverlayNode, "colorMap" | "visible" | "opacity">;
  /** The values its colour map spans, from its real range. */
  span: (range: [number, number]) => [number, number];
  /** What the two ends of the legend say, for a span. */
  legend: (span: [number, number]) => [string, string];
}

const peakOf = ([low, high]: [number, number]) => Math.max(Math.abs(low), Math.abs(high), 1e-6);

export const MAP_DISPLAY: Record<MapQuantity, MapDisplay> = {
  probability: {
    label: "Probability",
    meaning: "p: reaching an inside point first",
    compact: Uint8Array,
    firstOverlay: { colorMap: "jet", visible: true, opacity: 0.5 },
    span: () => [0, 1],
    legend: () => ["0", "1"],
  },
  affinity: {
    label: "Affinity",
    meaning: "mean log w at the voxel",
    compact: Int16Array,
    firstOverlay: { colorMap: "diverging", visible: false, opacity: 0.5 },
    // Symmetric about zero, the affinity of 1: positive flows, negative holds back.
    span: (range) => [-peakOf(range), peakOf(range)],
    legend: ([low, high]) => [`${low.toFixed(1)} wall`, `flows +${high.toFixed(1)}`],
  },
};

/** In the order the backend lists them and the panel shows them. */
export const MAP_QUANTITIES = ["affinity", "probability"] as const satisfies readonly MapQuantity[];
