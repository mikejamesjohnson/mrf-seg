/**
 * The automatic window and level, and the rule that chose it: the default a slice node starts
 * with, and what the "auto" button returns to. A Hounsfield volume is calibrated, so it gets
 * the soft-tissue window (most of a CT is air, so even a low percentile is still air);
 * anything else gets a 1-99% percentile clip.
 */

import type { Histogram } from "./histogram";
import type { GridDataset } from "./types";

export interface AutoWindowLevel {
  colorWindow: number;
  colorLevel: number;
  /** Which rule fired, for the UI to name. */
  rule: string;
}

/** Air plus something denser than water: a loose but reliable CT signature. */
function looksLikeHounsfield([min, max]: readonly [number, number]): boolean {
  return min < -300 && max - min > 800;
}

/** Value below which `fraction` of the samples fall, at bin resolution. */
function percentileOf(histogram: Histogram, fraction: number): number {
  const target = histogram.total * Math.min(Math.max(fraction, 0), 1);
  const binWidth = (histogram.hi - histogram.lo) / histogram.counts.length;
  let seen = 0;
  for (let bin = 0; bin < histogram.counts.length; bin += 1) {
    seen += histogram.counts[bin];
    if (seen >= target) return histogram.lo + (bin + 1) * binWidth;
  }
  return histogram.hi;
}

/** Fraction of samples outside `[black, white]` -- crushed to black or blown out. */
export function clippedFraction(histogram: Histogram, black: number, white: number): number {
  if (histogram.total === 0) return 0;
  const binWidth = (histogram.hi - histogram.lo) / histogram.counts.length;
  let clipped = 0;
  for (let bin = 0; bin < histogram.counts.length; bin += 1) {
    const from = histogram.lo + bin * binWidth;
    const to = from + binWidth;
    if (to <= black || from >= white) clipped += histogram.counts[bin];
  }
  return clipped / histogram.total;
}

/** The soft-tissue window, in Hounsfield units. */
const SOFT_TISSUE = { colorWindow: 400, colorLevel: 40 };

export function autoWindowLevel(dataset: GridDataset, histogram: Histogram | null): AutoWindowLevel {
  const range = dataset.scalarRange;
  if (looksLikeHounsfield(range)) return { ...SOFT_TISSUE, rule: "soft tissue" };

  if (histogram) {
    const lo = percentileOf(histogram, 0.01);
    const hi = percentileOf(histogram, 0.99);
    if (hi > lo) return { colorWindow: hi - lo, colorLevel: (hi + lo) / 2, rule: "1–99%" };
  }
  return {
    colorWindow: Math.max(range[1] - range[0], 1e-6),
    colorLevel: (range[0] + range[1]) / 2,
    rule: "full range",
  };
}
