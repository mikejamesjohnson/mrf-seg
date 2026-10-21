/**
 * Intensity histograms, what the window/level control is drawn on. Computing one needs the
 * voxels, so it is a cache beside the registry keyed by `id:version`, not store state.
 */

import { getImage } from "./dataRegistry";
import type { DatasetId, GridDataset } from "./types";

/** Bins: the resolution of the widget, not of the data. */
const BINS = 256;

export interface Histogram {
  /** Counts per bin, low value first. */
  counts: number[];
  /** The value range the bins span -- the dataset's own scalar range. */
  lo: number;
  hi: number;
  /** Largest count, so a caller can normalise without a second pass. */
  peak: number;
  /** Total samples, for percentage readouts. */
  total: number;
}

const cache = new Map<string, Histogram>();

/** The histogram of a dataset's voxels, computed once per version. */
export function histogramOf(dataset: GridDataset): Histogram | null {
  const key = `${dataset.id}:${dataset.version}`;
  const cached = cache.get(key);
  if (cached) return cached;

  const values = getImage(dataset.id)?.getPointData().getScalars()?.getData() as
    | ArrayLike<number>
    | undefined;
  if (!values || values.length === 0) return null;

  const [lo, hi] = dataset.scalarRange;
  const span = hi - lo;
  const counts = new Array<number>(BINS).fill(0);
  // A constant volume puts every sample in one bin rather than dividing by zero.
  const scale = span > 0 ? BINS / span : 0;
  for (let n = 0; n < values.length; n += 1) {
    const bin = scale === 0 ? 0 : Math.min(Math.max(((values[n] - lo) * scale) | 0, 0), BINS - 1);
    counts[bin] += 1;
  }

  let peak = 0;
  for (const count of counts) if (count > peak) peak = count;
  const histogram: Histogram = { counts, lo, hi, peak, total: values.length };

  dropHistogram(dataset.id);
  cache.set(key, histogram);
  return histogram;
}

/** Forget a dataset's histogram. Called when the dataset goes. */
export function dropHistogram(id: DatasetId): void {
  for (const key of cache.keys()) if (key.startsWith(`${id}:`)) cache.delete(key);
}
