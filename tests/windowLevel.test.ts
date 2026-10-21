import { describe, expect, it } from "vitest";

import type { Histogram } from "@/lib/scene/histogram";
import type { GridDataset } from "@/lib/scene/types";
import { autoWindowLevel, clippedFraction } from "@/lib/scene/windowLevel";
import { cssAt, labelRamp, opacityStops, OVERLAY_RAMPS, rampGradient } from "@/lib/viewer/colorRamp";
import { makeImage } from "./helpers";

const withRange = (scalarRange: [number, number]): GridDataset => ({ ...makeImage([2, 2, 2], [1, 1, 1], [0, 0, 0]).dataset, scalarRange });

/** 100 samples over [0, 100): one per unit, in 100 bins. */
const uniform: Histogram = { counts: new Array(100).fill(1), lo: 0, hi: 100, peak: 1, total: 100 };

describe("autoWindowLevel", () => {
  it("gives a CT the soft-tissue window", () => {
    expect(autoWindowLevel(withRange([-1024, 3071]), null)).toEqual({ colorWindow: 400, colorLevel: 40, rule: "soft tissue" });
  });

  it("clips anything else at its 1st and 99th percentiles", () => {
    const auto = autoWindowLevel(withRange([0, 100]), uniform);
    expect(auto.rule).toBe("1–99%");
    expect(auto.colorLevel - auto.colorWindow / 2).toBeCloseTo(1);
    expect(auto.colorLevel + auto.colorWindow / 2).toBeCloseTo(99);
  });

  it("falls back on the full range when the percentiles say nothing", () => {
    const constant: Histogram = { counts: [8, ...new Array(99).fill(0)], lo: 5, hi: 5, peak: 8, total: 8 };
    expect(autoWindowLevel(withRange([5, 5]), constant)).toMatchObject({ colorLevel: 5, rule: "full range" });
  });
});

describe("clippedFraction", () => {
  it("counts the samples outside the window", () => {
    expect(clippedFraction(uniform, 10, 90)).toBeCloseTo(0.2);
    expect(clippedFraction(uniform, -5, 200)).toBe(0);
  });
});

describe("colour ramps", () => {
  it("run jet from dark blue to dark red, clamped beyond its ends", () => {
    expect(cssAt(OVERLAY_RAMPS.jet, -1)).toBe("rgb(0 0 128)");
    expect(cssAt(OVERLAY_RAMPS.jet, 5)).toBe("rgb(128 0 0)");
    expect(cssAt(OVERLAY_RAMPS.jet, 0)).toBe("rgb(128 255 128)");
    const gradient = rampGradient(OVERLAY_RAMPS.jet);
    expect(gradient.startsWith("linear-gradient(to right, rgb(0 0 128) 0%")).toBe(true);
    expect(gradient.endsWith("rgb(128 0 0) 100%)")).toBe(true);
  });
});

describe("opacity", () => {
  it("is the ramp's own over its window, or opaque", () => {
    expect(opacityStops(OVERLAY_RAMPS.jet, 255, 127.5)).toEqual([
      [0, 1],
      [255, 1],
    ]);
    expect(opacityStops(labelRamp([1, 0, 0]), 1, 0.5)).toEqual([
      [0, 0],
      [1, 1],
    ]);
  });
});
