/**
 * Colour maps as plain data, so the renderer and the controls that show them use one
 * definition. Each stop sits at `at` in [-1, 1], meaning `level + at * window / 2`; values
 * beyond clamp, as the mapper does.
 */

import { rgbCss, type Rgb } from "@/lib/color";
import type { OverlayColorMap } from "@/lib/scene/types";

export interface ColorRamp {
  colors: { at: number; rgb: Rgb }[];
  /** Opacity along the same axis, so the image shows through where the value is low. None: opaque. */
  alpha?: { at: number; alpha: number }[];
}

export const GRAYSCALE: ColorRamp = {
  colors: [
    { at: -1, rgb: [0, 0, 0] },
    { at: 1, rgb: [1, 1, 1] },
  ],
};

export const OVERLAY_RAMPS: Record<OverlayColorMap, ColorRamp> = {
  /** MATLAB's jet, for p(y = 1): dark blue through cyan, yellow and red to dark red. */
  jet: {
    colors: [
      { at: -1, rgb: [0, 0, 0.5] },
      { at: -0.75, rgb: [0, 0, 1] },
      { at: -0.25, rgb: [0, 1, 1] },
      { at: 0.25, rgb: [1, 1, 0] },
      { at: 0.75, rgb: [1, 0, 0] },
      { at: 1, rgb: [0.5, 0, 0] },
    ],
  },
  /** A signed field about zero, in the points' colours: red outside, grey at the boundary, blue inside. */
  diverging: {
    colors: [
      { at: -1, rgb: [0.8, 0.16, 0.14] },
      { at: 0, rgb: [0.87, 0.87, 0.87] },
      { at: 1, rgb: [0.16, 0.4, 0.82] },
    ],
  },
  /** A magnitude from zero, dark to bright: the latent std. */
  sequential: {
    colors: [
      { at: -1, rgb: [0.05, 0.03, 0.2] },
      { at: -0.3, rgb: [0.45, 0.12, 0.5] },
      { at: 0.4, rgb: [0.9, 0.35, 0.3] },
      { at: 1, rgb: [0.99, 0.9, 0.55] },
    ],
  },
};

/** A label: one colour, transparent at 0 and opaque at 1 (over a window of 1 at level 0.5). */
export function labelRamp(rgb: Rgb): ColorRamp {
  return {
    colors: [
      { at: -1, rgb },
      { at: 1, rgb },
    ],
    alpha: [
      { at: -1, alpha: 0 },
      { at: 1, alpha: 1 },
    ],
  };
}

/** Opacity stops, as [stored value, alpha], for a ramp over `window` at `level`: the ramp's own, or opaque when it has none. */
export function opacityStops(ramp: ColorRamp, window: number, level: number): [number, number][] {
  const half = Math.max(window / 2, 1e-6);
  return (ramp.alpha ?? [{ at: -1, alpha: 1 }, { at: 1, alpha: 1 }]).map(({ at, alpha }): [number, number] => [level + at * half, alpha]);
}

/** The colour at `at`, interpolated between stops and clamped at the ends. */
function sampleRamp(ramp: ColorRamp, at: number): Rgb {
  const stops = ramp.colors;
  if (at <= stops[0].at) return stops[0].rgb;
  for (let i = 1; i < stops.length; i += 1) {
    const [a, b] = [stops[i - 1], stops[i]];
    if (at <= b.at) {
      const t = (at - a.at) / Math.max(b.at - a.at, 1e-9);
      return a.rgb.map((c, k) => c + (b.rgb[k] - c) * t) as Rgb;
    }
  }
  return stops[stops.length - 1].rgb;
}

/** CSS colour at `at`. */
export function cssAt(ramp: ColorRamp, at: number): string {
  return rgbCss(sampleRamp(ramp, at));
}

/** The ramp as a CSS gradient, left to right. */
export function rampGradient(ramp: ColorRamp): string {
  return `linear-gradient(to right, ${ramp.colors.map((stop) => `${cssAt(ramp, stop.at)} ${((stop.at + 1) / 2) * 100}%`).join(", ")})`;
}
