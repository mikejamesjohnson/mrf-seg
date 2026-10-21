/**
 * A slice pane's wheel, captured ahead of vtk.js (whose wheel would dolly the camera).
 *
 * Travel is accumulated into notches, so a trackpad swipe of many small events moves in
 * proportion rather than a notch per event. Acceleration goes by speed, not distance: a single
 * deliberate notch has no speed, so it always moves one step. Notches stay whole once
 * accelerated, so a slice moves by whole voxels. Cmd/Ctrl-wheel, or a pinch, zooms.
 */

/** One mouse-wheel detent in Chrome, and three lines in Firefox. */
const NOTCH_PX = 100;
const LINE_PX = 40;
/** An idle gap that ends a gesture, ms. */
const GESTURE_GAP_MS = 150;
/** Wheel speed (px/ms) at which acceleration begins and saturates, and the most it multiplies by. */
const GAIN_FROM = 1.2;
const GAIN_TO = 6;
const MAX_GAIN = 10;
/** Zoom per pixel of travel, as an exponent: one notch is about 15%. */
const ZOOM_PER_PX = 0.0015;

export interface WheelHandlers {
  /** Whole notches, positive for scrolling down. */
  scroll: (notches: number) => void;
  /** Multiply the view's scale by `factor`. */
  zoom: (factor: number) => void;
}

/** Listen for the wheel on `element`; returns the function that stops. */
export function bindSliceWheel(element: HTMLElement, handlers: WheelHandlers): () => void {
  let travel = 0;
  let travelAt = -Infinity;
  let speed = 0;

  const onWheel = (event: WheelEvent) => {
    event.preventDefault();
    event.stopPropagation();
    const perUnit = event.deltaMode === 1 ? LINE_PX : event.deltaMode === 2 ? element.clientHeight : 1;
    // A trackpad pinch arrives as a wheel event with ctrlKey.
    if (event.metaKey || event.ctrlKey) {
      handlers.zoom(Math.exp(event.deltaY * perUnit * ZOOM_PER_PX));
      return;
    }

    const dy = (event.shiftKey && event.deltaY === 0 ? event.deltaX : event.deltaY) * perUnit;
    const gap = event.timeStamp - travelAt;
    if (gap > GESTURE_GAP_MS) {
      travel = 0;
      speed = 0;
    } else {
      // Floored at a frame, and smoothed: trackpad delivery is ragged.
      speed = speed * 0.7 + (Math.abs(dy) / Math.max(gap, 8)) * 0.3;
    }
    travelAt = event.timeStamp;
    travel += dy;

    const notches = Math.trunc(travel / NOTCH_PX);
    if (notches === 0) return;
    travel -= notches * NOTCH_PX;
    const t = Math.min(Math.max((speed - GAIN_FROM) / (GAIN_TO - GAIN_FROM), 0), 1);
    handlers.scroll(Math.round(notches * (1 + (MAX_GAIN - 1) * t * t)));
  };

  element.addEventListener("wheel", onWheel, { capture: true, passive: false });
  return () => element.removeEventListener("wheel", onWheel, { capture: true });
}
