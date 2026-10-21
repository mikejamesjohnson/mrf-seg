"use client";

import { useEffect, useEffectEvent, useRef, useState } from "react";

import { Label } from "@/components/ui/label";
import type { Histogram } from "@/lib/scene/histogram";
import { clippedFraction, type AutoWindowLevel } from "@/lib/scene/windowLevel";
import { cssAt, type ColorRamp } from "@/lib/viewer/colorRamp";
import { cn } from "@/lib/utils";

/**
 * Window and level as two endpoints on a histogram: "black here, white there" is what you
 * want, and neither W nor L is anything you can point at. Drag a handle to move one end, the
 * plot to slide both (level), scroll to spread or pinch them (window).
 *
 * The axis spans the data and the endpoints, so it grows when an endpoint is dragged past the
 * data. A drag therefore keeps the axis it started on: read on the growing one, the value
 * under the pointer would grow with it, and the endpoint run away.
 */

/** Plot height in px. */
const PLOT_HEIGHT = 56;
/** How close to a handle a press has to be, as a fraction of the axis, to grab it. */
const GRAB_FRACTION = 0.06;
/** Minimum window, as a fraction of the axis, so the endpoints cannot cross. */
const MIN_WINDOW_FRACTION = 0.005;

interface Props {
  histogram: Histogram | null;
  window: number;
  level: number;
  /** The colour map the values become, so the control can show it. */
  ramp: ColorRamp;
  /** What the "auto" button restores, and the rule it is named after. */
  auto: AutoWindowLevel;
  onChange: (next: { colorWindow: number; colorLevel: number }) => void;
}

export function WindowLevel(props: Props) {
  if (!props.histogram) return <p className="text-[10px] text-muted-foreground">Nothing loaded to plot.</p>;
  return <Plot {...props} histogram={props.histogram} />;
}

/** A drag under way: what it moves, the axis it started on, and where it started. */
interface Drag {
  what: "black" | "white" | "level";
  lo: number;
  span: number;
  from: number;
  black: number;
  white: number;
}

function Plot({ histogram, window, level, auto, ramp, onChange }: Props & { histogram: Histogram }) {
  const plotRef = useRef<SVGSVGElement>(null);
  // Log-Y by default: a CT histogram is mostly air, which flattens everything else.
  const [logY, setLogY] = useState(true);
  const drag = useRef<Drag | null>(null);

  const black = level - window / 2;
  const white = level + window / 2;
  // The axis spans the data *and* the current endpoints, so a window outside the data
  // range can still be seen and dragged back.
  const axisLo = Math.min(histogram.lo, black);
  const axisHi = Math.max(histogram.hi, white);
  const axisSpan = Math.max(axisHi - axisLo, 1e-9);
  const toFraction = (value: number) => (value - axisLo) / axisSpan;

  const fractionAt = (clientX: number) => {
    const rect = plotRef.current?.getBoundingClientRect();
    if (!rect || rect.width <= 0) return null;
    return (clientX - rect.left) / rect.width;
  };

  const apply = (nextBlack: number, nextWhite: number) => {
    const floor = MIN_WINDOW_FRACTION * axisSpan;
    const lo = Math.min(nextBlack, nextWhite - floor);
    const hi = Math.max(nextWhite, lo + floor);
    onChange({ colorWindow: hi - lo, colorLevel: (hi + lo) / 2 });
  };

  const onPointerDown = (event: React.PointerEvent<SVGSVGElement>) => {
    const fraction = fractionAt(event.clientX);
    if (fraction === null) return;
    const dBlack = Math.abs(fraction - toFraction(black));
    const dWhite = Math.abs(fraction - toFraction(white));
    // Nearest handle if either is close; otherwise the body, which slides both.
    const what = Math.min(dBlack, dWhite) < GRAB_FRACTION ? (dBlack <= dWhite ? "black" : "white") : "level";
    drag.current = { what, lo: axisLo, span: axisSpan, from: fraction, black, white };
    event.currentTarget.setPointerCapture(event.pointerId);
    event.preventDefault();
  };

  const onPointerMove = (event: React.PointerEvent<SVGSVGElement>) => {
    const current = drag.current;
    const fraction = fractionAt(event.clientX);
    if (!current || fraction === null) return;
    // On the axis the drag started on, and no further past its ends than the plot reaches.
    const value = current.lo + Math.min(Math.max(fraction, 0), 1) * current.span;
    if (current.what === "black") apply(value, current.white);
    else if (current.what === "white") apply(current.black, value);
    else {
      const shift = (fraction - current.from) * current.span;
      apply(current.black + shift, current.white + shift);
    }
    event.preventDefault();
  };

  const endDrag = (event: React.PointerEvent<SVGSVGElement>) => {
    drag.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  };

  const onWheel = useEffectEvent((event: WheelEvent) => {
    event.preventDefault();
    if (event.deltaY === 0) return;
    // Widen or narrow about the level. Multiplicative, so one notch feels the same
    // whether the window is 40 HU or 4000.
    const factor = event.deltaY > 0 ? 1.1 : 1 / 1.1;
    const half = Math.max((window * factor) / 2, (MIN_WINDOW_FRACTION * axisSpan) / 2);
    apply(level - half, level + half);
  });
  // Bound natively: React's wheel listener is passive, so it could not keep the panel from
  // scrolling as well.
  useEffect(() => {
    const plot = plotRef.current;
    if (!plot) return;
    const listener = (event: WheelEvent) => onWheel(event);
    plot.addEventListener("wheel", listener, { passive: false });
    return () => plot.removeEventListener("wheel", listener);
  }, []);

  const height = (count: number) =>
    histogram.peak <= 0 ? 0 : logY ? Math.log1p(count) / Math.log1p(histogram.peak) : count / histogram.peak;

  const binWidth = (histogram.hi - histogram.lo) / histogram.counts.length;
  const bars = histogram.counts.map((count, index) => {
    // Bins are in data coordinates and the axis may be wider, so each is remapped.
    const value = histogram.lo + index * binWidth;
    const left = toFraction(value) * 100;
    const right = toFraction(value + binWidth) * 100;
    return { left, width: Math.max(right - left, 0.0001), h: height(count) };
  });

  const clipped = clippedFraction(histogram, black, white);
  const blackX = toFraction(black) * 100;
  const whiteX = toFraction(white) * 100;

  // The map across the whole axis, so the clamped tails outside the window are visible.
  const SAMPLES = 48;
  const strip = Array.from({ length: SAMPLES + 1 }, (_, i) => {
    const fraction = i / SAMPLES;
    const value = axisLo + fraction * axisSpan;
    const at = Math.min(Math.max((value - level) / Math.max(window / 2, 1e-9), -1), 1);
    return `${cssAt(ramp, at)} ${(fraction * 100).toFixed(2)}%`;
  }).join(", ");

  return (
    <div className="space-y-1.5">
      <div className="flex items-baseline justify-between gap-2">
        <Label className="text-xs">Window / level</Label>
        <span
          className="font-mono text-[10px] text-muted-foreground"
          title="Fraction of voxels outside the window — crushed to black or blown out to white"
        >
          {(clipped * 100).toFixed(1)}% clipped
        </span>
        <div className="flex-1" />
        <button
          type="button"
          className="rounded px-1 font-mono text-[10px] text-muted-foreground hover:text-foreground"
          title={`Reset to the automatic window for this image (${auto.rule})`}
          onClick={() => onChange({ colorWindow: auto.colorWindow, colorLevel: auto.colorLevel })}
        >
          auto · {auto.rule}
        </button>
        <button
          type="button"
          className={cn(
            "rounded px-1 font-mono text-[10px]",
            logY ? "bg-primary/20 text-primary" : "text-muted-foreground hover:text-foreground",
          )}
          aria-pressed={logY}
          title="Log-scale the counts. A CT histogram is mostly air, which flattens everything else on a linear axis."
          onClick={() => setLogY(!logY)}
        >
          log Y
        </button>
      </div>

      <div className="space-y-0.5">
        <svg
          ref={plotRef}
          viewBox={`0 0 100 ${PLOT_HEIGHT}`}
          preserveAspectRatio="none"
          className="h-14 w-full cursor-ew-resize touch-none rounded border border-border/60 bg-muted/30"
          onPointerDown={onPointerDown}
          onPointerMove={onPointerMove}
          onPointerUp={endDrag}
          onPointerCancel={endDrag}
        >
          {bars.map((bar, index) => (
            <rect
              key={index}
              x={bar.left}
              width={bar.width}
              y={PLOT_HEIGHT * (1 - bar.h)}
              height={PLOT_HEIGHT * bar.h}
              className="fill-foreground/35"
            />
          ))}
          {/* The transfer function itself: flat, ramp, flat. */}
          <polyline
            points={`0,${PLOT_HEIGHT} ${blackX},${PLOT_HEIGHT} ${whiteX},0 100,0`}
            className="stroke-primary"
            strokeWidth={1}
            fill="none"
            vectorEffect="non-scaling-stroke"
          />
          {[blackX, whiteX].map((x, index) => (
            <line
              key={index}
              x1={x}
              x2={x}
              y1={0}
              y2={PLOT_HEIGHT}
              className="stroke-primary"
              strokeWidth={2}
              vectorEffect="non-scaling-stroke"
            />
          ))}
        </svg>
        <div
          className="h-2 w-full rounded-sm"
          title="How values become grey, across the whole axis. Flat ends are the clamped tails."
          style={{ backgroundImage: `linear-gradient(to right, ${strip})` }}
        />
      </div>

      <div className="flex items-baseline justify-between font-mono text-[10px] text-muted-foreground">
        <span>black {black.toFixed(0)}</span>
        <span>white {white.toFixed(0)}</span>
      </div>
      <p className="font-mono text-[10px] text-muted-foreground">
        window {window.toFixed(0)} · level {level.toFixed(0)}
      </p>
      <p className="text-[10px] leading-snug text-muted-foreground">
        Drag a marker to move one end, drag the plot to move both, scroll to widen or narrow.
      </p>
    </div>
  );
}
