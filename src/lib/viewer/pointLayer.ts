/**
 * The labelled points, drawn and edited in one pane: a 2D canvas over the render window, with
 * dots of a constant size projected with the pane's camera and found again under the pointer
 * in the same screen space, so drawing, picking and dragging agree in every pane. A slice view
 * shows the points on its slice; the 3D pane shows them all, faded behind a slice plane.
 *
 * Pointer handling runs in the capture phase, ahead of vtk.js. A press on a point, or a
 * Shift-drag, belongs here and never reaches the camera; anything else goes on to vtk.js
 * (pan, rotate, the reformat handles) and adds a point if it turns out to be a click. So
 * navigating and placing share the left button.
 *
 * Being the pane's one screen-space canvas, it also draws the Shift-hover crosshair: where the
 * pointer in another slice view is, in this one (`SceneState.sliceSync`).
 */

import type vtkRenderer from "@kitware/vtk.js/Rendering/Core/Renderer";

import { currentImage } from "@/lib/scene/datasets";
import { pointColor, type DataPoint, type SelectHow } from "@/lib/scene/points";
import type { SceneState } from "@/lib/scene/store";
import { isSliceRole, planesForThreeD, scrollStep, type SlicePlane, type ViewportRole } from "@/lib/scene/views";
import { paneRay, paneSize, pickOnPlane, pointerInPane } from "@/lib/viewer/picking";
import { insideConvexPolygon, rayPlane, visiblePolygon, wholePlane } from "@/lib/viewer/planeGeometry";
import { add, dot, insideGrid, projectOntoPlane, scale, sub, type Bounds, type Vec3 } from "@/lib/world";

interface PointLayerCallbacks {
  add: (position: Vec3) => void;
  select: (ids: number[], how: SelectHow) => void;
  /** Where dragged points are now. */
  place: (positions: ReadonlyMap<number, Vec3>) => void;
  /** Whether the pointer is over something else that takes clicks, like a reformat handle. */
  busy?: () => boolean;
}

/** Pixels: how far from a dot a press still grabs it. */
const HIT_RADIUS = 8;
/** Pixels and ms within which a press-and-release is a click rather than a drag. */
const CLICK_SLOP = 4;
const CLICK_MS = 600;
/** A slice view shows points within this fraction of a slice step of its plane. */
const SLICE_BAND = 0.6;
/** Pixels: the Shift-hover crosshair's arm, and the gap it leaves at the centre, so the point itself stays visible. */
const CROSSHAIR_ARM = 9;
const CROSSHAIR_GAP = 3;

interface Drawn {
  point: DataPoint;
  x: number;
  y: number;
  depth: number;
  alpha: number;
}

/** The slice planes the 3D pane draws, each with the part of it that is drawn. */
function shownPlanes(state: SceneState, bounds: Bounds): { plane: SlicePlane; polygon: Vec3[] }[] {
  return planesForThreeD(state).map((role) => {
    const plane = state.slicePlanes[role];
    return { plane, polygon: visiblePolygon(plane, state.sliceExtents[role] ?? wholePlane(bounds), bounds) };
  });
}

type Gesture =
  | {
      kind: "drag";
      /**
       * Each dragged point where the press found it. A move puts them at these plus the drag
       * so far, never at where the last move left them: no drift, and a point held at the
       * image's edge comes back with the pointer.
       */
      origins: Map<number, Vec3>;
      clicked: number;
      wasSelected: boolean;
      plane: { origin: Vec3; normal: Vec3 };
      /** Where the press met the drag plane. */
      grabbed: Vec3;
      start: [number, number];
      moved: boolean;
    }
  | { kind: "rect"; start: [number, number]; end: [number, number]; additive: boolean }
  | { kind: "click"; start: [number, number]; time: number; busy: boolean };

export class PointLayer {
  private readonly canvas: HTMLCanvasElement;
  /** The render window's canvas, which shows the pointer's cursor (this one takes no events). */
  private readonly surface: HTMLCanvasElement | null;
  private state: SceneState | null = null;
  private gesture: Gesture | null = null;
  private hover: number | null = null;
  private drawPending = false;
  private destroyed = false;

  constructor(
    private readonly container: HTMLElement,
    private readonly renderer: vtkRenderer,
    private readonly role: ViewportRole,
    private readonly callbacks: PointLayerCallbacks,
  ) {
    this.surface = container.querySelector("canvas");
    this.canvas = document.createElement("canvas");
    this.canvas.dataset.layer = "points";
    Object.assign(this.canvas.style, { position: "absolute", inset: "0", width: "100%", height: "100%", pointerEvents: "none" });
    container.appendChild(this.canvas);
    for (const [type, listener, capture] of this.listeners()) container.addEventListener(type, listener, { capture });
  }

  private listeners(): [string, EventListener, boolean][] {
    return [
      ["pointerdown", this.onPointerDown as EventListener, true],
      ["pointermove", this.onPointerMove as EventListener, true],
      ["pointerup", this.onPointerUp as EventListener, true],
      ["pointercancel", this.onPointerCancel, true],
      ["pointerleave", this.onPointerLeave, false],
    ];
  }

  sync(state: SceneState): void {
    this.state = state;
    this.showCursor();
    this.requestDraw();
  }

  requestDraw(): void {
    if (this.drawPending || this.destroyed) return;
    this.drawPending = true;
    requestAnimationFrame(() => {
      this.drawPending = false;
      this.draw();
    });
  }

  /** Plain-data snapshot, for tests: where each drawn point is on screen. */
  describe(): { id: number; x: number; y: number; alpha: number }[] {
    return this.drawn().map(({ point, x, y, alpha }) => ({ id: point.id, x, y, alpha }));
  }

  /** Where the Shift-hover crosshair is drawn in this pane, in CSS pixels, or null. For tests. */
  describeCrosshair(): { x: number; y: number } | null {
    const at = this.crosshair();
    return at && { x: at.x, y: at.y };
  }

  draw(): void {
    if (this.destroyed) return;
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    const ratio = window.devicePixelRatio || 1;
    if (this.canvas.width !== Math.round(width * ratio) || this.canvas.height !== Math.round(height * ratio)) {
      this.canvas.width = Math.round(width * ratio);
      this.canvas.height = Math.round(height * ratio);
    }
    const context = this.canvas.getContext("2d");
    if (!context) return;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);
    const selected = new Set(this.state?.selectedPointIds ?? []);

    // Selected last, so they sit on top.
    const drawn = this.drawn().sort((a, b) => Number(selected.has(a.point.id)) - Number(selected.has(b.point.id)));
    for (const { point, x, y, alpha } of drawn) {
      const isSelected = selected.has(point.id);
      context.globalAlpha = alpha;
      context.beginPath();
      context.arc(x, y, isSelected ? 6 : 4.5, 0, Math.PI * 2);
      context.fillStyle = pointColor(point.label);
      context.fill();
      context.lineWidth = isSelected ? 2 : point.id === this.hover ? 1.5 : 1;
      context.strokeStyle = isSelected || point.id === this.hover ? "white" : "rgb(0 0 0 / 0.7)";
      context.stroke();
      if (isSelected) {
        context.font = "600 10px ui-monospace, monospace";
        context.lineWidth = 3;
        context.strokeStyle = "rgb(0 0 0 / 0.75)";
        context.strokeText(`#${point.id}`, x + 8, y - 7);
        context.fillStyle = "white";
        context.fillText(`#${point.id}`, x + 8, y - 7);
      }
    }
    context.globalAlpha = 1;

    const crosshair = this.crosshair();
    if (crosshair) this.drawCrosshair(context, crosshair.x, crosshair.y);

    if (this.gesture?.kind === "rect") {
      const [x0, y0] = this.gesture.start;
      const [x1, y1] = this.gesture.end;
      context.setLineDash([4, 3]);
      context.lineWidth = 1;
      context.strokeStyle = "white";
      context.fillStyle = "rgb(255 255 255 / 0.08)";
      context.fillRect(Math.min(x0, x1), Math.min(y0, y1), Math.abs(x1 - x0), Math.abs(y1 - y0));
      context.strokeRect(Math.min(x0, x1) + 0.5, Math.min(y0, y1) + 0.5, Math.abs(x1 - x0), Math.abs(y1 - y0));
      context.setLineDash([]);
    }
  }

  /**
   * Where a Shift-hover in another slice view falls in this one: its point, on this view's
   * plane, on screen. Not in the hovered view itself, which has the pointer there, nor in 3D,
   * where the three planes meet at the point.
   */
  private crosshair(): { x: number; y: number } | null {
    const state = this.state;
    const sync = state?.sliceSync;
    if (!state || !sync || !isSliceRole(this.role) || sync.from === this.role) return null;
    const plane = state.slicePlanes[this.role];
    const { x, y } = this.project(projectOntoPlane(sync.position, plane.origin, plane.normal));
    const [width, height] = this.size();
    return x < 0 || y < 0 || x > width || y > height ? null : { x, y };
  }

  /** Four thin arms about (x, y), open at the centre, with a dark halo so they read on any slice. */
  private drawCrosshair(context: CanvasRenderingContext2D, x: number, y: number): void {
    const arms: [number, number, number, number][] = [
      [x - CROSSHAIR_ARM, y, x - CROSSHAIR_GAP, y],
      [x + CROSSHAIR_GAP, y, x + CROSSHAIR_ARM, y],
      [x, y - CROSSHAIR_ARM, x, y - CROSSHAIR_GAP],
      [x, y + CROSSHAIR_GAP, x, y + CROSSHAIR_ARM],
    ];
    const px = (v: number) => Math.round(v) + 0.5; // on the pixel grid: crisp 1 px lines
    for (const [width, style] of [
      [3, "rgb(0 0 0 / 0.55)"],
      [1, "rgb(255 255 255 / 0.85)"],
    ] as const) {
      context.lineWidth = width;
      context.strokeStyle = style;
      context.beginPath();
      for (const [x0, y0, x1, y1] of arms) {
        context.moveTo(px(x0), px(y0));
        context.lineTo(px(x1), px(y1));
      }
      context.stroke();
    }
  }

  destroy(): void {
    this.destroyed = true;
    for (const [type, listener, capture] of this.listeners()) this.container.removeEventListener(type, listener, { capture });
    this.canvas.remove();
  }

  // ------------------------------------------------------------------------------- //
  // Geometry
  // ------------------------------------------------------------------------------- //

  private size(): [number, number] {
    return paneSize(this.container);
  }

  private project(p: Vec3): { x: number; y: number; depth: number } {
    const [width, height] = this.size();
    const [nx, ny, nz] = this.renderer.worldToNormalizedDisplay(p[0], p[1], p[2], width / height);
    return { x: nx * width, y: (1 - ny) * height, depth: nz };
  }

  private ray(x: number, y: number): { from: Vec3; direction: Vec3 } | null {
    return paneRay(this.renderer, this.container, x, y);
  }

  private onPlane(x: number, y: number, plane: { origin: Vec3; normal: Vec3 }): Vec3 | null {
    return pickOnPlane(this.renderer, this.container, x, y, plane);
  }

  /** The points this pane shows, where, and how strongly: none where they are hidden. */
  private drawn(): Drawn[] {
    const state = this.state;
    const image = state && currentImage(state);
    if (!state || !image || state.points.length === 0) return [];
    if (!state.pointsVisible[isSliceRole(this.role) ? "slices" : "threeD"]) return [];
    const [width, height] = this.size();
    const out: Drawn[] = [];

    if (isSliceRole(this.role)) {
      const plane = state.slicePlanes[this.role];
      const band = SLICE_BAND * scrollStep(state, plane.normal);
      for (const point of state.points) {
        if (Math.abs(dot(sub(point.position, plane.origin), plane.normal)) > band) continue;
        const { x, y, depth } = this.project(point.position);
        if (x < -HIT_RADIUS || y < -HIT_RADIUS || x > width + HIT_RADIUS || y > height + HIT_RADIUS) continue;
        out.push({ point, x, y, depth, alpha: 1 });
      }
      return out;
    }

    // 3D: faded where a slice plane is between the camera and the point.
    const eye = [...this.renderer.getActiveCamera().getPosition()] as Vec3;
    const planes = shownPlanes(state, image.bounds);
    for (const point of state.points) {
      const { x, y, depth } = this.project(point.position);
      if (depth < 0 || depth > 1) continue;
      const toPoint = sub(point.position, eye);
      const hidden = planes.some(({ plane, polygon }) => {
        if (Math.abs(dot(sub(point.position, plane.origin), plane.normal)) < 1e-3) return false;
        const t = rayPlane(eye, toPoint, plane);
        return t !== null && t > 0 && t < 1 && insideConvexPolygon(add(eye, scale(toPoint, t)), polygon, plane.normal);
      });
      out.push({ point, x, y, depth, alpha: hidden ? 0.35 : 1 });
    }
    return out;
  }

  private hitTest(x: number, y: number): Drawn | null {
    let best: Drawn | null = null;
    let bestScore = Infinity;
    for (const candidate of this.drawn()) {
      const distance = Math.hypot(candidate.x - x, candidate.y - y);
      if (distance > HIT_RADIUS) continue;
      // Nearest on screen; in 3D, ties go to whichever is in front.
      const score = distance + candidate.depth * 1e-3 + (candidate.alpha < 1 ? HIT_RADIUS : 0);
      if (score < bestScore) {
        best = candidate;
        bestScore = score;
      }
    }
    return best;
  }

  /** Where a click at (x, y) puts a new point: on this view's plane, or the nearest plane in 3D. */
  private pick(x: number, y: number): Vec3 | null {
    const state = this.state;
    const image = state && currentImage(state);
    if (!state || !image) return null;
    const tolerance = 0.5 * Math.max(...image.spacing);
    if (isSliceRole(this.role)) {
      const hit = this.onPlane(x, y, state.slicePlanes[this.role]);
      return hit && insideGrid(image, hit, tolerance) ? hit : null;
    }
    const ray = this.ray(x, y);
    if (!ray) return null;
    let best: { t: number; at: Vec3 } | null = null;
    for (const { plane, polygon } of shownPlanes(state, image.bounds)) {
      const t = rayPlane(ray.from, ray.direction, plane);
      if (t === null || t <= 0 || (best && t >= best.t)) continue;
      const at = add(ray.from, scale(ray.direction, t));
      if (insideConvexPolygon(at, polygon, plane.normal)) best = { t, at };
    }
    return best && insideGrid(image, best.at, tolerance) ? best.at : null;
  }

  /** The plane a dragged point moves in: its slice view's plane, or the screen's in 3D. */
  private dragPlane(position: Vec3): { origin: Vec3; normal: Vec3 } {
    if (isSliceRole(this.role) && this.state) return { origin: position, normal: this.state.slicePlanes[this.role].normal };
    const camera = this.renderer.getActiveCamera();
    return { origin: position, normal: [...camera.getDirectionOfProjection()] as Vec3 };
  }

  // ------------------------------------------------------------------------------- //
  // Pointer
  // ------------------------------------------------------------------------------- //

  private local(event: PointerEvent): [number, number] {
    return pointerInPane(this.container, event);
  }

  private claim(event: PointerEvent): void {
    event.stopImmediatePropagation();
    event.preventDefault();
  }

  /** What a press would do here: grab a point, place one, or neither. */
  private showCursor(): void {
    if (!this.surface) return;
    const placing = !!this.state && this.state.placeMode !== "navigate";
    const cursor = this.gesture?.kind === "drag" ? "grabbing" : this.hover !== null ? "pointer" : placing ? "crosshair" : "";
    if (this.surface.style.cursor !== cursor) this.surface.style.cursor = cursor;
  }

  private readonly onPointerDown = (event: PointerEvent): void => {
    if (event.button !== 0 || !this.state) return;
    const at = this.local(event);
    const hit = this.hitTest(...at);
    const toggle = event.metaKey || event.ctrlKey;

    if (hit) {
      this.claim(event);
      if (toggle || event.shiftKey) {
        this.callbacks.select([hit.point.id], "toggle");
        return;
      }
      const selected = this.state.selectedPointIds;
      const wasSelected = selected.includes(hit.point.id);
      if (!wasSelected) this.callbacks.select([hit.point.id], "replace");
      const dragged = new Set(wasSelected ? selected : [hit.point.id]);
      const plane = this.dragPlane(hit.point.position);
      this.gesture = {
        kind: "drag",
        origins: new Map(this.state.points.filter((p) => dragged.has(p.id)).map((p) => [p.id, p.position])),
        clicked: hit.point.id,
        wasSelected,
        plane,
        grabbed: this.onPlane(...at, plane) ?? hit.point.position,
        start: at,
        moved: false,
      };
      this.container.setPointerCapture?.(event.pointerId);
      this.showCursor();
      return;
    }
    if (event.shiftKey) {
      this.claim(event);
      this.gesture = { kind: "rect", start: at, end: at, additive: toggle };
      this.container.setPointerCapture?.(event.pointerId);
      return;
    }
    // Not ours: vtk.js pans or rotates, or a reformat handle takes it. If it ends where it
    // began, it was a click.
    this.gesture = { kind: "click", start: at, time: event.timeStamp, busy: this.callbacks.busy?.() ?? false };
  };

  private readonly onPointerMove = (event: PointerEvent): void => {
    const at = this.local(event);
    const gesture = this.gesture;
    if (gesture?.kind === "drag") {
      this.claim(event);
      if (!gesture.moved && Math.hypot(at[0] - gesture.start[0], at[1] - gesture.start[1]) < CLICK_SLOP) return;
      gesture.moved = true;
      const world = this.onPlane(...at, gesture.plane);
      if (world) {
        const delta = sub(world, gesture.grabbed);
        this.callbacks.place(new Map([...gesture.origins].map(([id, origin]) => [id, add(origin, delta)])));
      }
      return;
    }
    if (gesture?.kind === "rect") {
      this.claim(event);
      gesture.end = at;
      this.requestDraw();
      return;
    }
    if (gesture?.kind === "click") return;
    const hover = this.hitTest(...at)?.point.id ?? null;
    if (hover !== this.hover) {
      this.hover = hover;
      this.requestDraw();
    }
    this.showCursor();
  };

  private readonly onPointerUp = (event: PointerEvent): void => {
    const gesture = this.gesture;
    this.gesture = null;
    if (!gesture || !this.state) return;
    const at = this.local(event);

    if (gesture.kind === "drag") {
      this.claim(event);
      this.container.releasePointerCapture?.(event.pointerId);
      this.showCursor();
      // A click on one of several selected points selects just that one.
      if (!gesture.moved && gesture.wasSelected && gesture.origins.size > 1) this.callbacks.select([gesture.clicked], "replace");
      return;
    }
    if (gesture.kind === "rect") {
      this.claim(event);
      this.container.releasePointerCapture?.(event.pointerId);
      const [x0, y0] = gesture.start;
      const [x1, y1] = gesture.end;
      const inside = this.drawn().filter(
        ({ x, y }) => x >= Math.min(x0, x1) && x <= Math.max(x0, x1) && y >= Math.min(y0, y1) && y <= Math.max(y0, y1),
      );
      this.callbacks.select(
        inside.map(({ point }) => point.id),
        gesture.additive ? "add" : "replace",
      );
      this.requestDraw();
      return;
    }

    const isClick =
      Math.hypot(at[0] - gesture.start[0], at[1] - gesture.start[1]) <= CLICK_SLOP && event.timeStamp - gesture.time <= CLICK_MS;
    if (!isClick || gesture.busy || event.button !== 0) return;
    if (this.state.placeMode === "navigate") {
      if (this.state.selectedPointIds.length > 0) this.callbacks.select([], "replace");
      return;
    }
    const position = this.pick(...at);
    if (position) this.callbacks.add(position);
  };

  private readonly onPointerCancel = (): void => {
    this.gesture = null;
    this.showCursor();
    this.requestDraw();
  };

  private readonly onPointerLeave = (): void => {
    if (this.hover !== null) {
      this.hover = null;
      this.showCursor();
      this.requestDraw();
    }
  };
}
