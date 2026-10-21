/**
 * The bridge between the scene store and vtk.js, one per pane, and the only thing that
 * mutates a render pipeline. It is fed store snapshots outside React's render cycle and
 * diffs the store's nodes against the representations it has mounted, so React never holds a
 * vtk.js object and scrolling a slice re-renders no component.
 *
 * Its **role** decides which node types it draws, how its camera behaves, and what the wheel
 * does. A slice pane's camera follows its view's plane (`SceneState.slicePlanes`), centres on
 * another slice pane's Shift-hover (`SceneState.sliceSync`), and it has a second renderer over
 * the first, sharing its camera, for what must be drawn over everything else (`layers.ts`).
 */

// Side-effect imports register the WebGL backends: Geometry for actors and mappers, Volume
// for image slices.
import "@kitware/vtk.js/Rendering/Profiles/Geometry";
import "@kitware/vtk.js/Rendering/Profiles/Volume";

import type vtkAnnotatedCubeActor from "@kitware/vtk.js/Rendering/Core/AnnotatedCubeActor";
import vtkRenderer from "@kitware/vtk.js/Rendering/Core/Renderer";
import vtkGenericRenderWindow from "@kitware/vtk.js/Rendering/Misc/GenericRenderWindow";
import vtkOrientationMarkerWidget from "@kitware/vtk.js/Interaction/Widgets/OrientationMarkerWidget";

import { currentImage } from "@/lib/scene/datasets";
import type { SelectHow } from "@/lib/scene/points";
import type { SceneState } from "@/lib/scene/store";
import type { NodeId } from "@/lib/scene/types";
import { isSliceRole, planeRight, sameOrientation, type SliceExtent, type SlicePlane, type SliceRole, type SliceSync, type ViewportRole } from "@/lib/scene/views";
import { add, boundsCenter, boundsDiagonal, dot, insideGrid, projectOntoPlane, scale, sub, type Bounds, type Vec3 } from "@/lib/world";
import { lpsCube, orbitStyle, panOnlyStyle, VIEW_DIRECTIONS, type ViewDirection } from "./cameras";
import { pickOnPlane } from "./picking";
import { PointLayer } from "./pointLayer";
import { ReformatHandles } from "./reformatHandles";
import { showsNodeType } from "./redraw";
import { createRepresentation, type Representation, type RepresentationContext } from "./representations";
import { bindShiftHover } from "./shiftHover";
import { bindSliceWheel } from "./wheel";

/** Margin around a volume when a slice pane frames it. */
const FIT_MARGIN = 1.04;

export interface ViewportCallbacks {
  /** Whole wheel notches over a slice pane, positive for scrolling down. */
  onScroll?: (notches: number) => void;
  /** A slice pane's visible rectangle whenever it changes, and null once the pane is gone. */
  onExtent?: (extent: SliceExtent | null) => void;
  /** A slice view's plane, dragged with its reformat handle in the 3D pane. */
  onReformat?: (role: SliceRole, origin: Vec3, normal: Vec3) => void;
  /** A Shift-hover over a slice pane: the world point under the pointer, on the pane's plane and in the image. */
  onSync?: (position: Vec3) => void;
  /** The Shift-hover is over: Shift released, a button down, or the pointer gone from the pane. */
  onSyncEnd?: () => void;
  /** A click that places a point, at the world position it lands on. */
  onAddPoint?: (position: Vec3) => void;
  onSelectPoints?: (ids: number[], how: SelectHow) => void;
  /** Dragged points, at their new positions. */
  onPlacePoints?: (positions: ReadonlyMap<number, Vec3>) => void;
}

const boundsKey = (bounds: Bounds | null) => (bounds ? bounds.map((v) => v.toFixed(3)).join(",") : "none");

/**
 * Each pane's camera as it was when the pane went, so maximising a pane or restoring the
 * layout -- which mounts new panes -- keeps every view where it was. Kept only while it was
 * fitted for what is still shown: a new volume, or a reset, frames afresh.
 */
const keptCameras = new Map<
  ViewportRole,
  {
    fittedFor: string;
    aimedAt: SlicePlane | null;
    position: Vec3;
    focalPoint: Vec3;
    viewUp: Vec3;
    parallelProjection: boolean;
    parallelScale: number;
    clippingRange: [number, number];
  }
>();

/** What vtk.js's WebGL render window has but does not declare. */
interface WebGLWindow {
  getContext?: () => WebGLRenderingContext | WebGL2RenderingContext | null;
  releaseGraphicsResources?: () => void;
}

export class ViewportController {
  private readonly genericWindow: vtkGenericRenderWindow;
  private readonly renderer: vtkRenderer;
  /** Slice panes only: drawn after `renderer`, over it (`layers.ts`). */
  private readonly foreground: vtkRenderer | null = null;
  /** 3D pane only: the orientation cube, and the slice views' reformat handles. */
  private readonly cube: { widget: vtkOrientationMarkerWidget; actor: vtkAnnotatedCubeActor } | null = null;
  private readonly reformatHandles: ReformatHandles | null = null;
  private readonly pointLayer: PointLayer;
  private readonly resizeObserver: ResizeObserver;
  private readonly unbindWheel: (() => void) | null = null;
  private readonly unbindHover: (() => void) | null = null;
  private readonly representations = new Map<NodeId, Representation>();
  private readonly ctx: RepresentationContext;

  private state: SceneState | null = null;
  private renderPending = false;
  private destroyed = false;
  /** The scene bounds, and in a slice pane the fit request, the camera was last fitted for. */
  private fittedFor: string | null = null;
  /** The plane a slice pane's camera was last pointed at. */
  private aimedAt: SlicePlane | null = null;
  /** The Shift-hover a slice pane last centred on; undefined before its first sync, which takes the one pending as already seen. */
  private syncedTo: SliceSync | null | undefined = undefined;

  constructor(
    private readonly container: HTMLElement,
    readonly role: ViewportRole,
    private readonly callbacks: ViewportCallbacks,
  ) {
    this.genericWindow = vtkGenericRenderWindow.newInstance({
      // Near-neutral: a tinted surround shifts how greyscale reads, and so the window/level chosen.
      background: role === "3d" ? [0.03, 0.032, 0.038] : [0.014, 0.015, 0.019],
      // The ResizeObserver below also catches a side panel changing the canvas width.
      listenWindowResize: false,
    });
    this.genericWindow.setContainer(container);
    this.renderer = this.genericWindow.getRenderer();
    const interactor = this.genericWindow.getInteractor();
    const camera = this.renderer.getActiveCamera();

    if (isSliceRole(role)) {
      // A higher layer keeps what the first renderer drew (setLayer is what says so) and
      // clears only depth. Not interactive, so the camera's manipulators find the first.
      const foreground = vtkRenderer.newInstance();
      foreground.setLayer(1);
      foreground.setInteractive(false);
      foreground.setActiveCamera(camera);
      const renderWindow = this.genericWindow.getRenderWindow();
      renderWindow.setNumberOfLayers(2);
      renderWindow.addRenderer(foreground);
      this.foreground = foreground;

      interactor.setInteractorStyle(panOnlyStyle());
      this.unbindWheel = bindSliceWheel(container, {
        scroll: (notches) => callbacks.onScroll?.(notches),
        zoom: (factor) => {
          const current = camera.getParallelScale();
          if (current > 0) camera.setParallelScale(current * factor);
          this.requestRender();
        },
      });
      // From the camera rather than `sync`: panning and zooming do not touch the store.
      camera.onModified(() => this.publishExtent());
      this.unbindHover = bindShiftHover(container, { hover: (x, y) => this.syncOthersTo(x, y), end: () => callbacks.onSyncEnd?.() });
    } else {
      interactor.setInteractorStyle(orbitStyle());
      const actor = lpsCube();
      const widget = vtkOrientationMarkerWidget.newInstance({ actor, interactor });
      widget.setViewportCorner(vtkOrientationMarkerWidget.Corners.BOTTOM_RIGHT);
      widget.setViewportSize(0.16);
      widget.setMinPixelSize(60);
      widget.setMaxPixelSize(140);
      widget.setEnabled(true);
      this.cube = { widget, actor };
      this.reformatHandles = new ReformatHandles(this.renderer, (slice, origin, normal) => callbacks.onReformat?.(slice, origin, normal));
    }
    container.addEventListener("contextmenu", this.onContextMenu);

    this.ctx = {
      renderer: this.renderer,
      foreground: this.foreground ?? this.renderer,
      role,
      requestRender: () => this.requestRender(),
      getState: () => this.state as SceneState,
    };

    // After vtk.js has bound its pointer handlers: the layer's run ahead of them, in the
    // capture phase, and claim only what is aimed at a point.
    this.pointLayer = new PointLayer(container, this.renderer, role, {
      add: (position) => callbacks.onAddPoint?.(position),
      select: (ids, how) => callbacks.onSelectPoints?.(ids, how),
      place: (positions) => callbacks.onPlacePoints?.(positions),
      busy: () => this.reformatHandles?.pointerOnHandle() ?? false,
    });
    // The dots follow the camera, which vtk.js moves without asking.
    camera.onModified(() => this.pointLayer.requestDraw());

    this.resizeObserver = new ResizeObserver(() => this.resize());
    this.resizeObserver.observe(container);
    this.resize();
  }

  private readonly onContextMenu = (event: MouseEvent) => event.preventDefault();

  /** Apply a store snapshot. Safe to call at any frequency. */
  sync(state: SceneState): void {
    if (this.destroyed) return;
    this.state = state;

    for (const [nodeId, representation] of this.representations) {
      const node = state.nodes[nodeId];
      if (!node || !showsNodeType(this.role, node.type)) {
        representation.destroy();
        this.representations.delete(nodeId);
      }
    }

    // A new volume re-frames the camera (a new image, even on the same grid: the key names
    // it); a slice pane also re-fits when asked to.
    const bounds = state.worldBounds;
    const fitKey = `${state.datasetOrder[0] ?? ""}|${boundsKey(bounds)}|${isSliceRole(this.role) ? state.sliceFits[this.role] : 0}`;
    if (this.fittedFor === null) this.restoreCamera(fitKey);
    const refit = bounds !== null && fitKey !== this.fittedFor;
    if (refit) this.fittedFor = fitKey;

    // Cameras before representations, so whatever reads the camera sees it aimed.
    if (isSliceRole(this.role)) {
      const plane = state.slicePlanes[this.role];
      if (refit && bounds) this.fitSliceCamera(plane, bounds);
      else if (this.aimedAt && !sameOrientation(this.aimedAt, plane)) this.turnSliceCamera(this.aimedAt, plane, bounds);
      this.aimedAt = plane;
      // A Shift-hover in another slice pane: centre on its point, keeping the zoom. The plane
      // already passes through it (the store moved both at once), so this is a pan.
      if (this.syncedTo === undefined) this.syncedTo = state.sliceSync;
      else if (state.sliceSync && state.sliceSync !== this.syncedTo) {
        this.syncedTo = state.sliceSync;
        if (state.sliceSync.from !== this.role) this.aimSliceCamera(plane, projectOntoPlane(state.sliceSync.position, plane.origin, plane.normal), bounds);
      }
    }

    for (const nodeId of state.nodeOrder) {
      const node = state.nodes[nodeId];
      if (!node || !showsNodeType(this.role, node.type)) continue;
      let representation = this.representations.get(nodeId);
      if (!representation) {
        representation = createRepresentation(node, this.ctx);
        this.representations.set(nodeId, representation);
      }
      representation.update(node);
    }

    this.reformatHandles?.sync(state);
    this.pointLayer.sync(state);
    if (refit && this.role === "3d") this.setViewDirection("oblique");
    if (isSliceRole(this.role)) this.renderer.resetCameraClippingRange(bounds ?? undefined);
    this.requestRender();
  }

  /** The camera this pane's role had when its last pane went, if it was fitted for `fitKey`. */
  private restoreCamera(fitKey: string): void {
    const kept = keptCameras.get(this.role);
    if (!kept || kept.fittedFor !== fitKey) return;
    const camera = this.renderer.getActiveCamera();
    camera.setParallelProjection(kept.parallelProjection);
    camera.setFocalPoint(...kept.focalPoint);
    camera.setPosition(...kept.position);
    camera.setViewUp(...kept.viewUp);
    camera.setParallelScale(kept.parallelScale);
    camera.setClippingRange(...kept.clippingRange);
    this.fittedFor = fitKey;
    this.aimedAt = kept.aimedAt;
  }

  private keepCamera(): void {
    if (this.fittedFor === null) return;
    const camera = this.renderer.getActiveCamera();
    keptCameras.set(this.role, {
      fittedFor: this.fittedFor,
      aimedAt: this.aimedAt,
      position: [...camera.getPosition()] as Vec3,
      focalPoint: [...camera.getFocalPoint()] as Vec3,
      viewUp: [...camera.getViewUp()] as Vec3,
      parallelProjection: camera.getParallelProjection(),
      parallelScale: camera.getParallelScale(),
      clippingRange: [...camera.getClippingRange()] as [number, number],
    });
  }

  /**
   * Point a slice pane's camera at its plane and fit the volume's cross-section: when a volume
   * arrives, on reset, on a new standard orientation -- never on a scroll. The plane slides
   * through a fixed view, as in any DICOM viewer, so pan and zoom survive scrolling.
   */
  private fitSliceCamera(plane: SlicePlane, bounds: Bounds): void {
    this.aimSliceCamera(plane, projectOntoPlane(boundsCenter(bounds), plane.origin, plane.normal), bounds);
    const extentAlong = (v: Vec3) =>
      Math.abs(v[0]) * (bounds[1] - bounds[0]) + Math.abs(v[1]) * (bounds[3] - bounds[2]) + Math.abs(v[2]) * (bounds[5] - bounds[4]);
    const aspect = this.container.clientWidth / Math.max(this.container.clientHeight, 1) || 1;
    const halfHeight = Math.max(extentAlong(plane.viewUp) / 2, extentAlong(planeRight(plane)) / (2 * aspect), 1);
    this.renderer.getActiveCamera().setParallelScale(FIT_MARGIN * halfHeight);
  }

  /**
   * Follow a tilted plane, keeping zoom and pan: the view centre keeps its place relative to
   * the plane's origin, in the plane's own frame, so tilting turns the view about that point.
   */
  private turnSliceCamera(from: SlicePlane, to: SlicePlane, bounds: Bounds | null): void {
    const offset = sub([...this.renderer.getActiveCamera().getFocalPoint()] as Vec3, from.origin);
    const focal = add(add(to.origin, scale(planeRight(to), dot(offset, planeRight(from)))), scale(to.viewUp, dot(offset, from.viewUp)));
    this.aimSliceCamera(to, focal, bounds);
  }

  private aimSliceCamera(plane: SlicePlane, focal: Vec3, bounds: Bounds | null): void {
    const camera = this.renderer.getActiveCamera();
    camera.setParallelProjection(true);
    camera.setFocalPoint(...focal);
    camera.setPosition(...add(focal, scale(plane.normal, bounds ? boundsDiagonal(bounds) : 200)));
    camera.setViewUp(...plane.viewUp);
    this.renderer.resetCameraClippingRange(bounds ?? undefined);
  }

  /**
   * A Shift-hover at (x, y) in this slice pane: the other slice views follow the point under
   * the pointer, if it is in the image. Where there is no image under it there is nothing to
   * look at, so nothing moves.
   */
  private syncOthersTo(x: number, y: number): void {
    const state = this.state;
    const image = state && currentImage(state);
    if (!state || !image || !isSliceRole(this.role)) return;
    const hit = pickOnPlane(this.renderer, this.container, x, y, state.slicePlanes[this.role]);
    if (hit && insideGrid(image, hit, 0.5 * Math.max(...image.spacing))) this.callbacks.onSync?.(hit);
  }

  private publishExtent(): void {
    const height = this.container.clientHeight;
    if (this.destroyed || !this.callbacks.onExtent || !(height > 0)) return;
    const camera = this.renderer.getActiveCamera();
    this.callbacks.onExtent({
      center: [...camera.getFocalPoint()] as Vec3,
      halfHeight: camera.getParallelScale(),
      aspect: this.container.clientWidth / height,
    });
  }

  /** Frame the volume again, without changing the view direction. */
  resetCamera(): void {
    const bounds = this.state?.worldBounds;
    if (isSliceRole(this.role)) {
      if (this.state && bounds) this.fitSliceCamera(this.state.slicePlanes[this.role], bounds);
    } else {
      this.renderer.resetCamera(bounds ?? undefined);
    }
    this.requestRender();
  }

  /** The 3D pane's camera, looking from `direction`. */
  setViewDirection(direction: ViewDirection): void {
    if (isSliceRole(this.role)) return;
    const bounds = this.state?.worldBounds;
    const { position, viewUp } = VIEW_DIRECTIONS[direction];
    const distance = (bounds ? boundsDiagonal(bounds) : 200) / (Math.hypot(...position) || 1);
    const camera = this.renderer.getActiveCamera();
    camera.setParallelProjection(false);
    camera.setFocalPoint(0, 0, 0);
    camera.setPosition(...scale(position, distance));
    camera.setViewUp(...viewUp);
    this.renderer.resetCamera(bounds ?? undefined);
    this.requestRender();
  }

  /** Where a world point lands in this pane, in CSS pixels from its top left. For tests. */
  worldToPane(point: Vec3): [number, number] {
    const width = this.container.clientWidth;
    const height = this.container.clientHeight;
    const [x, y] = this.renderer.worldToNormalizedDisplay(point[0], point[1], point[2], width / Math.max(height, 1));
    return [x * width, (1 - y) * height];
  }

  /** Plain-data snapshot of the render state, for answering "is this where I think?". */
  describe(): Record<string, unknown> {
    const camera = this.renderer.getActiveCamera();
    return {
      role: this.role,
      camera: {
        position: [...camera.getPosition()],
        focalPoint: [...camera.getFocalPoint()],
        viewUp: [...camera.getViewUp()],
        parallelProjection: camera.getParallelProjection(),
        parallelScale: camera.getParallelScale(),
      },
      points: this.pointLayer.describe(),
      crosshair: this.pointLayer.describeCrosshair(),
      nodes: [...this.representations].map(([nodeId, representation]) => ({
        nodeId,
        type: this.state?.nodes[nodeId]?.type,
        ...(representation.describe?.() ?? {}),
      })),
    };
  }

  resize(): void {
    if (this.destroyed) return;
    this.genericWindow.resize();
    this.publishExtent();
    this.requestRender();
  }

  /** At most one render per frame; the points are drawn after it. */
  private requestRender(): void {
    if (this.renderPending || this.destroyed) return;
    this.renderPending = true;
    requestAnimationFrame(() => {
      this.renderPending = false;
      if (this.destroyed) return;
      this.genericWindow.getRenderWindow().render();
      this.pointLayer.draw();
    });
  }

  /**
   * Take everything down, including what vtk.js leaves behind: its window's delete neither
   * frees the GPU nor deletes the interactor, whose document listener would keep the whole
   * pipeline alive, and browsers allow only so many WebGL contexts.
   */
  destroy(): void {
    if (this.destroyed) return;
    this.keepCamera();
    this.destroyed = true;
    if (isSliceRole(this.role)) this.callbacks.onExtent?.(null);
    this.unbindWheel?.();
    this.unbindHover?.();
    this.container.removeEventListener("contextmenu", this.onContextMenu);
    this.resizeObserver.disconnect();
    for (const representation of this.representations.values()) representation.destroy();
    this.representations.clear();
    this.reformatHandles?.destroy();
    this.pointLayer.destroy();
    if (this.cube) {
      this.cube.widget.setEnabled(false);
      this.cube.widget.delete();
      this.cube.actor.delete();
    }

    const webgl = this.genericWindow.getApiSpecificRenderWindow() as unknown as WebGLWindow;
    const context = webgl.getContext?.() ?? null;
    webgl.releaseGraphicsResources?.();
    const interactor = this.genericWindow.getInteractor();
    this.genericWindow.delete();
    interactor.delete();
    this.foreground?.delete();
    context?.getExtension("WEBGL_lose_context")?.loseContext();
  }
}
