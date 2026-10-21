/**
 * The panes, and each slice view's plane.
 *
 * **The pane owns the cutting plane, not the node**: a slice node says how a volume is
 * coloured, and the pane decides where it is cut, so one node serves every slice view. Slice
 * views are named by colour, as in 3D Slicer, because what one shows is not fixed: each has
 * its own plane in world space, which starts in a standard orientation and can be put in
 * another or reformatted to any angle.
 */

import type { Rgb } from "@/lib/color";
import {
  add,
  boundsCenter,
  cross,
  dot,
  normalize,
  rotateWith,
  scale,
  sub,
  voxelAxis,
  voxelCenterCorners,
  type Vec3,
} from "@/lib/world";
import { currentImage, images } from "./datasets";
import type { SceneState, StoreSlice } from "./store";

export type ViewportRole = "3d" | "red" | "yellow" | "green";

export const SLICE_ROLES = ["red", "yellow", "green"] as const;
export type SliceRole = (typeof SLICE_ROLES)[number];

/** The panes of the 2x2 layout, in reading order. */
export const LAYOUT: ViewportRole[] = ["3d", "red", "yellow", "green"];

export function isSliceRole(role: ViewportRole): role is SliceRole {
  return role !== "3d";
}

export const ROLE_LABELS: Record<ViewportRole, string> = { "3d": "3D", red: "Red", yellow: "Yellow", green: "Green" };

/** Slicer's red, yellow and green. */
export const SLICE_COLORS: Record<SliceRole, Rgb> = {
  red: [0.953, 0.29, 0.2],
  yellow: [0.929, 0.835, 0.298],
  green: [0.431, 0.69, 0.294],
};

// --------------------------------------------------------------------------- //
// Planes and orientations
// --------------------------------------------------------------------------- //

export type Orientation = "axial" | "coronal" | "sagittal";

export const ORIENTATIONS: Orientation[] = ["axial", "coronal", "sagittal"];

export const ORIENTATION_LABELS: Record<Orientation, string> = { axial: "Axial", coronal: "Coronal", sagittal: "Sagittal" };

/**
 * The standard orientations in LPS: the world axis each cuts across, the normal (from the
 * plane toward the camera) and screen up. Radiological convention, as in Slicer: axial seen
 * from the feet and coronal from the front, so the patient's right is on the screen's left;
 * sagittal seen from the patient's left.
 */
export const ORIENTATION_AXES: Record<Orientation, { axis: 0 | 1 | 2; normal: Vec3; viewUp: Vec3 }> = {
  axial: { axis: 2, normal: [0, 0, -1], viewUp: [0, -1, 0] },
  coronal: { axis: 1, normal: [0, -1, 0], viewUp: [0, 0, 1] },
  sagittal: { axis: 0, normal: [1, 0, 0], viewUp: [0, 0, 1] },
};

/** What each slice view starts in, and what reset goes back to. */
export const DEFAULT_ORIENTATION: Record<SliceRole, Orientation> = { red: "axial", yellow: "coronal", green: "sagittal" };

const AXIS_LETTER = ["L", "P", "S"] as const;
const SCROLL_HINT: Record<Orientation, string> = { axial: "scroll: I–S", coronal: "scroll: A–P", sagittal: "scroll: R–L" };

/** A slice view's cutting plane, in world LPS mm. */
export interface SlicePlane {
  /** A point on the plane: reformatting tilts it about this, scrolling moves it along `normal`. */
  origin: Vec3;
  /** Unit normal, toward the view's camera. */
  normal: Vec3;
  /** Unit in-plane direction that is up on the screen. */
  viewUp: Vec3;
}

/** A Shift-hover: the world point under the pointer, and the slice view it was hovered in. */
export interface SliceSync {
  position: Vec3;
  from: SliceRole;
}

/** What a slice pane shows, published from its camera so the 3D pane can draw as much of its plane. */
export interface SliceExtent {
  /** The pane's centre, its camera's focal point; only the in-plane part means anything. */
  center: Vec3;
  /** Half the pane's height, mm. */
  halfHeight: number;
  /** Width over height. */
  aspect: number;
}

export function planeInOrientation(orientation: Orientation, origin: Vec3): SlicePlane {
  const { normal, viewUp } = ORIENTATION_AXES[orientation];
  return { origin: [...origin], normal: [...normal], viewUp: [...viewUp] };
}

/** Screen right, for a camera looking along -normal with the plane's view up. */
export function planeRight(plane: SlicePlane): Vec3 {
  return cross(plane.viewUp, plane.normal);
}

function nearlyEqual(a: Vec3, b: Vec3, epsilon: number): boolean {
  return Math.abs(a[0] - b[0]) < epsilon && Math.abs(a[1] - b[1]) < epsilon && Math.abs(a[2] - b[2]) < epsilon;
}

/** Same normal and screen up: a camera looks at both the same way. */
export function sameOrientation(a: SlicePlane, b: SlicePlane): boolean {
  return nearlyEqual(a.normal, b.normal, 1e-9) && nearlyEqual(a.viewUp, b.viewUp, 1e-9);
}

function samePlane(a: SlicePlane, b: SlicePlane): boolean {
  return sameOrientation(a, b) && nearlyEqual(a.origin, b.origin, 1e-9);
}

/** The standard orientation a plane is in, or null once it has been reformatted. */
export function orientationOf(plane: SlicePlane): Orientation | null {
  return (
    ORIENTATIONS.find((o) => nearlyEqual(plane.normal, ORIENTATION_AXES[o].normal, 1e-4) && nearlyEqual(plane.viewUp, ORIENTATION_AXES[o].viewUp, 1e-4)) ??
    null
  );
}

export function scrollHint(plane: SlicePlane): string {
  const orientation = orientationOf(plane);
  return orientation ? SCROLL_HINT[orientation] : "scroll: along normal";
}

/**
 * Where a plane is, for its view's corner: the world coordinate it cuts at in a standard
 * orientation, or once reformatted its depth from the volume centre, positive away from the
 * viewer (the way scrolling down moves it).
 */
export function planePositionLabel(plane: SlicePlane, volumeCenter: Vec3): string {
  const orientation = orientationOf(plane);
  if (orientation) {
    const axis = ORIENTATION_AXES[orientation].axis;
    return `${AXIS_LETTER[axis]} ${plane.origin[axis].toFixed(1)} mm`;
  }
  const depth = -dot(sub(plane.origin, volumeCenter), plane.normal);
  return `depth ${depth >= 0 ? "+" : ""}${depth.toFixed(1)} mm`;
}

// --------------------------------------------------------------------------- //
// State
// --------------------------------------------------------------------------- //

export interface ViewsSlice {
  /** Each slice view's plane, independent of the others, as in Slicer. */
  slicePlanes: Record<SliceRole, SlicePlane>;
  /**
   * Bumped to make a slice view fit its camera to the volume again (a reset, or a new
   * standard orientation): an event, so a counter each view compares with the one it fitted for.
   */
  sliceFits: Record<SliceRole, number>;
  /** Which slice views show their reformat handle in the 3D pane. */
  reformatInThreeD: Record<SliceRole, boolean>;
  /** A pane shown alone, or null for the 2x2 layout. */
  maximizedRole: ViewportRole | null;
  /** Which slice planes the 3D pane draws. */
  planesInThreeD: Record<SliceRole, boolean>;
  /** What each slice pane shows, published by the panes from their cameras while they are mounted. */
  sliceExtents: Partial<Record<SliceRole, SliceExtent>>;
  /**
   * Slicer's Shift-hover, while it is going: the point under the pointer, and the view it is
   * hovered in; null once Shift is released or the pointer leaves. The other slice views draw a
   * crosshair at the point while it lasts, and centre on it -- for which it is an event rather
   * than a position, a new object per move that a pane acts on once, because where a camera
   * points is otherwise not in the store (`viewer/activeViewport.ts`).
   */
  sliceSync: SliceSync | null;

  /** Move a view's plane by wheel notches; positive (scrolling down) is away from the viewer. */
  scrollSlice: (role: SliceRole, notches: number) => void;
  /** Put a view in a standard orientation, through the point its plane passes now. */
  setSliceOrientation: (role: SliceRole, orientation: Orientation) => void;
  /** Reformat a view: any plane, from the 3D handle. */
  setSlicePlane: (role: SliceRole, origin: Vec3, normal: Vec3) => void;
  /** Back to the view's default orientation, through the volume centre, fitted. */
  resetSliceView: (role: SliceRole) => void;
  /** Every view reset, for a new image: what the panes showed of the last one is forgotten too. */
  resetViews: () => void;
  /** Move every slice view's plane along its normal until it passes through `position`. */
  jumpSlicesTo: (position: Vec3) => void;
  /**
   * Shift-hover in the view `from`, as in Slicer: every other slice view's plane passes through
   * `position`, and its camera centres on it. The hovered view keeps still: the point is on its
   * plane already, and panning it would run the image away from the pointer.
   */
  syncSlicesTo: (position: Vec3, from: SliceRole) => void;
  /** The Shift-hover is over: the crosshairs go; the views stay where it left them. */
  endSliceSync: () => void;
  setReformatInThreeD: (role: SliceRole, visible: boolean) => void;
  setMaximizedRole: (role: ViewportRole | null) => void;
  setPlaneInThreeD: (role: SliceRole, visible: boolean) => void;
  /** What a slice pane shows, or null once it is gone. */
  setSliceExtent: (role: SliceRole, extent: SliceExtent | null) => void;
}

function perRole<T>(value: (role: SliceRole) => T): Record<SliceRole, T> {
  return Object.fromEntries(SLICE_ROLES.map((role) => [role, value(role)])) as Record<SliceRole, T>;
}

/** Slice planes the 3D pane draws. */
export function planesForThreeD(state: SceneState): SliceRole[] {
  return SLICE_ROLES.filter((role) => state.planesInThreeD[role]);
}

/**
 * One scroll notch along `normal`: the grid's spacing along it when it runs along a grid
 * axis, otherwise its finest spacing, so an oblique plane never skips a voxel.
 */
export function scrollStep(state: SceneState, normal: Vec3): number {
  let step = Infinity;
  for (const grid of images(state)) {
    const along = ([0, 1, 2] as const).find((a) => Math.abs(dot(voxelAxis(grid, a), normal)) > 1 - 1e-6);
    step = Math.min(step, along === undefined ? Math.min(...grid.spacing) : grid.spacing[along]);
  }
  return Number.isFinite(step) ? step : 1;
}

/**
 * `origin` moved along `normal` until the plane cuts the data: between the outermost voxel
 * centres, not the bounds, which reach half a voxel further and would show an empty slice.
 */
function clampToData(state: SceneState, origin: Vec3, normal: Vec3): Vec3 {
  const depths = images(state).flatMap((grid) => voxelCenterCorners(grid).map((corner) => dot(corner, normal)));
  if (depths.length === 0) return origin;
  const depth = dot(origin, normal);
  return add(origin, scale(normal, Math.min(Math.max(depth, Math.min(...depths)), Math.max(...depths)) - depth));
}

/**
 * `origin` moved along `normal` onto the nearest layer of voxel centres, when the plane lies
 * along the image's grid: a slice then shows one layer of voxels, not a blend of two, and a
 * labelmap drawn voxel by voxel shows the same layer as the image. An oblique plane cuts
 * across layers anyway, and is left where it is.
 */
export function onVoxelLayer(state: SceneState, origin: Vec3, normal: Vec3): Vec3 {
  const grid = currentImage(state);
  const a = grid && ([0, 1, 2] as const).find((axis) => Math.abs(dot(voxelAxis(grid, axis), normal)) > 1 - 1e-6);
  if (!grid || a === undefined) return origin;
  const axis = voxelAxis(grid, a);
  const index = dot(sub(origin, grid.gridOrigin), axis) / grid.spacing[a];
  return add(origin, scale(axis, (Math.round(index) - index) * grid.spacing[a]));
}

/** Where a plane through `origin` with `normal` settles: cutting the data, on a voxel layer. */
function settle(state: SceneState, origin: Vec3, normal: Vec3): Vec3 {
  return onVoxelLayer(state, clampToData(state, origin, normal), normal);
}

/**
 * Every view's plane moved along its normal until it passes through `position`, settled. A
 * plane that ends where it is stays the same object, and the record does too when they all do,
 * so a pane that compares identity is not redrawn for nothing.
 */
function planesThrough(state: SceneState, position: Vec3): Record<SliceRole, SlicePlane> {
  let changed = false;
  const planes = perRole((role) => {
    const plane = state.slicePlanes[role];
    const through = add(plane.origin, scale(plane.normal, dot(sub(position, plane.origin), plane.normal)));
    const origin = settle(state, through, plane.normal);
    if (nearlyEqual(origin, plane.origin, 1e-9)) return plane;
    changed = true;
    return { ...plane, origin };
  });
  return changed ? planes : state.slicePlanes;
}

export const viewsSlice: StoreSlice<ViewsSlice> = (set, get) => {
  /** Replace a view's plane; a no-op when nothing changed, as the 3D handle repeats itself. */
  const putPlane = (role: SliceRole, plane: SlicePlane, refit = false) =>
    set((state) => {
      if (!refit && samePlane(state.slicePlanes[role], plane)) return state;
      return {
        slicePlanes: { ...state.slicePlanes, [role]: plane },
        ...(refit ? { sliceFits: { ...state.sliceFits, [role]: state.sliceFits[role] + 1 } } : {}),
      };
    });

  const resetSliceView = (role: SliceRole) => {
    const state = get();
    const plane = planeInOrientation(DEFAULT_ORIENTATION[role], state.worldBounds ? boundsCenter(state.worldBounds) : [0, 0, 0]);
    putPlane(role, { ...plane, origin: settle(state, plane.origin, plane.normal) }, true);
  };

  return {
    slicePlanes: perRole((role) => planeInOrientation(DEFAULT_ORIENTATION[role], [0, 0, 0])),
    sliceFits: perRole(() => 0),
    reformatInThreeD: perRole(() => false),
    maximizedRole: null,
    planesInThreeD: perRole(() => true),
    sliceExtents: {},
    sliceSync: null,

    scrollSlice: (role, notches) => {
      const state = get();
      const plane = state.slicePlanes[role];
      // Away from the viewer is -normal: the normal points at the camera.
      const moved = add(plane.origin, scale(plane.normal, -notches * scrollStep(state, plane.normal)));
      putPlane(role, { ...plane, origin: settle(state, moved, plane.normal) });
    },

    setSliceOrientation: (role, orientation) => {
      const state = get();
      const origin = settle(state, state.slicePlanes[role].origin, ORIENTATION_AXES[orientation].normal);
      putPlane(role, planeInOrientation(orientation, origin), true);
    },

    setSlicePlane: (role, origin, normal) => {
      const unit = normalize(normal);
      if (!unit) return;
      const previous = get().slicePlanes[role];
      // Screen up turns with the plane, then is made exactly perpendicular to it again.
      const turned = rotateWith(previous.viewUp, previous.normal, unit);
      const viewUp = normalize(sub(turned, scale(unit, dot(turned, unit))));
      if (!viewUp) return;
      // Not onto a voxel layer: the handle moves the plane a little at a time, from where it is.
      putPlane(role, { origin: clampToData(get(), [...origin], unit), normal: unit, viewUp });
    },

    resetSliceView,
    resetViews: () => {
      set({ sliceExtents: {}, sliceSync: null });
      SLICE_ROLES.forEach(resetSliceView);
    },

    jumpSlicesTo: (position) =>
      set((state) => {
        const slicePlanes = planesThrough(state, position);
        return slicePlanes === state.slicePlanes ? state : { slicePlanes };
      }),

    // One update for the planes and the event, so the panes see them together: a pane centring
    // on the point reads its plane as already moved through it.
    syncSlicesTo: (position, from) => set((state) => ({ slicePlanes: planesThrough(state, position), sliceSync: { position: [...position], from } })),
    endSliceSync: () => set((state) => (state.sliceSync === null ? state : { sliceSync: null })),

    setReformatInThreeD: (role, visible) => set((state) => ({ reformatInThreeD: { ...state.reformatInThreeD, [role]: visible } })),
    setMaximizedRole: (maximizedRole) => set({ maximizedRole }),
    setPlaneInThreeD: (role, visible) => set((state) => ({ planesInThreeD: { ...state.planesInThreeD, [role]: visible } })),

    setSliceExtent: (role, extent) =>
      set((state) => {
        if (!extent) {
          const { [role]: _gone, ...rest } = state.sliceExtents;
          return { sliceExtents: rest };
        }
        // Written from a camera callback: an unchanged extent must change nothing, or every
        // frame would notify every pane.
        const current = state.sliceExtents[role];
        const close = (a: number, b: number) => Math.abs(a - b) < 1e-3;
        if (
          current &&
          current.center.every((v, axis) => close(v, extent.center[axis])) &&
          close(current.halfHeight, extent.halfHeight) &&
          close(current.aspect, extent.aspect)
        ) {
          return state;
        }
        return { sliceExtents: { ...state.sliceExtents, [role]: extent } };
      }),
  };
};
