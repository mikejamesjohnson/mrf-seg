/** The labelled points the segmentation is made from (clamped in the CRF), and their selection. */

import { clampToGrid, type Vec3 } from "@/lib/world";
import { currentImage } from "./datasets";
import type { StoreSlice } from "./store";

/** +1: inside what is being segmented. -1: outside it. */
export type PointLabel = 1 | -1;

export interface DataPoint {
  /** Fixed for the point's life: #7 stays #7 when #3 is deleted. */
  id: number;
  label: PointLabel;
  /** World position, LPS mm, between the image's outermost voxel centres. */
  position: Vec3;
}

/** What a click in a view does: nothing new, or add an inside or outside point. */
export type PlaceMode = "navigate" | "inside" | "outside";

export type SelectHow = "replace" | "add" | "toggle";

/** Where points can be shown: the 3D pane, and the slice views. */
export type PointViews = "threeD" | "slices";

/** Blue inside, red outside, wherever a point is drawn. */
export function pointColor(label: PointLabel): string {
  return label > 0 ? "rgb(59 130 246)" : "rgb(239 68 68)";
}

export function pointCounts(points: DataPoint[]): { inside: number; outside: number } {
  const inside = points.filter((point) => point.label > 0).length;
  return { inside, outside: points.length - inside };
}

export interface PointsSlice {
  /** In the order they were added. */
  points: DataPoint[];
  nextPointId: number;
  selectedPointIds: number[];
  /** Bumped by every change to the points, so a segmentation can tell it is out of date. */
  pointsVersion: number;
  placeMode: PlaceMode;
  /** Whether the points are drawn there. Hidden ones cannot be picked or dragged there either. */
  pointsVisible: Record<PointViews, boolean>;

  /** A point at `position` with the place mode's label, kept inside the image; its id, or null. */
  addPoint: (position: Vec3) => number | null;
  /** Put points at new positions, each kept inside the image. */
  setPointPositions: (positions: ReadonlyMap<number, Vec3>) => void;
  removePoints: (ids: number[]) => void;
  clearPoints: () => void;
  selectPoints: (ids: number[], how: SelectHow) => void;
  setPlaceMode: (mode: PlaceMode) => void;
  setPointsVisible: (where: PointViews, visible: boolean) => void;
}

export const pointsSlice: StoreSlice<PointsSlice> = (set, get) => {
  /** New points, in one store update with anything that goes with them. */
  const changePoints = (points: DataPoint[], also: Partial<PointsSlice> = {}) =>
    set((state) => ({ points, pointsVersion: state.pointsVersion + 1, ...also }));

  return {
    points: [],
    nextPointId: 1,
    selectedPointIds: [],
    pointsVersion: 0,
    placeMode: "navigate",
    pointsVisible: { threeD: true, slices: true },

    addPoint: (position) => {
      const state = get();
      const image = currentImage(state);
      if (!image || state.placeMode === "navigate") return null;
      const point: DataPoint = { id: state.nextPointId, label: state.placeMode === "inside" ? 1 : -1, position: clampToGrid(image, position) };
      changePoints([...state.points, point], { nextPointId: point.id + 1 });
      return point.id;
    },

    setPointPositions: (positions) => {
      const state = get();
      const image = currentImage(state);
      if (!image || !state.points.some((p) => positions.has(p.id))) return;
      changePoints(state.points.map((p) => (positions.has(p.id) ? { ...p, position: clampToGrid(image, positions.get(p.id)!) } : p)));
    },

    removePoints: (ids) => {
      if (ids.length === 0) return;
      const removing = new Set(ids);
      const state = get();
      changePoints(
        state.points.filter((p) => !removing.has(p.id)),
        { selectedPointIds: state.selectedPointIds.filter((id) => !removing.has(id)) },
      );
    },

    clearPoints: () => changePoints([], { nextPointId: 1, selectedPointIds: [] }),

    selectPoints: (ids, how) =>
      set((state) => {
        if (how === "replace") return { selectedPointIds: [...new Set(ids)] };
        const selected = new Set(state.selectedPointIds);
        for (const id of ids) {
          if (how === "toggle" && selected.has(id)) selected.delete(id);
          else selected.add(id);
        }
        return { selectedPointIds: [...selected] };
      }),

    setPlaceMode: (placeMode) => set({ placeMode }),
    setPointsVisible: (where, visible) => set((state) => ({ pointsVisible: { ...state.pointsVisible, [where]: visible } })),
  };
};
