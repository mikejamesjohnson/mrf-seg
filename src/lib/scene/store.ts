/**
 * The scene store: plain, serialisable data only. vtk.js objects live in `dataRegistry.ts`, and
 * the render pipelines in `viewer/ViewportController.ts`, which subscribes to this outside
 * React. One slice per concern, each in the file that owns the concept.
 */

import { create, type StateCreator } from "zustand";

import { segmentationSlice, type SegmentationSlice } from "@/lib/segmentation/state";
import { datasetsSlice, type DatasetsSlice } from "./datasets";
import { pointsSlice, type PointsSlice } from "./points";
import { viewsSlice, type ViewsSlice } from "./views";

export type SceneState = DatasetsSlice & ViewsSlice & PointsSlice & SegmentationSlice;

/** A slice of the store: its own state and actions, with `set` and `get` on the whole. */
export type StoreSlice<T> = StateCreator<SceneState, [], [], T>;

export const useSceneStore = create<SceneState>()((...args) => ({
  ...datasetsSlice(...args),
  ...viewsSlice(...args),
  ...pointsSlice(...args),
  ...segmentationSlice(...args),
}));
