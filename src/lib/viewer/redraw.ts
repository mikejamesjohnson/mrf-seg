/**
 * Which store changes a pane draws from, by its role. A change that touches none of them (a
 * job's progress, a status message, another slice pane's plane) leaves the pane as it was, so it
 * is not synced or rendered for. Pure, so it can be tested without vtk.js.
 *
 * A representation that reads anything else from the store must add it here, and
 * `showsNodeType` says which panes draw each node type.
 */

import type { SceneState } from "@/lib/scene/store";
import type { SceneNodeType } from "@/lib/scene/types";
import { isSliceRole, type ViewportRole } from "@/lib/scene/views";

/** What every pane draws from. */
const EVERY_PANE = ["datasets", "datasetOrder", "nodes", "nodeOrder", "worldBounds", "points", "selectedPointIds", "placeMode", "pointsVisible"] as const satisfies readonly (keyof SceneState)[];

/** What only the 3D pane draws from: where each slice pane is looking, and what is shown of it there. */
const THREE_D_ONLY = ["sliceExtents", "reformatInThreeD", "planesInThreeD"] as const satisfies readonly (keyof SceneState)[];

/** Everything any pane draws from. */
export const DRAWN_FROM = [...EVERY_PANE, ...THREE_D_ONLY, "slicePlanes", "sliceFits", "sliceSync"] as const satisfies readonly (keyof SceneState)[];

/** Whether a pane of `role` shows something else after the change from `previous` to `state`. */
export function drawsDifferently(role: ViewportRole, state: SceneState, previous: SceneState): boolean {
  if (EVERY_PANE.some((key) => state[key] !== previous[key])) return true;
  if (isSliceRole(role)) {
    // Its own plane and fit: another slice pane scrolling changes nothing here. A Shift-hover
    // in another slice pane centres this one on its point; one in this pane leaves it still.
    return (
      state.slicePlanes[role] !== previous.slicePlanes[role] ||
      state.sliceFits[role] !== previous.sliceFits[role] ||
      (state.sliceSync !== previous.sliceSync && state.sliceSync?.from !== role)
    );
  }
  return state.slicePlanes !== previous.slicePlanes || THREE_D_ONLY.some((key) => state[key] !== previous[key]);
}

/** Everything everywhere -- the surface in a slice view as its outline -- but the bounding box, which is 3D only. */
export function showsNodeType(role: ViewportRole, type: SceneNodeType): boolean {
  return role === "3d" || type !== "outline";
}
