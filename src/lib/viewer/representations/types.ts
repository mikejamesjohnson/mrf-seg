/**
 * The renderer-side counterpart of a `SceneNode`: a `Representation` owns the vtk.js pipeline
 * for one node in one pane. The controller creates it, calls `update` on every store change and
 * `destroy` when the node goes. It reads plain node data and the data registry, never writes
 * the store, and keys any cached input on `inputKey`.
 */

import type vtkRenderer from "@kitware/vtk.js/Rendering/Core/Renderer";

import type { SceneState } from "@/lib/scene/store";
import type { Dataset, SceneNode } from "@/lib/scene/types";
import type { ViewportRole } from "@/lib/scene/views";

export interface RepresentationContext {
  renderer: vtkRenderer;
  /**
   * Drawn after `renderer`, over it, sharing its camera: where a slice view puts the layers in
   * `FOREGROUND` (`viewer/layers.ts`). In the 3D pane it is `renderer`.
   */
  foreground: vtkRenderer;
  /** The pane drawn into; slices use it to pick their plane. */
  role: ViewportRole;
  /** Coalesced re-render request; safe to call from anywhere. */
  requestRender: () => void;
  getState: () => SceneState;
}

/** A dataset's identity *and* version: one replaced under the same id must be re-read. */
export function inputKey(dataset: Dataset | undefined): string {
  return dataset ? `${dataset.id}:${dataset.version}` : "none";
}

export interface Representation<TNode extends SceneNode = SceneNode> {
  update: (node: TNode) => void;
  /** Detach the actors and release what they hold. */
  destroy: () => void;
  /** Plain-data snapshot of what it put in the scene, for debugging and tests. */
  describe?: () => Record<string, unknown>;
}

export type RepresentationFactory<TNode extends SceneNode = SceneNode> = (node: TNode, ctx: RepresentationContext) => Representation<TNode>;
