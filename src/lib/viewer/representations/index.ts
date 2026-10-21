/**
 * Node type -> representation. Drawing something new means a node type in `scene/types.ts`
 * (with its `NODE_TYPES_FOR_KIND` and `NODE_TYPE_LABELS` entries and a `defaultNode` case), a
 * factory here, which panes draw it (`redraw.showsNodeType`), and, if it reads store state no
 * representation reads yet, that state in `redraw.ts`.
 */

import type { SceneNode, SceneNodeType } from "@/lib/scene/types";
import { createImageSliceRepresentation } from "./imageSlice";
import { createOutlineRepresentation } from "./outline";
import { createSurfaceRepresentation } from "./surface";
import type { Representation, RepresentationContext, RepresentationFactory } from "./types";

const FACTORIES: { [K in SceneNodeType]: RepresentationFactory<Extract<SceneNode, { type: K }>> } = {
  imageSlice: createImageSliceRepresentation,
  // Cut by the same planes as the image; only the colouring differs.
  overlay: createImageSliceRepresentation,
  labelmap: createImageSliceRepresentation,
  outline: createOutlineRepresentation,
  surface: createSurfaceRepresentation,
};

export function createRepresentation(node: SceneNode, ctx: RepresentationContext): Representation {
  // Each factory takes exactly its own node variant, which TypeScript cannot see through the union.
  return (FACTORIES[node.type] as RepresentationFactory)(node, ctx);
}

export type { Representation, RepresentationContext } from "./types";
