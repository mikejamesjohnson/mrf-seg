/**
 * What a slice view draws on its plane, bottom to top, and how each layer stays where it
 * belongs.
 *
 * Coplanar geometry has identical depth, so which is in front would be decided pixel by pixel,
 * and shimmer: each layer is *drawn* a fixed fraction of a millimetre closer to the camera than
 * the one below (`layerOffset`), by moving its actor, never the plane it samples, so every layer
 * shows the voxels of the view's plane itself. That orders opaque layers, but not translucent
 * ones: vtk.js composites those order-independently, which would mix a labelmap with the map
 * under it. So the layers in `FOREGROUND` are drawn by a second renderer, after the first, over
 * what it drew.
 */

import { scale, type Vec3 } from "@/lib/world";

export const LAYER_OFFSET_MM = { image: 0, map: 0.08, labelmap: 0.16, outline: 0.3 } as const;

export type Layer = keyof typeof LAYER_OFFSET_MM;

/** Drawn over everything else in a slice view (`ViewportController`'s foreground renderer). */
export const FOREGROUND: ReadonlySet<Layer> = new Set<Layer>(["labelmap", "outline"]);

/** How far to move a layer's actor along the plane's `normal`, toward the camera (`side` +1 where the normal points). */
export function layerOffset(layer: Layer, normal: Vec3, side: 1 | -1): Vec3 {
  return scale(normal, LAYER_OFFSET_MM[layer] * side);
}
