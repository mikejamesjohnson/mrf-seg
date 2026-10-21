/**
 * Geometry on a slice plane: the part of it drawn in the 3D pane (the rectangle its view
 * shows, clipped to the volume), where a mesh crosses it, and the ray and polygon tests the
 * point layer picks with.
 */

import { planeRight, type SliceExtent, type SlicePlane } from "@/lib/scene/views";
import { boundsCenter, boundsDiagonal, cross, dot, projectOntoPlane, sub, type Bounds, type Vec3 } from "@/lib/world";

type Polygon2D = [number, number][];

/** Clip a convex polygon to the half-plane `a*u + b*w + c >= 0` (Sutherland-Hodgman). */
function clipHalfPlane(polygon: Polygon2D, a: number, b: number, c: number): Polygon2D {
  const out: Polygon2D = [];
  for (let i = 0; i < polygon.length; i += 1) {
    const current = polygon[i];
    const next = polygon[(i + 1) % polygon.length];
    const dCurrent = a * current[0] + b * current[1] + c;
    const dNext = a * next[0] + b * next[1] + c;
    if (dCurrent >= 0) out.push(current);
    if (dCurrent >= 0 !== dNext >= 0) {
      const t = dCurrent / (dCurrent - dNext);
      out.push([current[0] + (next[0] - current[0]) * t, current[1] + (next[1] - current[1]) * t]);
    }
  }
  return out;
}

/** Until its view has published what it shows: the plane's whole cross-section. */
export function wholePlane(bounds: Bounds): SliceExtent {
  return { center: boundsCenter(bounds), halfHeight: boundsDiagonal(bounds), aspect: 1 };
}

/** The view's rectangle on the plane, clipped to `bounds`, as world-space corners (convex). */
export function visiblePolygon(plane: SlicePlane, extent: SliceExtent, bounds: Bounds): Vec3[] {
  const up = plane.viewUp;
  const right = planeRight(plane);
  // The camera looks along the normal, so how far in front of the plane its focal point
  // sits is not seen: only where it falls on the plane.
  const center = projectOntoPlane(extent.center, plane.origin, plane.normal);
  const halfHeight = extent.halfHeight;
  const halfWidth = halfHeight * Math.max(extent.aspect, 1e-3);
  let polygon: Polygon2D = [
    [-halfWidth, -halfHeight],
    [halfWidth, -halfHeight],
    [halfWidth, halfHeight],
    [-halfWidth, halfHeight],
  ];
  for (let axis = 0; axis < 3 && polygon.length > 0; axis += 1) {
    polygon = clipHalfPlane(polygon, right[axis], up[axis], center[axis] - bounds[axis * 2]);
    polygon = clipHalfPlane(polygon, -right[axis], -up[axis], bounds[axis * 2 + 1] - center[axis]);
  }
  return polygon.map(([u, w]) => [
    center[0] + right[0] * u + up[0] * w,
    center[1] + right[1] * u + up[1] * w,
    center[2] + right[2] * u + up[2] * w,
  ]);
}

/** Whether `point`, on the polygon's plane, lies inside the convex `polygon`. */
export function insideConvexPolygon(point: Vec3, polygon: Vec3[], normal: Vec3): boolean {
  if (polygon.length < 3) return false;
  let sign = 0;
  for (let i = 0; i < polygon.length; i += 1) {
    const a = polygon[i];
    const b = polygon[(i + 1) % polygon.length];
    const edge = sub(b, a);
    const toPoint = sub(point, a);
    const side = dot(cross(edge, toPoint), normal);
    if (Math.abs(side) < 1e-9) continue;
    if (sign === 0) sign = Math.sign(side);
    else if (Math.sign(side) !== sign) return false;
  }
  return true;
}

/**
 * Where a triangle mesh crosses a plane, as line segments: two (x, y, z) endpoints each.
 * `cells` is vtk.js's cell array, [3, a, b, c, ...]; other cells are skipped. No merging of
 * shared endpoints (vtk.js's cutter does that pairwise, which grows as n squared): one pass
 * counts the crossed triangles, a second writes their segments.
 */
export function meshPlaneSegments(points: ArrayLike<number>, cells: ArrayLike<number>, plane: { origin: Vec3; normal: Vec3 }): Float32Array {
  const [nx, ny, nz] = plane.normal;
  const offset = dot(plane.origin, plane.normal);
  // Which side each point is on; a point on the plane counts as below, consistently.
  const above = new Uint8Array(points.length / 3);
  const distance = new Float64Array(above.length);
  for (let i = 0; i < above.length; i += 1) {
    distance[i] = points[3 * i] * nx + points[3 * i + 1] * ny + points[3 * i + 2] * nz - offset;
    above[i] = distance[i] > 0 ? 1 : 0;
  }
  const crossed = (c: number) => {
    if (cells[c] !== 3) return false;
    const sides = above[cells[c + 1]] + above[cells[c + 2]] + above[cells[c + 3]];
    return sides > 0 && sides < 3;
  };

  let count = 0;
  for (let c = 0; c < cells.length; c += cells[c] + 1) if (crossed(c)) count += 1;
  const out = new Float32Array(count * 6);
  let at = 0;
  const crossing = (a: number, b: number) => {
    const t = distance[a] / (distance[a] - distance[b]);
    for (let axis = 0; axis < 3; axis += 1) out[at++] = points[3 * a + axis] + t * (points[3 * b + axis] - points[3 * a + axis]);
  };
  for (let c = 0; c < cells.length; c += cells[c] + 1) {
    if (!crossed(c)) continue;
    const [a, b, d] = [cells[c + 1], cells[c + 2], cells[c + 3]];
    // Exactly two of the three edges change side.
    if (above[a] !== above[b]) crossing(a, b);
    if (above[b] !== above[d]) crossing(b, d);
    if (above[d] !== above[a]) crossing(d, a);
  }
  return out;
}

/** Where the line from `from` along `direction` meets the plane, as a parameter t, or null. */
export function rayPlane(from: Vec3, direction: Vec3, plane: { origin: Vec3; normal: Vec3 }): number | null {
  const denominator = dot(direction, plane.normal);
  if (Math.abs(denominator) < 1e-12) return null;
  return dot(sub(plane.origin, from), plane.normal) / denominator;
}
