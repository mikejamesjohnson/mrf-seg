import { describe, expect, it } from "vitest";

import { planeInOrientation } from "@/lib/scene/views";
import { insideConvexPolygon, meshPlaneSegments, rayPlane, visiblePolygon } from "@/lib/viewer/planeGeometry";
import type { Bounds, Vec3 } from "@/lib/world";

const BOUNDS: Bounds = [-50, 50, -50, 50, -50, 50];

describe("visiblePolygon", () => {
  it("is the view's rectangle while it fits in the volume", () => {
    const plane = planeInOrientation("axial", [0, 0, 5]);
    const polygon = visiblePolygon(plane, { center: [1, 2, 99], halfHeight: 10, aspect: 2 }, BOUNDS);
    expect(polygon).toHaveLength(4);
    // On the plane, wherever the camera's focal point was along the normal.
    for (const p of polygon) expect(p[2]).toBeCloseTo(5);
    const xs = polygon.map((p) => p[0]);
    const ys = polygon.map((p) => p[1]);
    expect(Math.min(...xs)).toBeCloseTo(-19);
    expect(Math.max(...xs)).toBeCloseTo(21);
    expect(Math.min(...ys)).toBeCloseTo(-8);
    expect(Math.max(...ys)).toBeCloseTo(12);
  });

  it("is clipped to the volume where the view reaches past it", () => {
    const plane = planeInOrientation("coronal", [0, 0, 0]);
    const polygon = visiblePolygon(plane, { center: [40, 0, 0], halfHeight: 100, aspect: 1 }, BOUNDS);
    const xs = polygon.map((p) => p[0]);
    const zs = polygon.map((p) => p[2]);
    expect(Math.min(...xs)).toBeCloseTo(-50);
    expect(Math.max(...xs)).toBeCloseTo(50);
    expect(Math.max(...zs)).toBeCloseTo(50);
  });

  it("is nothing when the plane misses the volume", () => {
    const plane = planeInOrientation("axial", [0, 0, 0]);
    expect(visiblePolygon(plane, { center: [500, 0, 0], halfHeight: 10, aspect: 1 }, BOUNDS)).toHaveLength(0);
  });
});

describe("insideConvexPolygon", () => {
  const square: Vec3[] = [
    [0, 0, 0],
    [1, 0, 0],
    [1, 1, 0],
    [0, 1, 0],
  ];
  it("tells inside from outside, either winding", () => {
    expect(insideConvexPolygon([0.5, 0.5, 0], square, [0, 0, 1])).toBe(true);
    expect(insideConvexPolygon([0.5, 0.5, 0], [...square].reverse(), [0, 0, 1])).toBe(true);
    expect(insideConvexPolygon([1.5, 0.5, 0], square, [0, 0, 1])).toBe(false);
  });
});

/** The unit cube as triangles, in vtk.js's cell layout. */
function cube(): { points: Float32Array; cells: Uint32Array } {
  const points = new Float32Array([0, 0, 0, 1, 0, 0, 1, 1, 0, 0, 1, 0, 0, 0, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1]);
  const faces = [
    [0, 1, 2, 3],
    [4, 5, 6, 7],
    [0, 1, 5, 4],
    [1, 2, 6, 5],
    [2, 3, 7, 6],
    [3, 0, 4, 7],
  ];
  const cells = faces.flatMap(([a, b, c, d]) => [3, a, b, c, 3, a, c, d]);
  return { points, cells: new Uint32Array(cells) };
}

describe("meshPlaneSegments", () => {
  it("cuts the cube's four sides at the plane, as one segment per crossed triangle", () => {
    const { points, cells } = cube();
    const segments = meshPlaneSegments(points, cells, { origin: [0, 0, 0.25], normal: [0, 0, 1] });
    expect(segments.length).toBe(8 * 6);
    let length = 0;
    for (let s = 0; s < segments.length; s += 6) {
      expect(segments[s + 2]).toBeCloseTo(0.25);
      expect(segments[s + 5]).toBeCloseTo(0.25);
      length += Math.hypot(segments[s + 3] - segments[s], segments[s + 4] - segments[s + 1]);
    }
    // The cut is the square's perimeter.
    expect(length).toBeCloseTo(4);
  });

  it("finds nothing when the plane misses the mesh, and skips cells that are not triangles", () => {
    const { points, cells } = cube();
    expect(meshPlaneSegments(points, cells, { origin: [0, 0, 2], normal: [0, 0, 1] })).toHaveLength(0);
    const lines = new Uint32Array([2, 0, 4]);
    expect(meshPlaneSegments(points, lines, { origin: [0, 0, 0.5], normal: [0, 0, 1] })).toHaveLength(0);
  });
});

describe("rayPlane", () => {
  it("meets the plane where it crosses it, and never when parallel", () => {
    expect(rayPlane([0, 0, 10], [0, 0, -1], { origin: [0, 0, 2], normal: [0, 0, 1] })).toBeCloseTo(8);
    expect(rayPlane([0, 0, 10], [1, 0, 0], { origin: [0, 0, 2], normal: [0, 0, 1] })).toBeNull();
  });
});
