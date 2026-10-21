import { describe, expect, it } from "vitest";

import {
  clampToGrid,
  insideGrid,
  normalize,
  rotateWith,
  transposeMat3,
  unionBounds,
  voxelAxis,
  voxelCenterCorners,
  type Bounds,
  type GridGeometry,
  type Mat3,
} from "@/lib/world";

/** 30 degrees about z: the i axis tilted toward +y. */
const c = Math.cos(Math.PI / 6);
const s = Math.sin(Math.PI / 6);
const TILTED: Mat3 = [c, s, 0, -s, c, 0, 0, 0, 1];
const grid: GridGeometry = { gridOrigin: [10, 20, 30], spacing: [1, 2, 3], dimensions: [4, 5, 6], direction: TILTED };

describe("matrices and bounds", () => {
  it("transposes, twice back to where it started", () => {
    const m: Mat3 = [1, 2, 3, 4, 5, 6, 7, 8, 9];
    expect(transposeMat3(m)).toEqual([1, 4, 7, 2, 5, 8, 3, 6, 9]);
    expect(transposeMat3(transposeMat3(m))).toEqual(m);
  });

  it("unions bounds, ignoring missing and invalid ones", () => {
    const a: Bounds = [0, 1, 0, 1, 0, 1];
    const b: Bounds = [-1, 0.5, 2, 3, 0, 4];
    expect(unionBounds(a, b)).toEqual([-1, 1, 0, 3, 0, 4]);
    expect(unionBounds(null, a)).toEqual(a);
    expect(unionBounds(a, [0, -1, 0, 1, 0, 1])).toEqual(a);
    expect(unionBounds(null, null)).toBeNull();
  });

  it("normalises, and has no direction for a zero vector", () => {
    const unit = normalize([3, 0, 4])!;
    [0.6, 0, 0.8].forEach((v, axis) => expect(unit[axis]).toBeCloseTo(v));
    expect(normalize([0, 0, 0])).toBeNull();
  });
});

describe("rotateWith", () => {
  it("carries a vector along with the rotation from one unit vector to another", () => {
    // A quarter turn about +x takes +z to -y, and so +y to +z.
    const turned = rotateWith([0, 1, 0], [0, 0, 1], [0, -1, 0]);
    expect(turned[0]).toBeCloseTo(0);
    expect(turned[1]).toBeCloseTo(0);
    expect(turned[2]).toBeCloseTo(1);
  });

  it("leaves it alone when there is nothing to turn", () => {
    expect(rotateWith([1, 2, 3], [0, 0, 1], [0, 0, 1])).toEqual([1, 2, 3]);
  });
});

describe("voxel grids", () => {
  it("reads each voxel axis from its row of the direction", () => {
    expect(voxelAxis(grid, 0)).toEqual([c, s, 0]);
    expect(voxelAxis(grid, 1)).toEqual([-s, c, 0]);
  });

  it("puts the outermost voxel centres along the tilted axes", () => {
    const corners = voxelCenterCorners(grid);
    expect(corners).toHaveLength(8);
    expect(corners[0]).toEqual([10, 20, 30]);
    // Corner 1 is voxel (3, 0, 0): three voxels along i.
    expect(corners[1][0]).toBeCloseTo(10 + 3 * c);
    expect(corners[1][1]).toBeCloseTo(20 + 3 * s);
    // Corner 7 is the far one: (3, 4, 5).
    expect(corners[7][2]).toBeCloseTo(30 + 5 * 3);
  });

  it("clamps into the grid in its own frame", () => {
    const inside: [number, number, number] = [10 + c, 20 + s, 33];
    expect(insideGrid(grid, inside)).toBe(true);
    const below = clampToGrid(grid, [10, 20, 0]);
    expect(below[2]).toBeCloseTo(30);
    // One voxel before voxel 0 along i comes back to voxel 0.
    const before = clampToGrid(grid, [10 - c, 20 - s, 30]);
    expect(before[0]).toBeCloseTo(10);
    expect(before[1]).toBeCloseTo(20);
    expect(insideGrid(grid, [10 - c, 20 - s, 30])).toBe(false);
  });
});
