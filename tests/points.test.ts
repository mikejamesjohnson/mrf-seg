import { describe, expect, it } from "vitest";

import { pointCounts } from "@/lib/scene/points";
import { useSceneStore } from "@/lib/scene/store";
import { freshStorePerTest, openImage } from "./helpers";

freshStorePerTest();
const store = () => useSceneStore.getState();

/** Voxel centres from (0, 0, 0) to (9, 9, 9). */
const cube = () => openImage([10, 10, 10], [1, 1, 1], [0, 0, 0]);

describe("points", () => {
  it("are added only in a place mode, labelled by it, and kept in the image", () => {
    cube();
    expect(store().addPoint([1, 1, 1])).toBeNull();
    store().setPlaceMode("inside");
    expect(store().addPoint([1, 2, 3])).toBe(1);
    store().setPlaceMode("outside");
    expect(store().addPoint([20, -5, 4])).toBe(2);
    const [a, b] = store().points;
    expect(a).toEqual({ id: 1, label: 1, position: [1, 2, 3] });
    expect(b).toEqual({ id: 2, label: -1, position: [9, 0, 4] });
    expect(pointCounts(store().points)).toEqual({ inside: 1, outside: 1 });
  });

  it("keep their ids when others go, and a cleared list starts again from 1", () => {
    cube();
    store().setPlaceMode("inside");
    for (const x of [1, 2, 3]) store().addPoint([x, 0, 0]);
    store().selectPoints([1, 2], "replace");
    store().removePoints([2]);
    expect(store().points.map((p) => p.id)).toEqual([1, 3]);
    expect(store().selectedPointIds).toEqual([1]);
    expect(store().addPoint([4, 0, 0])).toBe(4);
    store().clearPoints();
    expect(store().points).toEqual([]);
    expect(store().selectedPointIds).toEqual([]);
    expect(store().addPoint([4, 0, 0])).toBe(1);
  });

  it("move as asked, kept in the image, and only a real move counts as a change", () => {
    cube();
    store().setPlaceMode("inside");
    store().addPoint([1, 1, 1]);
    store().addPoint([2, 2, 2]);
    const version = store().pointsVersion;
    store().setPointPositions(new Map([[99, [5, 5, 5]]]));
    expect(store().pointsVersion).toBe(version);
    store().setPointPositions(new Map([[2, [3, 4, 50]]]));
    expect(store().points.map((p) => p.position)).toEqual([
      [1, 1, 1],
      [3, 4, 9],
    ]);
    expect(store().pointsVersion).toBe(version + 1);
  });

  it("are selected by replacing, adding or toggling", () => {
    cube();
    store().selectPoints([1, 2, 2], "replace");
    expect(store().selectedPointIds).toEqual([1, 2]);
    store().selectPoints([3], "add");
    store().selectPoints([1], "toggle");
    expect(store().selectedPointIds).toEqual([2, 3]);
  });

  it("are shown in 3D and on the slices until hidden there", () => {
    expect(store().pointsVisible).toEqual({ threeD: true, slices: true });
    store().setPointsVisible("slices", false);
    expect(store().pointsVisible).toEqual({ threeD: true, slices: false });
    store().setPointsVisible("threeD", false);
    store().setPointsVisible("slices", true);
    expect(store().pointsVisible).toEqual({ threeD: false, slices: true });
  });

  it("need an image", () => {
    store().setPlaceMode("inside");
    expect(store().addPoint([0, 0, 0])).toBeNull();
  });
});
