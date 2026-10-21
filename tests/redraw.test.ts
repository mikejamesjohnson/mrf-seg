/** What makes a pane redraw, by role, and what each pane draws; and how layers are placed. */

import { describe, expect, it } from "vitest";

import { useSceneStore } from "@/lib/scene/store";
import { layerOffset } from "@/lib/viewer/layers";
import { drawsDifferently, showsNodeType } from "@/lib/viewer/redraw";
import { freshStorePerTest, openImage } from "./helpers";

freshStorePerTest();
const store = () => useSceneStore.getState();

describe("drawsDifferently", () => {
  it("ignores a store change no pane draws from", () => {
    openImage([10, 10, 10], [1, 1, 1], [0, 0, 0]);
    const before = store();
    store().setSegmentation({ job: { state: "running", stage: "inference", progress: 0.5 } });
    for (const role of ["red", "yellow", "green", "3d"] as const) expect(drawsDifferently(role, store(), before), role).toBe(false);
  });

  it("redraws a slice pane for its own plane only, and the 3D pane for every plane", () => {
    openImage([10, 10, 10], [1, 1, 1], [0, 0, 0]);
    const before = store();
    const red = before.slicePlanes.red;
    store().setSlicePlane("red", [red.origin[0], red.origin[1], red.origin[2] + 1], red.normal);
    expect(drawsDifferently("red", store(), before)).toBe(true);
    expect(drawsDifferently("yellow", store(), before)).toBe(false);
    expect(drawsDifferently("green", store(), before)).toBe(false);
    expect(drawsDifferently("3d", store(), before)).toBe(true);
  });

  it("redraws only the 3D pane for what is shown of a slice view there", () => {
    openImage([10, 10, 10], [1, 1, 1], [0, 0, 0]);
    const before = store();
    store().setSliceExtent("red", { center: [0, 0, 0], halfHeight: 5, aspect: 1 });
    expect(drawsDifferently("red", store(), before)).toBe(false);
    expect(drawsDifferently("3d", store(), before)).toBe(true);
  });

  it("redraws the other slice panes for a Shift-hover, not the hovered one, and the 3D pane only if a plane moved", () => {
    openImage([10, 10, 10], [1, 1, 1], [0, 0, 0]);
    const red = store().slicePlanes.red;
    // A point every plane passes through already: the planes stay, only the event is new.
    const center = [store().slicePlanes.green.origin[0], store().slicePlanes.yellow.origin[1], red.origin[2]] as const;
    let before = store();
    store().syncSlicesTo([...center], "red");
    expect(store().slicePlanes).toBe(before.slicePlanes);
    expect(drawsDifferently("red", store(), before)).toBe(false);
    expect(drawsDifferently("yellow", store(), before)).toBe(true);
    expect(drawsDifferently("green", store(), before)).toBe(true);
    expect(drawsDifferently("3d", store(), before)).toBe(false);
    // Off the other planes: they move, and the 3D pane draws them where they are now.
    before = store();
    store().syncSlicesTo([center[0] + 2, center[1] + 2, red.origin[2]], "red");
    expect(drawsDifferently("red", store(), before)).toBe(false);
    expect(drawsDifferently("3d", store(), before)).toBe(true);
  });

  it("redraws every pane for the points and the datasets", () => {
    openImage([10, 10, 10], [1, 1, 1], [0, 0, 0]);
    const before = store();
    store().setPlaceMode("inside");
    store().addPoint([1, 1, 1]);
    for (const role of ["red", "yellow", "green", "3d"] as const) expect(drawsDifferently(role, store(), before), role).toBe(true);
  });
});

describe("showsNodeType", () => {
  it("draws the bounding box in 3D only, and everything else everywhere", () => {
    expect(showsNodeType("red", "outline")).toBe(false);
    expect(showsNodeType("3d", "outline")).toBe(true);
    for (const type of ["imageSlice", "overlay", "labelmap", "surface"] as const) {
      expect(showsNodeType("red", type), type).toBe(true);
      expect(showsNodeType("3d", type), type).toBe(true);
    }
  });
});

describe("layers", () => {
  it("move the actor toward the camera along the plane's normal, the image not at all", () => {
    expect(layerOffset("image", [0, 0, 1], 1).map(Math.abs)).toEqual([0, 0, 0]);
    expect(layerOffset("map", [0, 0, 1], 1)[2]).toBeCloseTo(0.08);
    expect(layerOffset("map", [0, 0, 1], -1)[2]).toBeCloseTo(-0.08);
    expect(layerOffset("labelmap", [0, 1, 0], 1)[1]).toBeGreaterThan(layerOffset("map", [0, 1, 0], 1)[1]);
  });
});
