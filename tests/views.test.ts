import { describe, expect, it } from "vitest";

import { useSceneStore } from "@/lib/scene/store";
import { onVoxelLayer, orientationOf, planePositionLabel, scrollStep, SLICE_ROLES } from "@/lib/scene/views";
import { dot, sub } from "@/lib/world";
import { freshStorePerTest, openImage } from "./helpers";

freshStorePerTest();
const store = () => useSceneStore.getState();

/** Voxel centres at x = -47.5 ... 47.5 and z = -49.375 + 1.25 k: the volume's centre is between layers. */
const sphereLike = () => openImage([96, 96, 80], [1, 1, 1.25], [-47.5, -47.5, -49.375]);

/** Where `value` falls among the layers `origin + n * step`. */
const layerIndex = (value: number, origin: number, step: number) => (value - origin) / step;

describe("voxel layers", () => {
  it("puts a view through the volume's centre on the nearest layer", () => {
    sphereLike();
    const { red, yellow, green } = store().slicePlanes;
    expect(red.origin[2]).toBeCloseTo(0.625);
    expect(yellow.origin[1]).toBeCloseTo(0.5);
    expect(green.origin[0]).toBeCloseTo(0.5);
  });

  it("leaves an oblique plane where it is", () => {
    sphereLike();
    const normal: [number, number, number] = [Math.SQRT1_2, 0, Math.SQRT1_2];
    expect(onVoxelLayer(store(), [0.3, 0.2, 0.1], normal)).toEqual([0.3, 0.2, 0.1]);
  });

  it("scrolls a voxel per notch, staying on layers, and stops at the outermost ones", () => {
    sphereLike();
    expect(scrollStep(store(), [0, 0, -1])).toBe(1.25);
    store().scrollSlice("red", 3);
    // Scrolling down goes away from the viewer, which an axial view has at its feet.
    expect(store().slicePlanes.red.origin[2]).toBeCloseTo(0.625 + 3 * 1.25);
    store().scrollSlice("red", 1000);
    expect(store().slicePlanes.red.origin[2]).toBeCloseTo(-49.375 + 79 * 1.25);
    store().scrollSlice("red", -1000);
    expect(store().slicePlanes.red.origin[2]).toBeCloseTo(-49.375);
  });

  it("jumps every view to within half a voxel of a point, on a layer", () => {
    sphereLike();
    store().jumpSlicesTo([3.3, -7.8, 10.1]);
    const { red, yellow, green } = store().slicePlanes;
    expect(Math.abs(red.origin[2] - 10.1)).toBeLessThanOrEqual(0.625);
    const k = layerIndex(red.origin[2], -49.375, 1.25);
    expect(k).toBeCloseTo(Math.round(k), 9);
    expect(yellow.origin[1]).toBeCloseTo(-7.5);
    expect(green.origin[0]).toBeCloseTo(3.5);
  });

  it("jumps in one update, and in none when every plane already passes through the point", () => {
    sphereLike();
    let updates = 0;
    const unsubscribe = useSceneStore.subscribe(() => void (updates += 1));
    store().jumpSlicesTo([3.3, -7.8, 10.1]);
    expect(updates).toBe(1);
    const planes = store().slicePlanes;
    store().jumpSlicesTo([3.3, -7.8, 10.1]);
    expect(updates).toBe(1);
    expect(store().slicePlanes).toBe(planes);
    unsubscribe();
  });

  it("syncs the other views to a Shift-hover: planes through the point, and an event naming the hovered view", () => {
    sphereLike();
    const red = store().slicePlanes.red;
    store().syncSlicesTo([3.3, -7.8, red.origin[2]], "red");
    const { slicePlanes, sliceSync } = store();
    // The hovered view's plane holds the point already and is the same object: nothing to redraw.
    expect(slicePlanes.red).toBe(red);
    expect(slicePlanes.yellow.origin[1]).toBeCloseTo(-7.5);
    expect(slicePlanes.green.origin[0]).toBeCloseTo(3.5);
    expect(sliceSync).toEqual({ position: [3.3, -7.8, red.origin[2]], from: "red" });
    // Each hover is its own event, even at the same point.
    store().syncSlicesTo([3.3, -7.8, red.origin[2]], "red");
    expect(store().sliceSync).not.toBe(sliceSync);
    expect(store().sliceSync).toEqual(sliceSync);
    // Over: the hover goes, the views stay where it left them; ending twice changes nothing.
    let updates = 0;
    const unsubscribe = useSceneStore.subscribe(() => void (updates += 1));
    store().endSliceSync();
    store().endSliceSync();
    unsubscribe();
    expect(updates).toBe(1);
    expect(store().sliceSync).toBeNull();
    expect(store().slicePlanes).toBe(slicePlanes);
  });

  it("follows the reformat handle exactly, only keeping the plane in the data", () => {
    sphereLike();
    store().setSlicePlane("red", [0.3, 0.2, 0.1], [0, 0.6, -0.8]);
    const plane = store().slicePlanes.red;
    expect(plane.origin).toEqual([0.3, 0.2, 0.1]);
    expect(orientationOf(plane)).toBeNull();
    // Screen up turned with the plane, and stays in it.
    expect(dot(plane.viewUp, plane.normal)).toBeCloseTo(0);
    store().setSlicePlane("red", [0, 0, 500], [0, 0, -1]);
    expect(store().slicePlanes.red.origin[2]).toBeCloseTo(-49.375 + 79 * 1.25);
  });

  it("puts a view in another orientation through the point it passes now", () => {
    sphereLike();
    store().setSliceOrientation("red", "sagittal");
    const plane = store().slicePlanes.red;
    expect(orientationOf(plane)).toBe("sagittal");
    expect(plane.origin[0]).toBeCloseTo(0.5);
    // Fitted once on opening, and again for the new orientation.
    expect(store().sliceFits.red).toBe(2);
  });

  it("labels a plane by where it cuts", () => {
    sphereLike();
    expect(planePositionLabel(store().slicePlanes.red, [0, 0, 0])).toBe("S 0.6 mm");
    store().setSlicePlane("red", [0, 0, 0], [0, 0.6, -0.8]);
    // Depth is positive away from the viewer, the way scrolling down moves it.
    expect(planePositionLabel(store().slicePlanes.red, [0, 0, 1])).toBe("depth -0.8 mm");
  });
});

describe("extents", () => {
  it("are kept per view, only when they change, forgotten when a view goes or a new image opens", () => {
    sphereLike();
    const extent = { center: [1, 2, 3] as [number, number, number], halfHeight: 10, aspect: 1.5 };
    store().setSliceExtent("red", extent);
    const before = store();
    store().setSliceExtent("red", { ...extent, halfHeight: 10 + 1e-5 });
    expect(store()).toBe(before);
    store().setSliceExtent("yellow", extent);
    store().setSliceExtent("red", null);
    expect(Object.keys(store().sliceExtents)).toEqual(["yellow"]);
    store().resetViews();
    expect(store().sliceExtents).toEqual({});
  });
});

describe("every slice view", () => {
  it("starts in its own standard orientation, through the centre", () => {
    sphereLike();
    const orientations = SLICE_ROLES.map((role) => orientationOf(store().slicePlanes[role]));
    expect(orientations).toEqual(["axial", "coronal", "sagittal"]);
    for (const role of SLICE_ROLES) {
      const plane = store().slicePlanes[role];
      expect(Math.abs(dot(sub(plane.origin, [0, 0, 0]), plane.normal))).toBeLessThanOrEqual(0.625);
    }
  });
});
