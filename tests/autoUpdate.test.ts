import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import { useSceneStore } from "@/lib/scene/store";
import { autoUpdateSettled, installAutoUpdate } from "@/lib/segmentation/autoUpdate";
import { freshStorePerTest, openImage } from "./helpers";

/**
 * What ran, in order; Generates to hold open until released; whether Generate fails; and
 * whether it marks the job done a tick before it stores the result, as the real one would if it
 * reported the backend's state as its own.
 */
const steps = vi.hoisted(() => ({ ran: [] as string[], hold: 0, release: [] as (() => void)[], fail: false, doneEarly: false }));

const SESSION = "test-session";

// Generate and the builds, as far as the store sees them; whether things are stale stays real.
vi.mock("@/lib/segmentation/generate", async () => ({
  generateSegmentation: async () => {
    const { useSceneStore: scene } = await import("@/lib/scene/store");
    const { points, pointsVersion, segmentation } = scene.getState();
    steps.ran.push(`generate ${points.length}`);
    scene.getState().setSegmentation({ job: { state: "running", progress: 0 } });
    if (steps.hold > 0) {
      steps.hold -= 1;
      await new Promise<void>((resolve) => steps.release.push(resolve));
    }
    if (steps.fail) return void scene.getState().setSegmentation({ job: { state: "error", progress: 0 } });
    if (steps.doneEarly) {
      scene.getState().setSegmentation({ job: { state: "done", progress: 1 } });
      await new Promise((resolve) => setTimeout(resolve, 5));
    }
    const inference = { sweeps: 10, delta: 0.0001, converged: true, components: null };
    scene.getState().setSegmentation({
      job: { state: "done", progress: 1 },
      generation: scene.getState().segmentation.generation + 1,
      generated: {
        points: points.length,
        pointsVersion,
        settings: segmentation.settings,
        sessionId: segmentation.sessionId ?? SESSION,
        seconds: { features: 0, potentials: 0, inference: 0, maps: 0 },
        inference,
      },
    });
  },
}));

vi.mock("@/lib/segmentation/products", async (importOriginal) => {
  const { useSceneStore: scene } = await import("@/lib/scene/store");
  const { surfaceLevel, productStale } = await import("@/lib/segmentation/state");
  const buildSegmentationLabelmap = async () => {
    const s = scene.getState();
    steps.ran.push("labelmap");
    s.setProduct("labelmap", { built: { threshold: s.labelmap.threshold, voxels: 0, seconds: 0, generation: s.segmentation.generation } });
  };
  const buildSegmentationSurface = async () => {
    const s = scene.getState();
    steps.ran.push("surface");
    s.setProduct("surface", { built: { source: s.surface.source, level: surfaceLevel(s.surface), triangles: 0, seconds: 0, generation: s.segmentation.generation } });
  };
  const original = await importOriginal<typeof import("@/lib/segmentation/products")>();
  const PRODUCTS = original.PRODUCTS.map((product) => ({
    ...product,
    build: product.key === "labelmap" ? buildSegmentationLabelmap : buildSegmentationSurface,
    stale: (state: Parameters<typeof productStale>[0]) => productStale(state, product.key),
  }));
  return { ...original, PRODUCTS, buildSegmentationLabelmap, buildSegmentationSurface };
});

freshStorePerTest();
const store = () => useSceneStore.getState();
let stop: () => void;

beforeEach(() => {
  Object.assign(steps, { ran: [], hold: 0, release: [], fail: false, doneEarly: false });
  openImage([10, 10, 10], [1, 1, 1], [0, 0, 0]);
  store().setSegmentation({ sessionId: SESSION });
  stop = installAutoUpdate({ quietMs: 0 });
});
afterEach(() => stop());

function add(label: "inside" | "outside", x = 1) {
  store().setPlaceMode(label);
  store().addPoint([x, 2, 3]);
}

const tick = (ms = 0) => new Promise((resolve) => setTimeout(resolve, ms));

/** Let the held Generates go, one at a time, until none is left or waiting. */
async function finish() {
  for (;;) {
    await tick();
    const next = steps.release.shift();
    if (!next) break;
    next();
  }
  await autoUpdateSettled();
}

/**
 * The segmentation and both products ticked, with an inside and an outside point, and brought
 * up to date; what ran is then forgotten.
 */
async function upToDate() {
  add("inside");
  add("outside");
  store().setAutoUpdate({ segmentation: true, labelmap: true, surface: true });
  await finish();
  steps.ran = [];
}

describe("auto update", () => {
  it("is off until a step is ticked, and then runs the ticked steps", async () => {
    expect(store().autoUpdate).toEqual({ segmentation: false, labelmap: false, surface: false });
    add("inside");
    add("outside");
    await finish();
    expect(steps.ran).toEqual([]);
    store().setAutoUpdate({ segmentation: true, surface: true });
    await finish();
    expect(steps.ran).toEqual(["generate 2", "surface"]);
  });

  it("brings everything up to date when ticked, and again when a point is added", async () => {
    add("outside");
    store().setAutoUpdate({ segmentation: true, labelmap: true, surface: true });
    await finish();
    expect(steps.ran).toEqual([]); // no inside point: nothing to segment from
    add("inside");
    await finish();
    expect(steps.ran).toEqual(["generate 2", "labelmap", "surface"]);
    add("outside", 4);
    await finish();
    expect(steps.ran).toEqual(["generate 2", "labelmap", "surface", "generate 3", "labelmap", "surface"]);
  });

  it("runs when a point is moved or removed", async () => {
    await upToDate();
    store().setPointPositions(new Map([[1, [4, 4, 4]]]));
    await finish();
    store().removePoints([2]);
    add("outside", 5);
    await finish();
    expect(steps.ran).toEqual(["generate 2", "labelmap", "surface", "generate 2", "labelmap", "surface"]);
  });

  it("runs when a setting changes", async () => {
    await upToDate();
    store().setSegmentationSettings({ sweeps: 300 });
    await finish();
    expect(steps.ran).toEqual(["generate 2", "labelmap", "surface"]);
    store().setSegmentationSettings({ coupling: 2 });
    await finish();
    expect(steps.ran).toEqual(["generate 2", "labelmap", "surface", "generate 2", "labelmap", "surface"]);
  });

  it("rebuilds only the product whose own settings changed", async () => {
    await upToDate();
    store().setProduct("labelmap", { threshold: 0.7 });
    await finish();
    store().setProduct("surface", { source: "unary" });
    await finish();
    store().setProduct("surface", { source: "probability", threshold: 0.6 });
    await finish();
    expect(steps.ran).toEqual(["labelmap", "surface", "surface"]);
  });

  it("builds only what is ticked, and a product once it is ticked", async () => {
    await upToDate();
    store().setAutoUpdate({ surface: false });
    store().setSegmentationSettings({ sweeps: 200 });
    await finish();
    expect(steps.ran).toEqual(["generate 2", "labelmap"]);
    store().setAutoUpdate({ surface: true }); // out of date: built from the segmentation before
    await finish();
    expect(steps.ran).toEqual(["generate 2", "labelmap", "surface"]);
  });

  it("does not build from a stale segmentation when the segmentation is not ticked", async () => {
    await upToDate();
    store().setAutoUpdate({ segmentation: false });
    add("outside", 7);
    await finish();
    expect(steps.ran).toEqual([]); // the products wait for a Generate from the button
  });

  it("takes only the latest: changes during a run make one more run, without builds in between", async () => {
    await upToDate();
    steps.hold = 1;
    add("inside", 2); // a Generate with 3 points starts, and is held
    await tick();
    add("outside", 3);
    add("outside", 4);
    store().setSegmentationSettings({ sweeps: 250 });
    await finish();
    expect(steps.ran).toEqual(["generate 3", "generate 5", "labelmap", "surface"]);
  });

  it("does not start a second Generate while the first is still storing its maps", async () => {
    await upToDate();
    steps.doneEarly = true;
    steps.hold = 1;
    add("inside", 2);
    await tick();
    add("outside", 3); // asked for while the first runs: one more run, after it has stored its maps
    await finish();
    expect(steps.ran).toEqual(["generate 3", "generate 4", "labelmap", "surface"]);
  });

  it("waits for a burst of changes to end", async () => {
    await upToDate();
    stop();
    stop = installAutoUpdate({ quietMs: 60 });
    for (const x of [2, 3, 4]) {
      add("outside", x);
      await tick(20);
    }
    await finish();
    expect(steps.ran).toEqual(["generate 5", "labelmap", "surface"]);
  });

  it("counts a new backend session as out of date", async () => {
    await upToDate();
    store().setSegmentation({ sessionId: "another-session" });
    await finish();
    expect(steps.ran).toEqual(["generate 2", "labelmap", "surface"]);
  });

  it("does not repeat a Generate that failed until something changes, and stops when unticked", async () => {
    steps.fail = true;
    await upToDate();
    await finish();
    store().setSegmentation({ job: { state: "idle", progress: 0 } }); // a store change that is not a new input
    await finish();
    expect(steps.ran).toEqual([]);
    steps.fail = false;
    store().setSegmentationSettings({ sweeps: 100 });
    await finish();
    expect(steps.ran).toEqual(["generate 2", "labelmap", "surface"]);

    steps.ran = [];
    steps.hold = 1;
    add("outside", 6);
    await tick();
    store().setAutoUpdate({ segmentation: false, labelmap: false, surface: false }); // while that Generate runs
    await finish();
    expect(steps.ran).toEqual(["generate 3"]);
  });
});
