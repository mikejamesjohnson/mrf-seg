/** What the panels and auto update ask of the segmentation state, as pure functions of it. */

import { describe, expect, it } from "vitest";

import { useSceneStore } from "@/lib/scene/store";
import { canBuild, enoughPoints, isBusy, needsGenerate, productStale, segmentationIsStale, staleReasons, surfaceLevel, type GeneratedSegmentation } from "@/lib/segmentation/state";
import { freshStorePerTest, openImage } from "./helpers";

freshStorePerTest();
const store = () => useSceneStore.getState();

const inference = { iterations: 12, residual: 0.00005, converged: true, components: 1, seeds: { inside: 1, outside: 1, background: 0 } };

/** The store with an image, an inside and an outside point, a session, and a segmentation generated from all of that. */
function generated(): GeneratedSegmentation {
  openImage([10, 10, 10], [1, 1, 1], [0, 0, 0]);
  store().setPlaceMode("inside");
  store().addPoint([1, 1, 1]);
  store().setPlaceMode("outside");
  store().addPoint([8, 8, 8]);
  store().setSegmentation({ sessionId: "s1" });
  const made: GeneratedSegmentation = {
    points: 2,
    pointsVersion: store().pointsVersion,
    settings: store().segmentation.settings,
    sessionId: "s1",
    seconds: { features: 0, affinities: 0, seeds: 0, solve: 0, maps: 0 },
    inference,
  };
  store().setSegmentation({ generated: made, generation: 1 });
  return made;
}

describe("the segmentation state", () => {
  it("knows when there is anything to segment from: an inside point, whichever the extractor", () => {
    openImage([10, 10, 10], [1, 1, 1], [0, 0, 0]);
    expect(enoughPoints(store())).toBe(false);
    store().setPlaceMode("outside");
    store().addPoint([2, 2, 2]);
    expect(enoughPoints(store())).toBe(false);
    store().setPlaceMode("inside");
    store().addPoint([1, 1, 1]);
    expect(enoughPoints(store())).toBe(true);
    store().removePoints([2]);
    expect(enoughPoints(store())).toBe(false);
    store().setSegmentationSettings({ featureExtractor: "learned" });
    expect(enoughPoints(store())).toBe(false); // the walker needs a seed to carry, whatever the affinities
  });

  it("is busy while a Generate or any build runs", () => {
    expect(isBusy(store())).toBe(false);
    store().setSegmentation({ job: { state: "running", progress: 0 } });
    expect(isBusy(store())).toBe(true);
    store().setSegmentation({ job: { state: "done", progress: 1 } });
    store().setProduct("surface", { status: { state: "running", message: "Building…" } });
    expect(isBusy(store())).toBe(true);
    store().setProduct("surface", { status: { state: "idle" } });
    expect(isBusy(store())).toBe(false);
  });

  it("says why the maps are out of date: points, settings, or a new session", () => {
    expect(staleReasons(store())).toEqual([]); // nothing generated: nothing to be out of date
    generated();
    expect(staleReasons(store())).toEqual([]);
    expect(needsGenerate(store())).toBe(false);

    store().setPlaceMode("outside");
    store().addPoint([5, 5, 5]);
    expect(staleReasons(store())).toEqual(["the points have changed"]);
    store().setSegmentationSettings({ backgroundSeeds: false });
    store().setSegmentation({ sessionId: "s2" });
    expect(staleReasons(store())).toEqual(["the points have changed", "the settings have changed", "the backend session is new, and has no segmentation yet"]);
    expect(segmentationIsStale(store())).toBe(true);
    expect(needsGenerate(store())).toBe(true);
  });

  it("counts any setting as a change to the segmentation", () => {
    generated();
    store().setSegmentationSettings({ seedHu: 12 });
    expect(staleReasons(store())).toEqual(["the settings have changed"]);
  });

  it("can build a product only from a segmentation the current session holds", () => {
    expect(canBuild(store())).toBe(false);
    generated();
    expect(canBuild(store())).toBe(true);
    store().setSegmentation({ sessionId: "s2" });
    expect(canBuild(store())).toBe(false);
  });

  it("lets a new image choose the grid, but keeps a value the extractor brought", () => {
    openImage([10, 10, 10], [1, 1, 1], [0, 0, 0]);
    store().setExtractors({ learned: { available: true, settings: { affinityScale: 1.5, featureSpacing: 0.5 }, nativeSpacing: 0.5, note: null } });
    store().setSegmentationSettings({ featureExtractor: "basic", featureSpacing: 2, affinityScale: 2 });
    store().resetSegmentation("another");
    expect([store().segmentation.settings.featureSpacing, store().segmentation.settings.affinityScale]).toEqual([null, 2]); // Basic: the grid is the image's to choose
    store().setSegmentationSettings({ featureExtractor: "learned", featureSpacing: 0.5, affinityScale: 1.5 });
    store().resetSegmentation("yet-another");
    expect([store().segmentation.settings.featureSpacing, store().segmentation.settings.affinityScale]).toEqual([0.5, 1.5]); // Learned: the checkpoint's grid stays
  });

  it("does not count a Generate made before the session chose the image's settings as current", () => {
    const made = generated();
    // As `generate.ts` must not do: settings copied with the image-derived values still null.
    store().setSegmentation({ generated: { ...made, settings: { ...made.settings, featureSpacing: null } } });
    store().setSegmentationSettings({ featureSpacing: 1 });
    expect(staleReasons(store())).toEqual(["the settings have changed"]);
  });

  it("needs a Generate only with an inside point", () => {
    openImage([10, 10, 10], [1, 1, 1], [0, 0, 0]);
    expect(needsGenerate(store())).toBe(false);
    store().setPlaceMode("outside");
    store().addPoint([2, 2, 2]);
    expect(needsGenerate(store())).toBe(false);
    store().setPlaceMode("inside");
    store().addPoint([1, 1, 1]);
    expect(needsGenerate(store())).toBe(true); // nothing generated yet
  });

  it("builds a surface at the probability threshold", () => {
    expect(surfaceLevel({ source: "probability", threshold: 0.7 })).toBe(0.7);
  });

  it("knows a product is out of date by generation and by its own settings", () => {
    generated();
    expect(productStale(store(), "labelmap")).toBe(true); // never built
    store().setProduct("labelmap", { built: { threshold: 0.5, voxels: 1, seconds: 0, generation: 1 } });
    store().setProduct("surface", { built: { source: "probability", level: 0.5, triangles: 1, seconds: 0, generation: 1 } });
    expect([productStale(store(), "labelmap"), productStale(store(), "surface")]).toEqual([false, false]);
    store().setProduct("labelmap", { threshold: 0.6 });
    expect(productStale(store(), "labelmap")).toBe(true);
    store().setProduct("labelmap", { threshold: 0.5 });
    store().setProduct("surface", { threshold: 0.6 });
    expect(productStale(store(), "surface")).toBe(true);
    store().setProduct("surface", { threshold: 0.5 });
    store().setSegmentation({ generation: 2 });
    expect([productStale(store(), "labelmap"), productStale(store(), "surface")]).toEqual([true, true]);
  });
});
