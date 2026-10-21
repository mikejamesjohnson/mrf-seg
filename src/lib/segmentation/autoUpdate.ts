/**
 * Auto update: keeps each step whose Auto box is ticked up to date with what it is made from.
 * Any change that a fresh run of a step would show (a point added, moved or removed, a setting,
 * a product's threshold, or ticking a box) runs the ticked steps it makes out of date, in the
 * pipeline's order, and nothing else: Generate if the segmentation is stale (`needsGenerate`),
 * then each product that is (`PRODUCTS[].stale`). A step not ticked is left to its button, and
 * the steps after it wait for it: a stale segmentation is not built from.
 *
 * Only the latest state matters. Changes are not queued a run each: a run starts once nothing
 * is running (here or from the panel), no point is being dragged and nothing has changed for
 * `quietMs`, and takes everything changed by then. A change during a run asks for one more,
 * and that run's builds are skipped, as the next run replaces them. Number fields commit when
 * they are left, so typing in one changes nothing until then. Outputs changing never ask for a
 * run, so a run that fails is not repeated until an input changes again.
 */

import { useSceneStore, type SceneState } from "@/lib/scene/store";
import { sleep, store } from "./flow";
import { generateSegmentation } from "./generate";
import { PRODUCTS } from "./products";
import { isBusy, needsGenerate, segmentationIsStale, type AutoUpdateSettings } from "./state";

/** ms with no change before a run starts, so a burst of changes makes one run. */
const QUIET_MS = 250;

let quietMs = QUIET_MS;
let pending = false;
let draining: Promise<void> | null = null;
let lastChange = 0;
let pointersDown = 0;
const pointersUp: (() => void)[] = [];

/** What a run is made from: when any of it changes, a run is asked for. The session counts: a new one has no segmentation. */
function inputs(state: SceneState): unknown[] {
  return [state.pointsVersion, state.segmentation.settings, state.segmentation.sessionId, state.autoUpdate, ...PRODUCTS.flatMap((product) => product.inputs(state))];
}

/** Whether any step runs by itself. */
export function anyAuto(auto: AutoUpdateSettings): boolean {
  return Object.values(auto).some(Boolean);
}

/** Resolves once the store satisfies `ready`. */
function until(ready: (state: SceneState) => boolean): Promise<void> {
  if (ready(store())) return Promise.resolve();
  return new Promise((resolve) => {
    const unsubscribe = useSceneStore.subscribe((state) => {
      if (!ready(state)) return;
      unsubscribe();
      resolve();
    });
  });
}

/** Until nothing is running, no point is being dragged, and nothing has changed for `quietMs`, all at once. */
async function settle(): Promise<void> {
  for (;;) {
    await until((state) => !isBusy(state));
    if (pointersDown > 0) {
      await new Promise<void>((resolve) => pointersUp.push(resolve));
      continue;
    }
    const wait = lastChange + quietMs - Date.now();
    if (wait <= 0 && !isBusy(store())) return;
    await sleep(Math.max(wait, 0));
  }
}

/** Run the pipeline now, or once more after the run in progress. Resolves when it is idle again. */
export function requestAutoUpdate(): Promise<void> {
  pending = true;
  draining ??= drain().finally(() => {
    draining = null;
    if (pending) void requestAutoUpdate(); // asked for between the last check and here
  });
  return draining;
}

async function drain(): Promise<void> {
  while (pending) {
    await settle();
    pending = false; // this run takes everything changed until now
    if (!anyAuto(store().autoUpdate)) return;
    await runOnce();
  }
}

async function runOnce(): Promise<void> {
  if (store().autoUpdate.segmentation && needsGenerate(store())) {
    const before = store().segmentation.generation;
    await generateSegmentation();
    if (store().segmentation.generation === before) return; // it failed, or a newer one replaced it
  }
  const state = store();
  if (state.segmentation.generated === null || segmentationIsStale(state)) return; // nothing current to build from
  for (const product of PRODUCTS) {
    const now = store();
    if (pending || !anyAuto(now.autoUpdate)) return;
    if (!now.autoUpdate[product.key] || !product.stale(now)) continue;
    await until((s) => !isBusy(s));
    await product.build();
  }
}

/** Resolves once no run is going or asked for: for tests. */
export async function autoUpdateSettled(): Promise<void> {
  while (draining) await draining;
}

/**
 * Bring things up to date whenever what they are made from changes, while any step is ticked
 * (and when one is ticked). Returns the uninstall. `quietMs` is for tests.
 */
export function installAutoUpdate(options: { quietMs?: number } = {}): () => void {
  quietMs = options.quietMs ?? QUIET_MS;
  let last = inputs(store());
  const unsubscribe = useSceneStore.subscribe((state) => {
    const now = inputs(state);
    if (now.every((value, i) => value === last[i])) return;
    last = now;
    lastChange = Date.now();
    if (anyAuto(state.autoUpdate)) void requestAutoUpdate();
  });
  // A dragged point moves with every pointer move: wait for the drop. A button released outside
  // the window sends no pointerup, so a move with no button down, or leaving the window, counts too.
  const released = () => {
    pointersDown = 0;
    for (const resolve of pointersUp.splice(0)) resolve();
  };
  const listeners: [string, (event: PointerEvent) => void][] = [
    ["pointerdown", () => void (pointersDown += 1)],
    ["pointerup", () => (pointersDown <= 1 ? released() : void (pointersDown -= 1))],
    ["pointercancel", released],
    ["pointermove", (event) => event.buttons === 0 && pointersDown > 0 && released()],
  ];
  const listen = typeof window !== "undefined";
  if (listen) {
    for (const [type, listener] of listeners) window.addEventListener(type, listener as EventListener, true);
    window.addEventListener("blur", released);
  }
  return () => {
    unsubscribe();
    if (listen) {
      for (const [type, listener] of listeners) window.removeEventListener(type, listener as EventListener, true);
      window.removeEventListener("blur", released);
    }
    released();
  };
}
