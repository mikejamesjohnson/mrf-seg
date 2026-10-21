/** Generate: build the CRF's potentials from the labelled points, infer the marginals with the points clamped, and store the three maps. */

import { imageOnGrid } from "@/lib/io/loaders";
import { currentImage, onGridOf, voxelCount } from "@/lib/scene/datasets";
import type { ImageDataset, MapDataset, MapQuantity } from "@/lib/scene/types";
import { BackendError, fetchMap, startSegmentation, type MapSummary, type SessionStatus } from "./backend";
import { isCurrent, messageOf, poll, progressOf, sessionFailed, store } from "./flow";
import { MAP_DISPLAY, MAP_QUANTITIES } from "./maps";
import { connectSegmentation } from "./session";
import { enoughPoints, type TaskProgress } from "./state";

/** Counts Generates asked for: only the latest may write its result. */
let latestRequest = 0;

export async function generateSegmentation(): Promise<void> {
  const state = store();
  const image = currentImage(state);
  if (!image) return;
  const request = ++latestRequest;
  const current = () => request === latestRequest && isCurrent(image.id);
  const report = (job: TaskProgress) => current() && store().setSegmentation({ job });
  const fail = (message: string) => report({ state: "error", progress: 0, message });

  if (!enoughPoints(state)) return void fail("Add at least one inside point.");
  const { pointsVersion } = state;
  const points = state.points.map((point) => point.position);
  const labels = state.points.map((point) => point.label);
  report({ state: "running", stage: "queued", progress: 0 });

  try {
    // No session yet (the backend was off when the image was opened), or it has been lost
    // since (the backend restarted, or dropped it for a newer one): once more, afresh.
    let sessionId = state.segmentation.sessionId ?? (await connectSegmentation(image));
    if (!current()) return;
    // Read once the session exists: opening one fills in what the settings leave to the image.
    const { settings } = store().segmentation;
    if (sessionId) {
      try {
        await startSegmentation(sessionId, points, labels, settings);
      } catch (error) {
        if (!(error instanceof BackendError && error.sessionLost)) throw error;
        if (!current()) return;
        store().setSegmentation({ sessionId: null });
        sessionId = await connectSegmentation(image);
        if (!current()) return;
        if (sessionId) await startSegmentation(sessionId, points, labels, settings);
      }
    }
    if (!sessionId) {
      const status = store().segmentation.status;
      return void fail(status.state === "error" ? status.message : "The segmentation backend could not be reached.");
    }

    let finished: SessionStatus["job"] | null = null;
    const id = sessionId;
    let seen: boolean;
    try {
      seen = await poll(id, image.id, (status) => {
        if (!current()) return true;
        store().setSegmentation({
          job: progressOf(status.job),
          features: { ...progressOf(status.features), ...status.features.result },
          // Features extracted afresh clear a feature error the session opened with.
          ...(status.features.state === "done" && store().segmentation.status.state === "error" ? { status: { state: "idle" as const } } : {}),
        });
        if (status.job.state === "running") return false;
        finished = status.job;
        return true;
      });
    } catch (error) {
      sessionFailed(image.id, id, error);
      throw error;
    }
    if (!current()) return;
    if (!seen) return void fail("The backend session changed while generating; generate again.");
    const job = finished as SessionStatus["job"] | null;
    if (!job || job.state !== "done" || !job.result) return void fail(job?.message ?? "Generate failed");

    // The job is "done" in the backend, but not here until the maps are stored: nothing may
    // start another Generate in between, and the maps must arrive together.
    report({ state: "running", stage: "maps", progress: 0 });
    const result = job.result;
    const maps = await Promise.all(MAP_QUANTITIES.map((quantity) => fetchMap(id, quantity, voxelCount(image))));
    if (!current()) return;
    MAP_QUANTITIES.forEach((quantity, i) => storeMap(image, quantity, maps[i], result.maps[quantity]));
    store().setSegmentation({
      generated: { points: result.points, pointsVersion, settings, sessionId: id, seconds: result.seconds, inference: result.inference },
      generation: store().segmentation.generation + 1,
      job: { state: "done", stage: null, progress: 1 },
    });
  } catch (error) {
    fail(messageOf(error));
  }
}

/** A map as a dataset on the image's grid, added or replaced in place. */
function storeMap(image: ImageDataset, quantity: MapQuantity, values: Int16Array | Uint8Array, { range, scale }: MapSummary): void {
  const display = MAP_DISPLAY[quantity];
  const dataset: MapDataset = {
    ...onGridOf(image, quantity),
    kind: "map",
    label: display.label,
    quantity,
    valueScale: scale,
    arrayName: quantity,
    scalarType: values.constructor.name,
    scalarRange: range,
  };
  const span = display.span(range);
  const isNew = !store().datasets[dataset.id];
  store().putMap(dataset, imageOnGrid(image, quantity, values), isNew ? { ...display.firstOverlay, range: span } : { range: span });
}
