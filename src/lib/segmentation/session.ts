/**
 * The open image's MRFSegmentation in the backend: opened with the image (one connection at a
 * time per image), followed while it extracts features, and closed when the image is replaced
 * or the page goes away.
 */

import { currentImage } from "@/lib/scene/datasets";
import type { ImageDataset } from "@/lib/scene/types";
import { BackendError, closeSession, openSession, type BackendGeometry } from "./backend";
import { refreshExtractors } from "./extractors";
import { isCurrent, messageOf, poll, progressOf, sessionFailed, store } from "./flow";
import { IMAGE_DERIVED, type MRFSettings } from "./settings";

/** The backend session this page holds, and the connection being made, if one is. */
let held: string | null = null;
let connecting: { imageId: string; promise: Promise<string | null> } | null = null;

function hold(sessionId: string): void {
  if (held && held !== sessionId) closeSession(held);
  held = sessionId;
}

/** Close the session this page holds: its image is going, or the page is. */
export function releaseSession(): void {
  if (held) closeSession(held);
  held = null;
}

/**
 * Keep the session in step with the page's life: closed as the page goes, and opened again if
 * the page comes back from the back/forward cache. Returns the uninstall.
 */
export function installSessionLifecycle(): () => void {
  const show = (event: PageTransitionEvent) => {
    const image = currentImage(store());
    if (event.persisted && image) {
      store().setSegmentation({ sessionId: null });
      void connectSegmentation(image);
    }
  };
  window.addEventListener("pagehide", releaseSession);
  window.addEventListener("pageshow", show);
  return () => {
    window.removeEventListener("pagehide", releaseSession);
    window.removeEventListener("pageshow", show);
  };
}

/** Where the backend's reading of the file differs from the editor's, or null: maps are drawn on the editor's grid. */
function geometryMismatch(image: ImageDataset, geometry: BackendGeometry): string | null {
  const close = (a: readonly number[], b: readonly number[], tolerance: number) =>
    a.length === b.length && a.every((v, i) => Math.abs(v - b[i]) <= tolerance);
  if (!close(image.dimensions, geometry.dimensions, 0)) return `${geometry.dimensions.join(" × ")} voxels, not ${image.dimensions.join(" × ")}`;
  if (!close(image.spacing, geometry.spacing, 1e-6)) return "a different voxel spacing";
  if (!close(image.gridOrigin, geometry.origin, 1e-4)) return "a different origin";
  if (!close(image.direction, geometry.direction, 1e-6)) return "a different orientation";
  return null;
}

/**
 * The image's MRFSegmentation, opened if there is none: its session id, or null if it could not
 * be opened (the panel then says why). Asking again while it is being opened waits for that.
 */
export function connectSegmentation(image: ImageDataset): Promise<string | null> {
  if (connecting?.imageId === image.id) return connecting.promise;
  const promise = open(image).finally(() => {
    if (connecting?.promise === promise) connecting = null;
  });
  connecting = { imageId: image.id, promise };
  return promise;
}

async function open(image: ImageDataset): Promise<string | null> {
  if (!isCurrent(image.id)) return null;
  store().setSegmentation({
    sessionId: null,
    status: { state: "running", message: "Starting MRFSegmentation…" },
    features: { state: "running", stage: "features", progress: 0 },
  });
  try {
    // What the backend offers, and so which extractor (and settings) this session opens with:
    // asked first, so the features it extracts are the right extractor's from the start.
    await refreshExtractors();
    if (!isCurrent(image.id)) return null;
    const session = await openSession(image.path, store().segmentation.settings);
    if (!isCurrent(image.id)) {
      closeSession(session.id);
      return null;
    }
    const mismatch = geometryMismatch(image, session.geometry);
    if (mismatch) {
      closeSession(session.id);
      throw new BackendError(`The backend read ${image.label} with ${mismatch}.`, 0);
    }
    hold(session.id);
    store().setSegmentation({ sessionId: session.id, status: { state: "idle" } });
    // What the backend chose from the image, where the settings left it to it.
    const current = store().segmentation.settings;
    const chosen: Partial<MRFSettings> = {};
    for (const key of IMAGE_DERIVED) if (current[key] === null) chosen[key] = session.settings[key];
    store().setSegmentationSettings(chosen);
    void followFeatures(session.id, image.id);
    return session.id;
  } catch (error) {
    if (isCurrent(image.id)) {
      store().setSegmentation({ sessionId: null, features: { state: "idle", progress: 0 }, status: { state: "error", message: messageOf(error) } });
    }
    return null;
  }
}

async function followFeatures(sessionId: string, imageId: string): Promise<void> {
  try {
    await poll(sessionId, imageId, ({ features }) => {
      store().setSegmentation({
        features: { ...progressOf(features), ...features.result },
        ...(features.state === "error" ? { status: { state: "error", message: features.message ?? "Feature extraction failed" } } : {}),
      });
      return features.state !== "running";
    });
  } catch (error) {
    sessionFailed(imageId, sessionId, error);
  }
}
