/**
 * What the segmentation's asynchronous flows share: the store, the guard that drops anything
 * for an image or a session no longer current, polling the backend, and what to do when a
 * session cannot be reached any more.
 */

import { currentImage } from "@/lib/scene/datasets";
import { useSceneStore } from "@/lib/scene/store";
import { BackendError, sessionStatus, type BackendTask, type SessionStatus } from "./backend";
import type { TaskProgress } from "./state";

const POLL_MS = 400;
/** Polls in a row that may fail with no answer, or a server error, before the session is given up. */
const POLL_RETRIES = 5;

export const store = () => useSceneStore.getState();

/** Whether `imageId` is still the image being segmented; anything for another is dropped. */
export function isCurrent(imageId: string): boolean {
  const state = store();
  return state.segmentation.imageId === imageId && currentImage(state)?.id === imageId;
}

/** Whether `sessionId` is still the backend session for `imageId`. */
export function isLive(imageId: string, sessionId: string): boolean {
  return isCurrent(imageId) && store().segmentation.sessionId === sessionId;
}

export function messageOf(error: unknown): string {
  return error instanceof Error ? error.message : String(error);
}

export function progressOf(task: BackendTask<unknown>): TaskProgress {
  const { result: _result, ...progress } = task;
  return progress;
}

export const sleep = (ms: number) => new Promise((resolve) => setTimeout(resolve, ms));

/**
 * Poll the session until `seen` says it has what it wants. False if, by the time an answer
 * arrives, the image or the session is no longer the current one: the answer is then dropped,
 * and so is any error, which belongs to a session nobody is waiting on.
 */
export async function poll(sessionId: string, imageId: string, seen: (status: SessionStatus) => boolean): Promise<boolean> {
  const live = () => isLive(imageId, sessionId);
  for (let failures = 0; live(); ) {
    let status: SessionStatus;
    try {
      status = await sessionStatus(sessionId);
      failures = 0;
    } catch (error) {
      if (!live()) return false;
      if (!(error instanceof BackendError && error.transient) || ++failures > POLL_RETRIES) throw error;
      await sleep(POLL_MS);
      continue;
    }
    if (!live()) return false;
    if (seen(status)) return true;
    await sleep(POLL_MS);
  }
  return false;
}

/**
 * Say that `sessionId` could not be reached: without a session, the panel offers to open
 * another, and a Generate that was running is over. Nothing happens if that session has
 * already been replaced, or its image closed.
 */
export function sessionFailed(imageId: string, sessionId: string, error: unknown): void {
  if (!isLive(imageId, sessionId)) return;
  const lost = error instanceof BackendError && error.sessionLost;
  const message = lost ? "The backend no longer has this image's session (it restarted, or another tab replaced it)." : messageOf(error);
  const { job } = store().segmentation;
  const over = (task: TaskProgress): TaskProgress => (task.state === "running" ? { state: "error", progress: 0, message } : task);
  store().setSegmentation({
    sessionId: null,
    features: { state: "idle", progress: 0 },
    job: over(job),
    status: { state: "error", message },
  });
}
