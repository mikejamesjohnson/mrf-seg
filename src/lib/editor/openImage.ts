/**
 * Open an image by path: load it, replace whatever was open, start its segmentation, and keep
 * the path in the URL (`/editor?image=/path/to/file.mhd`) so a reload or a bookmark opens it.
 */

import { toast } from "sonner";

import { fileName, loadImageFromPath } from "@/lib/io/loaders";
import { useSceneStore } from "@/lib/scene/store";
import { connectSegmentation, releaseSession } from "@/lib/segmentation/session";

export const IMAGE_URL_PARAM = "image";

/** Only the newest request is applied, so a slow load cannot overwrite a later one. */
let latestRequest = 0;

/** Whether the image was opened; if not, the status bar and a toast say why. */
export async function openImage(path: string): Promise<boolean> {
  const trimmed = path.trim();
  if (!trimmed) return false;
  const request = ++latestRequest;
  const store = useSceneStore.getState();
  store.setStatus({ state: "running", message: `Loading ${fileName(trimmed)}…` });

  try {
    const { dataset, image } = await loadImageFromPath(trimmed);
    if (request !== latestRequest) {
      image.delete();
      return false;
    }
    // One image at a time: its maps, mesh and backend session go with it, and the new one
    // starts with every view through its centre, no points, and a fresh segmentation.
    releaseSession();
    for (const id of useSceneStore.getState().datasetOrder) store.removeDataset(id);
    store.addDataset(dataset, image);
    store.resetViews();
    store.clearPoints();
    store.resetSegmentation(dataset.id);
    store.setStatus({ state: "idle" });
    // Not awaited: the image is usable without it, and the Segmentation panel says how it went.
    void connectSegmentation(dataset);

    const url = new URL(window.location.href);
    url.searchParams.set(IMAGE_URL_PARAM, trimmed);
    window.history.replaceState(null, "", url);
    return true;
  } catch (error) {
    if (request !== latestRequest) return false;
    const message = error instanceof Error ? error.message : String(error);
    store.setStatus({ state: "error", message });
    toast.error(`Could not open ${fileName(trimmed)}`, { description: message });
    return false;
  }
}
