/**
 * The extractors as the backend describes them (`GET /extractors`): which can run here, and
 * the settings each brings. Choosing an extractor loads those settings, and the first time the
 * backend answers, Learned is chosen by itself if its checkpoint is there -- the editor's
 * default extractor is the trained one whenever there is one.
 */

import { fetchExtractors } from "./backend";
import { store } from "./flow";
import { settingsForExtractor, type FeatureExtractorName } from "./settings";

let fetching: Promise<void> | null = null;

/**
 * Ask the backend about the extractors (once at a time), remember the answer, and choose the
 * default extractor if nobody has chosen one yet. Quiet if the backend cannot be reached: the
 * panel says so when a session is opened.
 */
export function refreshExtractors(): Promise<void> {
  fetching ??= (async () => {
    try {
      const infos = await fetchExtractors();
      store().setExtractors(infos);
      const { defaultsFrom, settings } = store().segmentation;
      if (defaultsFrom === null) chooseExtractor(infos.learned?.available ? "learned" : settings.featureExtractor);
    } catch {
      // The backend is off, or old: the extractors stay as the settings know them.
    } finally {
      fetching = null;
    }
  })();
  return fetching;
}

/** Switch to `extractor`, with its settings (`settingsForExtractor`) and the feature grid left to the image. */
export function chooseExtractor(extractor: FeatureExtractorName): void {
  store().setSegmentationSettings(settingsForExtractor(extractor, store().extractors[extractor]));
  store().setSegmentation({ defaultsFrom: extractor });
}

/** Put the current extractor's potential and inference settings back, leaving the features as they are. */
export function resetPotentials(): void {
  const { featureExtractor } = store().segmentation.settings;
  const { featureExtractor: _extractor, featureSpacing: _spacing, ...potentials } = settingsForExtractor(featureExtractor, store().extractors[featureExtractor]);
  store().setSegmentationSettings(potentials);
}

/** Describe the extractors when the editor starts. Returns the uninstall (nothing to undo). */
export function installExtractors(): () => void {
  void refreshExtractors();
  return () => {};
}
