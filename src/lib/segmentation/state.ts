/**
 * The segmentation's state in the scene store. The MRFSegmentation itself lives in the backend,
 * one per image; this is the editor's handle on it: its settings, how far its background work
 * has got, what the last Generate produced, and what has been built from that since. Its maps,
 * labelmap and mesh are datasets.
 */

import { pointCounts } from "@/lib/scene/points";
import type { SceneState, StoreSlice } from "@/lib/scene/store";
import type { DatasetId, SurfaceSource, TaskStatus } from "@/lib/scene/types";
import type { Vec3 } from "@/lib/world";
import { DEFAULT_MRF_SETTINGS, IMAGE_DERIVED, needsInsidePoint, settingsForExtractor, type ExtractorInfos, type FeatureExtractorName, type MRFSettings } from "./settings";

/** Where a Generate has got to: the backend's stages, then the editor fetching the maps. */
export type TaskStage = "queued" | "features" | "potentials" | "inference" | "maps";

/** A background task in the backend, as last polled. */
export interface TaskProgress {
  state: "idle" | "running" | "done" | "error";
  stage?: TaskStage | null;
  /** 0-1 through the current stage. */
  progress: number;
  message?: string;
}

/** What the backend says about the features, once extracted (`sessions._features_summary`). */
export interface FeaturesSummary {
  /** Which extractor made them. */
  extractor: FeatureExtractorName;
  dims: Vec3;
  /** mm: the grid they were extracted on. */
  spacing: number;
  channels: number;
  seconds: number;
}

/** How the mean-field inference ended (`sessions._inference_summary`). */
export interface InferenceSummary {
  /** Full sweeps made: fewer than asked when it converged early. */
  sweeps: number;
  /** The largest change of any probability in the last sweep. */
  delta: number;
  converged: boolean;
  /** How many connected components the inside points kept (the Connected setting); null when nothing was dropped. */
  components: number | null;
}

/** What the maps show: the last successful Generate. */
export interface GeneratedSegmentation {
  points: number;
  /** The points' version, the settings, and the backend session it was made with: when any of
   * them differs from the current one, the maps are out of date (`segmentationIsStale`). */
  pointsVersion: number;
  settings: MRFSettings;
  sessionId: string;
  seconds: { features: number; potentials: number; inference: number; maps: number };
  inference: InferenceSummary;
}

export interface SegmentationState {
  /** The image being segmented. */
  imageId: DatasetId | null;
  /** The backend session holding its MRFSegmentation, once there is one. */
  sessionId: string | null;
  settings: MRFSettings;
  /** The extractor whose settings were loaded (`settingsForExtractor`); null: nobody has chosen one yet. */
  defaultsFrom: FeatureExtractorName | null;
  /** Extracted in the background when the session opens, and again when a Generate needs new ones. */
  features: TaskProgress & Partial<FeaturesSummary>;
  /** The Generate in progress, or the last one. */
  job: TaskProgress;
  generated: GeneratedSegmentation | null;
  /** Successful Generates so far, so a product can tell it was built from an older one. */
  generation: number;
  /** The connection to the backend: opening the session, and losing it. */
  status: TaskStatus;
}

/** Something built from the last segmentation on request, and what it was built from. */
export interface Product<Built> {
  status: TaskStatus;
  built: (Built & { seconds: number; generation: number }) | null;
}

export interface LabelmapState extends Product<{ threshold: number; voxels: number }> {
  /** Voxels with p(y = 1) at least this are labelled 1. */
  threshold: number;
}

export interface SurfaceState extends Product<{ source: SurfaceSource; level: number; triangles: number }> {
  source: SurfaceSource;
  /** p(y = 1) to threshold at, for a probability surface. */
  threshold: number;
}

export type ProductKey = "labelmap" | "surface";

/** Each step with an Auto box in the Generate section. */
export type AutoKey = "segmentation" | ProductKey;

/** Which steps run by themselves when what they are made from changes (`autoUpdate.ts`). */
export type AutoUpdateSettings = Record<AutoKey, boolean>;

export interface SegmentationSlice {
  segmentation: SegmentationState;
  labelmap: LabelmapState;
  surface: SurfaceState;
  /** A preference, so a new image keeps it. */
  autoUpdate: AutoUpdateSettings;
  /** What the backend said about each extractor, once asked (`fetchExtractors`). */
  extractors: ExtractorInfos;
  setSegmentation: (patch: Partial<SegmentationState>) => void;
  setSegmentationSettings: (patch: Partial<MRFSettings>) => void;
  setProduct: <K extends ProductKey>(key: K, patch: Partial<SegmentationSlice[K]>) => void;
  setAutoUpdate: (patch: Partial<AutoUpdateSettings>) => void;
  setExtractors: (extractors: ExtractorInfos) => void;
  /** A fresh start for a newly opened image, keeping the settings and the thresholds, but for those the image chooses. */
  resetSegmentation: (imageId: DatasetId) => void;
}

// --- what the panels and auto update ask of the state -------------------------------------

/** Whether a Generate or a build is running, whoever started it: nothing else may start. */
export function isBusy(state: SceneState): boolean {
  return state.segmentation.job.state === "running" || state.labelmap.status.state === "running" || state.surface.status.state === "running";
}

/** Whether there is enough to segment from: an inside point, unless the extractor needs none. */
export function enoughPoints(state: SceneState): boolean {
  if (!needsInsidePoint(state.segmentation.settings.featureExtractor)) return true;
  return pointCounts(state.points).inside > 0;
}

/** Why the maps no longer show what Generate would give now; empty if they do (or there are none). */
export function staleReasons(state: SceneState): string[] {
  const { generated, settings, sessionId } = state.segmentation;
  if (!generated) return [];
  const reasons: string[] = [];
  if (generated.pointsVersion !== state.pointsVersion) reasons.push("the points have changed");
  if ((Object.keys(settings) as (keyof MRFSettings)[]).some((key) => settings[key] !== generated.settings[key])) reasons.push("the settings have changed");
  if (sessionId !== generated.sessionId) reasons.push("the backend session is new, and has no segmentation yet");
  return reasons;
}

/** Whether there is a segmentation to build a product from: one generated, in the session that still holds it. */
export function canBuild(state: SceneState): boolean {
  const { generated, sessionId } = state.segmentation;
  return generated !== null && sessionId === generated.sessionId;
}

/** Whether the maps no longer show what Generate would give now. */
export function segmentationIsStale(state: SceneState): boolean {
  return staleReasons(state).length > 0;
}

/** Whether Generate would show something the maps do not. */
export function needsGenerate(state: SceneState): boolean {
  return enoughPoints(state) && (state.segmentation.generated === null || segmentationIsStale(state));
}

/** The isovalue a surface is built at: 0 for the unary logit, the threshold for the probability. */
export function surfaceLevel({ source, threshold }: Pick<SurfaceState, "source" | "threshold">): number {
  return source === "unary" ? 0 : threshold;
}

/** Whether building `key` now would give something other than what is there. */
export function productStale(state: SceneState, key: ProductKey): boolean {
  const generation = state.segmentation.generation;
  if (key === "labelmap") {
    const built = state.labelmap.built;
    return !built || built.generation !== generation || built.threshold !== state.labelmap.threshold;
  }
  const built = state.surface.built;
  return !built || built.generation !== generation || built.source !== state.surface.source || built.level !== surfaceLevel(state.surface);
}

// --- the slice -----------------------------------------------------------------------------

const IDLE: TaskProgress = { state: "idle", progress: 0 };

function freshSegmentation(imageId: DatasetId | null, settings: MRFSettings, defaultsFrom: FeatureExtractorName | null): SegmentationState {
  return {
    imageId,
    sessionId: null,
    settings,
    defaultsFrom,
    features: IDLE,
    job: IDLE,
    generated: null,
    generation: 0,
    status: { state: "idle" },
  };
}

export const segmentationSlice: StoreSlice<SegmentationSlice> = (set) => ({
  segmentation: freshSegmentation(null, DEFAULT_MRF_SETTINGS, null),
  labelmap: { threshold: 0.5, status: { state: "idle" }, built: null },
  surface: { source: "probability", threshold: 0.5, status: { state: "idle" }, built: null },
  autoUpdate: { segmentation: false, labelmap: false, surface: false },
  extractors: {},

  setSegmentation: (patch) => set((state) => ({ segmentation: { ...state.segmentation, ...patch } })),
  setSegmentationSettings: (patch) =>
    set((state) => ({ segmentation: { ...state.segmentation, settings: { ...state.segmentation.settings, ...patch } } })),
  setProduct: (key, patch) => set((state) => ({ [key]: { ...state[key], ...patch } }) as Partial<SceneState>),
  setAutoUpdate: (patch) => set((state) => ({ autoUpdate: { ...state.autoUpdate, ...patch } })),
  setExtractors: (extractors) => set({ extractors }),
  resetSegmentation: (imageId) =>
    set((state) => {
      // The new image chooses what the extractor leaves to the image, as a new session fills it
      // in; a value the extractor brings is not the image's to choose.
      const { settings } = state.segmentation;
      const defaults = settingsForExtractor(settings.featureExtractor, state.extractors[settings.featureExtractor]);
      const chosenByImage = IMAGE_DERIVED.filter((key) => defaults[key] === null);
      return {
        segmentation: freshSegmentation(imageId, { ...settings, ...Object.fromEntries(chosenByImage.map((key) => [key, null])) }, state.segmentation.defaultsFrom),
        labelmap: { ...state.labelmap, status: { state: "idle" }, built: null },
        surface: { ...state.surface, status: { state: "idle" }, built: null },
      };
    }),
});
