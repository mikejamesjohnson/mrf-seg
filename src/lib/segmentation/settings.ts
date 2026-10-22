/**
 * The walker's settings: what the backend's `MRFSettings` holds (in camelCase; `backend.ts` converts),
 * and how the side panel offers each. Adding a setting means a field here and in
 * `python/mrfseg/settings.py`.
 */

export type FeatureExtractorName = "basic" | "learned";
export type PreconditionerName = "multigrid" | "jacobi";

export interface MRFSettings {
  /**
   * What gives the graph its affinities: Basic, Grady's contrast affinities from the image's
   * Hounsfield units; or Learned, our own network's log-affinities, trained on coronary CTA.
   * Changing it extracts the features again.
   */
  featureExtractor: FeatureExtractorName;
  /**
   * mm: the isotropic grid the features are extracted and the walker solved on; changing it
   * re-extracts the features. Null until the backend has chosen it from the image (`IMAGE_DERIVED`).
   */
  featureSpacing: number | null;
  /** Where the network and the solver run. */
  device: "auto" | "cpu";
  /** Basic: delta, HU. An edge whose two voxels differ by this much keeps 61 % of its affinity. */
  contrast: number;
  /** Learned: what the network's log-affinities are multiplied by; 1 is as trained. */
  affinityScale: number;
  /** Seed every voxel below seedHu (eroded seedErosion voxels) as outside: fat and air. */
  backgroundSeeds: boolean;
  /** HU: voxels below this are background seeds. */
  seedHu: number;
  /** Voxels: how far the background seeds keep from anything above seedHu. */
  seedErosion: number;
  /** The solver stops once no voxel's probability would still move by more than this. */
  tolerance: number;
  /** Conjugate-gradient iterations at most. */
  maxIterations: number;
  /** Multigrid (a few dozen iterations whatever the vessel's length) or Jacobi (cheaper iterations, hundreds of them). */
  preconditioner: PreconditionerName;
}

/** What the backend says about an extractor before it is chosen (`GET /extractors`). */
export interface ExtractorInfo {
  /** Whether it can run on this computer: Learned needs its checkpoint. */
  available: boolean;
  /** The settings to start from with it (Learned: as it was trained), or null for the editor's own defaults. */
  settings: Partial<MRFSettings> | null;
  /** mm: the grid it was trained on, or null. */
  nativeSpacing: number | null;
  /** Where its weights are, or why it is unavailable. */
  note: string | null;
}

export type ExtractorInfos = Partial<Record<FeatureExtractorName, ExtractorInfo>>;

/** The settings that are the affinities', the seeds' and the solver's, not the features': what an extractor may bring its own values for. */
export const POTENTIAL_KEYS = [
  "contrast",
  "affinityScale",
  "backgroundSeeds",
  "seedHu",
  "seedErosion",
  "tolerance",
  "maxIterations",
  "preconditioner",
] as const satisfies readonly (keyof MRFSettings)[];

/**
 * The settings to start from with `extractor`: the editor's defaults for the affinities, the
 * seeds and the solver, overlaid with the extractor's own where the backend knows them. The
 * feature grid is left to the image either way, so the backend chooses the extractor's native spacing.
 */
export function settingsForExtractor(extractor: FeatureExtractorName, info: ExtractorInfo | undefined): Partial<MRFSettings> {
  const own: Partial<MRFSettings> = {};
  for (const key of POTENTIAL_KEYS) (own as Record<string, unknown>)[key] = DEFAULT_MRF_SETTINGS[key];
  return { featureExtractor: extractor, featureSpacing: null, ...own, ...(info?.settings ?? {}) };
}

/**
 * Left to each image: when one opens it is null, and the backend chooses it from the image's
 * grid (`MRFSettings.resolved_for`): the feature grid at its finest voxel size (at least
 * 0.5 mm, doubled if the grid would be too large).
 */
export const IMAGE_DERIVED = ["featureSpacing"] as const satisfies readonly (keyof MRFSettings)[];

export const DEFAULT_MRF_SETTINGS: MRFSettings = {
  featureExtractor: "basic",
  featureSpacing: null,
  device: "auto",
  contrast: 40,
  affinityScale: 1,
  backgroundSeeds: true,
  seedHu: 0,
  seedErosion: 1,
  tolerance: 0.0001,
  maxIterations: 1000,
  preconditioner: "multigrid",
};

/** What the panel and the status line call each extractor. */
export const EXTRACTOR_LABELS: Record<FeatureExtractorName, string> = { basic: "Basic", learned: "Learned" };

/** Whether a Generate needs an inside point: always, for the walker -- without one nothing is selected. */
export function needsInsidePoint(_extractor: FeatureExtractorName): boolean {
  return true;
}

type Keys<T> = { [K in keyof MRFSettings]: MRFSettings[K] extends T ? K : never }[keyof MRFSettings];

/** A dropdown over a setting whose values are a few names: the options are typed to that setting's. */
type Choice = { [K in Keys<string>]: { kind: "choice"; key: K; options: { value: MRFSettings[K]; label: string }[] } }[Keys<string>];

export type SettingField = {
  label: string;
  hint?: string;
  /** Shown only when this holds: options of one extractor are hidden while the other is chosen. */
  when?: (settings: MRFSettings) => boolean;
} & (
  /** `min` and `max` are the backend's `BOUNDS` (a test checks); `step` is the spinner's. */
  | { kind: "number"; key: Keys<number | null>; min: number; max: number; step: number; integer?: boolean }
  | Choice
  | { kind: "check"; key: Keys<boolean> }
);

/** The settings a field sets. */
export function fieldKeys(field: SettingField): (keyof MRFSettings)[] {
  return [field.key];
}

/** The fields the panel shows for `settings`. */
export function shownFields(group: SettingGroup, settings: MRFSettings): SettingField[] {
  return group.fields.filter((field) => !field.when || field.when(settings));
}

export interface SettingGroup {
  title: string;
  note: string;
  fields: SettingField[];
}

const basic = (settings: MRFSettings) => settings.featureExtractor === "basic";
const learned = (settings: MRFSettings) => settings.featureExtractor === "learned";

export const SETTING_GROUPS: SettingGroup[] = [
  {
    title: "Affinities",
    note: "w: how freely the label flows between neighbours",
    fields: [
      {
        kind: "choice",
        key: "featureExtractor",
        label: "Affinities from",
        hint:
          "Basic: Grady's contrast affinities, exp(-(ΔHU)² / 2δ²), the classic random walker. " +
          "Learned: our own network's log-affinities, trained on coronary CTA so that one point carries through its vessel and no further (needs its checkpoint, MRFSEG_LEARNED_WEIGHTS). " +
          "Changing it extracts the features again, and loads that extractor's settings.",
        options: (Object.keys(EXTRACTOR_LABELS) as FeatureExtractorName[]).map((value) => ({ value, label: EXTRACTOR_LABELS[value] })),
      },
      {
        kind: "number",
        key: "featureSpacing",
        label: "Resampling spacing, mm",
        hint:
          "The isotropic grid the features are extracted and the walker solved on; changing it re-extracts the features. " +
          "Starts at the image's voxel size, at least 0.5 mm (Learned: the 0.5 mm it was trained on), doubled until the grid fits in memory. " +
          "Coarser is faster: the solver's time and memory go with the number of voxels.",
        min: 0.001,
        max: 100,
        step: 0.05,
      },
      {
        kind: "number",
        key: "contrast",
        label: "δ, contrast HU",
        hint: "An edge whose two voxels differ by this much in HU keeps 61 % of its affinity; by three times this, 1 %. Smaller cuts fainter edges, and the noise within the lumen too.",
        min: 1,
        max: 10000,
        step: 5,
        when: basic,
      },
      {
        kind: "number",
        key: "affinityScale",
        label: "Scale",
        hint:
          "What the network's log-affinities are multiplied by (1 is as trained). Above 1 sharpens the contrast between the edges it finds strong and weak: " +
          "a point reaches further and leaks less where the network was right, and its mistakes count more too.",
        min: 0.1,
        max: 5,
        step: 0.1,
        when: learned,
      },
      {
        kind: "choice",
        key: "device",
        label: "Compute on",
        hint: "The network and the solver",
        options: [
          { value: "auto", label: "GPU if present" },
          { value: "cpu", label: "CPU" },
        ],
      },
    ],
  },
  {
    title: "Seeds",
    note: "the points, and the background, held at 1 and 0",
    fields: [
      {
        kind: "check",
        key: "backgroundSeeds",
        label: "Seed the background",
        hint:
          "Hold every voxel below the HU threshold at 0, as an outside point would: fat and air surround the coronaries within a few voxels, " +
          "and drain what a point's vessel does not hold in. Off, only the outside points pull the probability down.",
      },
      {
        kind: "number",
        key: "seedHu",
        label: "Below, HU",
        hint: "Voxels below this are background seeds. Fat is about −100 HU, soft tissue above 0, contrast-filled lumen above 200.",
        min: -1000,
        max: 1000,
        step: 10,
        when: (settings) => settings.backgroundSeeds,
      },
      {
        kind: "number",
        key: "seedErosion",
        label: "Keep away, voxels",
        hint: "How far the background seeds keep from anything above the threshold, so a partial-volume voxel at a vessel's edge is never seeded.",
        min: 0,
        max: 5,
        step: 1,
        integer: true,
        when: (settings) => settings.backgroundSeeds,
      },
    ],
  },
  {
    title: "Solver",
    note: "L p = b: every free voxel the affinity-weighted mean of its neighbours",
    fields: [
      {
        kind: "number",
        key: "tolerance",
        label: "Tolerance",
        hint: "Stop once the solver proposes no correction larger than this to any voxel's probability. Smaller is more exact and slower.",
        min: 1e-7,
        max: 0.1,
        step: 0.0001,
      },
      {
        kind: "number",
        key: "maxIterations",
        label: "Iterations at most",
        hint: "The result says whether the solver converged within these.",
        min: 1,
        max: 100000,
        step: 100,
        integer: true,
      },
      {
        kind: "choice",
        key: "preconditioner",
        label: "Preconditioner",
        hint: "Multigrid: a few dozen iterations whatever the vessel's length. Jacobi: cheaper iterations, but about two per voxel of vessel, and a convergence measure a long vessel can fool.",
        options: [
          { value: "multigrid", label: "Multigrid" },
          { value: "jacobi", label: "Jacobi" },
        ],
      },
    ],
  },
];

/** The group whose fields a panel section shows. */
export function settingGroup(title: string): SettingGroup {
  const group = SETTING_GROUPS.find((candidate) => candidate.title === title);
  if (!group) throw new Error(`No setting group ${title}`);
  return group;
}
