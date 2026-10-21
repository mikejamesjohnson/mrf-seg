/**
 * The CRF's settings: what the backend's `MRFSettings` holds (in camelCase; `backend.ts` converts),
 * and how the side panel offers each. Adding a setting means a field here and in
 * `python/mrfseg/settings.py`.
 */

export type FeatureExtractorName = "basic" | "learned";

export interface MRFSettings {
  /**
   * What gives each voxel its features, and so how the CRF's potentials are made: Basic, the
   * image's brightness at several scales with handcrafted potentials from the points; or
   * Learned, our own network's potentials, trained on coronary CTA. Changing it extracts the
   * features again.
   */
  featureExtractor: FeatureExtractorName;
  /**
   * mm: the isotropic grid the features are extracted and the CRF inferred on; changing it
   * re-extracts the features. Null until the backend has chosen it from the image (`IMAGE_DERIVED`).
   */
  featureSpacing: number | null;
  /** Basic: how many blurs, sigma = the grid spacing and doublings of it. */
  basicScales: number;
  /** Basic: add each blur's edge strength. */
  basicEdges: boolean;
  /** Basic: add each blur's local shape, the eigenvalues of its Hessian. */
  basicShape: boolean;
  /** Where the network, the potentials and the inference run. */
  device: "auto" | "cpu";
  /** Basic: l_f, the width of the points' kernels in feature space, in standard deviations of the image's features. */
  featureLengthscale: number;
  /** Basic: l_s, mm, how far a point's evidence reaches. Null until the backend has chosen it from the image. */
  spatialLengthscale: number | null;
  /** Basic: a_0, added to every unary logit; negative asks for more evidence before a voxel is inside. */
  priorLogit: number;
  /** Basic: beta, how strongly the points' evidence counts against the coupling. */
  unaryScale: number;
  /** Learned: added to every unary logit the network gives. */
  unaryShift: number;
  /** lambda: the coupling of an edge with no contrast across it (Basic), or what the network's couplings are multiplied by (Learned). */
  coupling: number;
  /** Basic: delta, HU: how large an intensity step makes an edge cheap to cut. */
  contrast: number;
  /** Basic: the share of the coupling every edge keeps whatever its contrast. */
  couplingFloor: number;
  /** Mean-field sweeps; each updates both parities of the checkerboard. */
  sweeps: number;
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

/** The settings that are the potentials' and the inference's, not the features': what an extractor may bring its own values for. */
export const POTENTIAL_KEYS = [
  "featureLengthscale",
  "spatialLengthscale",
  "priorLogit",
  "unaryScale",
  "unaryShift",
  "coupling",
  "contrast",
  "couplingFloor",
  "sweeps",
] as const satisfies readonly (keyof MRFSettings)[];

/**
 * The settings to start from with `extractor`: the editor's defaults for the potentials and
 * the inference, overlaid with the extractor's own where the backend knows them (Learned: the
 * sweeps it was trained through, its couplings as trained). The feature grid is left to the
 * image either way, so the backend chooses the extractor's native spacing.
 */
export function settingsForExtractor(extractor: FeatureExtractorName, info: ExtractorInfo | undefined): Partial<MRFSettings> {
  const own: Partial<MRFSettings> = {};
  for (const key of POTENTIAL_KEYS) (own as Record<string, unknown>)[key] = DEFAULT_MRF_SETTINGS[key];
  return { featureExtractor: extractor, featureSpacing: null, ...own, ...(info?.settings ?? {}) };
}

/**
 * Left to each image: when one opens they are null, and the backend chooses them from its
 * grid (`MRFSettings.resolved_for`): the feature grid at its finest voxel size (at least
 * 0.5 mm, doubled if the grid would be too large), and l_s at 20 of its largest.
 */
export const IMAGE_DERIVED = ["featureSpacing", "spatialLengthscale"] as const satisfies readonly (keyof MRFSettings)[];

export const DEFAULT_MRF_SETTINGS: MRFSettings = {
  featureExtractor: "basic",
  featureSpacing: null,
  basicScales: 4,
  basicEdges: false,
  basicShape: false,
  device: "auto",
  featureLengthscale: 0.5,
  spatialLengthscale: null,
  priorLogit: -2,
  unaryScale: 1,
  unaryShift: 0,
  coupling: 1,
  contrast: 100,
  couplingFloor: 0.1,
  sweeps: 50,
};

/** What the panel and the status line call each extractor. */
export const EXTRACTOR_LABELS: Record<FeatureExtractorName, string> = { basic: "Basic", learned: "Learned" };

/** Whether a Generate needs an inside point: the handcrafted unary does, the network segments with none. */
export function needsInsidePoint(extractor: FeatureExtractorName): boolean {
  return extractor !== "learned";
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
    title: "Feature extraction",
    note: "one vector per voxel, on the CRF's grid",
    fields: [
      {
        kind: "choice",
        key: "featureExtractor",
        label: "Extractor",
        hint:
          "Basic: the image's brightness at several scales, with handcrafted potentials from the points. " +
          "Learned: our own network's potentials, trained on coronary CTA (needs its checkpoint, MRFSEG_LEARNED_WEIGHTS). " +
          "Changing it extracts the features again, and loads that extractor's settings.",
        options: (Object.keys(EXTRACTOR_LABELS) as FeatureExtractorName[]).map((value) => ({ value, label: EXTRACTOR_LABELS[value] })),
      },
      {
        kind: "number",
        key: "featureSpacing",
        label: "Resampling spacing, mm",
        hint:
          "The isotropic grid the features are extracted and the CRF inferred on; changing it re-extracts the features. " +
          "Starts at the image's voxel size, at least 0.5 mm (Learned: the 0.5 mm it was trained on), doubled until the grid fits in memory. " +
          "Coarser is faster: the inference's time and memory go with the number of voxels.",
        min: 0.001,
        max: 100,
        step: 0.05,
      },
      {
        kind: "number",
        key: "basicScales",
        label: "Scales",
        hint:
          "How many blurs: sigma is the grid spacing and doublings of it, so 4 is 1, 2, 4 and 8 times it. " +
          "The coarse blurs tell a thin vessel from a chamber: both are bright, but a blur wider than the vessel dims it and not the chamber.",
        min: 1,
        max: 8,
        step: 1,
        integer: true,
        when: basic,
      },
      { kind: "check", key: "basicEdges", label: "Edges", hint: "Add each blur's edge strength, the magnitude of its gradient.", when: basic },
      {
        kind: "check",
        key: "basicShape",
        label: "Shape",
        hint: "Add each blur's local shape: the three eigenvalues of its Hessian, which tell a tube from a blob, a sheet or a flat region. Slower to extract.",
        when: basic,
      },
      {
        kind: "choice",
        key: "device",
        label: "Compute on",
        hint: "The network, the potentials and the inference; Basic's blurs always run on the CPU",
        options: [
          { value: "auto", label: "GPU if present" },
          { value: "cpu", label: "CPU" },
        ],
      },
    ],
  },
  {
    title: "Unary potential",
    note: "a: evidence per voxel, positive for lumen",
    fields: [
      {
        kind: "number",
        key: "featureLengthscale",
        label: "ℓ features",
        hint:
          "How far from an inside point's features a voxel may be and still count as lumen, in standard deviations of the image's own features. " +
          "Wider reaches further along a vessel, but into other things too.",
        min: 0.001,
        max: 10,
        step: 0.05,
        when: basic,
      },
      {
        kind: "number",
        key: "spatialLengthscale",
        label: "ℓ position, mm",
        hint: "How far a point's evidence reaches; 20 times the image's largest voxel size to start with. Large makes the evidence feature-only: a point anywhere speaks for every voxel like it.",
        min: 0.1,
        max: 10000,
        step: 1,
        when: basic,
      },
      {
        kind: "number",
        key: "priorLogit",
        label: "a₀, prior logit",
        hint: "Added to every voxel's logit: below 0 asks for more evidence before a voxel is inside. A voxel like no point is background whatever this is.",
        min: -20,
        max: 20,
        step: 0.5,
        when: basic,
      },
      {
        kind: "number",
        key: "unaryScale",
        label: "β, evidence scale",
        hint: "What the points' evidence is multiplied by, against the coupling. 0 leaves only a₀.",
        min: 0,
        max: 20,
        step: 0.1,
        when: basic,
      },
      {
        kind: "number",
        key: "unaryShift",
        label: "Shift",
        hint: "Added to every unary logit the network gives, to move its decision boundary: above 0 segments more, below 0 less.",
        min: -20,
        max: 20,
        step: 0.5,
        when: learned,
      },
    ],
  },
  {
    title: "Pairwise potential",
    note: "w: the cost of a label change between neighbours",
    fields: [
      {
        kind: "number",
        key: "coupling",
        label: "λ, coupling",
        hint:
          "Basic: the coupling of an edge with no contrast across it. Learned: what the network's couplings are multiplied by (1 is as trained). " +
          "0 turns the pairwise term off, and the probability is then the unary's sigmoid.",
        min: 0,
        max: 100,
        step: 0.1,
      },
      {
        kind: "number",
        key: "contrast",
        label: "δ, contrast HU",
        hint: "An edge whose two voxels differ by this much in HU keeps 61 % of the coupling; by three times this, 1 %. Smaller lets the label change at fainter edges.",
        min: 1,
        max: 10000,
        step: 10,
        when: basic,
      },
      {
        kind: "number",
        key: "couplingFloor",
        label: "Floor",
        hint: "The share of the coupling every edge keeps whatever its contrast, for smoothness in flat regions.",
        min: 0,
        max: 1,
        step: 0.05,
        when: basic,
      },
    ],
  },
  {
    title: "Inference",
    note: "mean field: mᵢ = σ(aᵢ + Σⱼ wᵢⱼ (2 mⱼ − 1))",
    fields: [
      {
        kind: "number",
        key: "sweeps",
        label: "Sweeps",
        hint:
          "Mean-field sweeps over the checkerboard. Information travels one voxel per half-sweep, so this bounds how far a point reaches through the coupling. " +
          "Stops early once a sweep changes no probability by more than 0.001.",
        min: 0,
        max: 2000,
        step: 10,
        integer: true,
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
