/**
 * The MRF settings, the extractors and the maps are each written down twice, here and in the
 * backend: these read the Python source and check the two agree.
 */

import { readFileSync } from "node:fs";

import { describe, expect, it } from "vitest";

import { toCamelCase, toSnakeCase } from "@/lib/segmentation/backend";
import { MAP_DISPLAY, MAP_QUANTITIES } from "@/lib/segmentation/maps";
import {
  DEFAULT_MRF_SETTINGS,
  EXTRACTOR_LABELS,
  fieldKeys,
  IMAGE_DERIVED,
  needsInsidePoint,
  POTENTIAL_KEYS,
  SETTING_GROUPS,
  settingGroup,
  settingsForExtractor,
  shownFields,
  type MRFSettings,
} from "@/lib/segmentation/settings";

const python = (file: string) => readFileSync(new URL(`../python/mrfseg/${file}`, import.meta.url), "utf8");

const words: Record<string, unknown> = { True: true, False: false, None: null };
const literal = (text: string) => (text in words ? words[text] : /^["']/.test(text) ? text.slice(1, -1) : Number(text));

/** `MRFSettings`'s fields and defaults, read from `settings.py`. */
function backendDefaults(): Record<string, unknown> {
  const body = python("settings.py").split(/^class MRFSettings\b.*$/m)[1].split(/^\s+def /m)[0];
  return Object.fromEntries([...body.matchAll(/^ {4}(\w+): [^=\n]+ = (.+)$/gm)].map(([, name, value]) => [name, literal(value.trim())]));
}

/** A tuple of names at module level in a Python file, e.g. `IMAGE_DERIVED = ("a", "b")`. */
function backendTuple(file: string, name: string): string[] {
  const match = new RegExp(`^${name}(?:: [^=]+)? = \\(([^)]*)\\)`, "m").exec(python(file));
  if (!match) throw new Error(`${name} not found in ${file}`);
  return [...match[1].matchAll(/"(\w+)"/g)].map(([, word]) => word);
}

/** `BOUNDS` from `settings.py`: each setting's (low, high), with `MAX_SCALES` read from `basic.py`. */
function backendBounds(): Record<string, [number, number]> {
  const constants: Record<string, number> = { MAX_SCALES: Number(/^MAX_SCALES = (\d+)$/m.exec(python("basic.py"))?.[1]) };
  const body = python("settings.py").split(/^BOUNDS[^\n]*\n/m)[1].split(/^}/m)[0];
  const number = (text: string) => (text in constants ? constants[text] : Number(text));
  return Object.fromEntries([...body.matchAll(/^ {4}"(\w+)": \(([^,]+), ([^)]+)\),$/gm)].map(([, name, low, high]) => [name, [number(low), number(high)]]));
}

describe("MRF settings", () => {
  it("have the backend's names and defaults", () => {
    expect(toSnakeCase(DEFAULT_MRF_SETTINGS)).toEqual(backendDefaults());
  });

  it("convert to the backend's names and back", () => {
    expect(toCamelCase(toSnakeCase(DEFAULT_MRF_SETTINGS))).toEqual(DEFAULT_MRF_SETTINGS);
  });

  it("each have exactly one control in the panel", () => {
    const keys = SETTING_GROUPS.flatMap((group) => group.fields.flatMap(fieldKeys));
    expect([...keys].sort()).toEqual(Object.keys(DEFAULT_MRF_SETTINGS).sort());
  });

  it("allow each number what the backend allows", () => {
    const bounds = backendBounds();
    const numbers = SETTING_GROUPS.flatMap((group) => group.fields).flatMap((field) => (field.kind === "number" ? [field] : []));
    expect(Object.fromEntries(numbers.map((field) => [Object.keys(toSnakeCase({ [field.key]: 0 }))[0], [field.min, field.max]]))).toEqual(bounds);
    for (const field of numbers) {
      const value = DEFAULT_MRF_SETTINGS[field.key];
      if (value === null) continue; // the image chooses it
      expect(value, field.key).toBeGreaterThanOrEqual(field.min);
      expect(value, field.key).toBeLessThanOrEqual(field.max);
    }
  });

  it("group the potentials and the inference as the backend's model does", () => {
    expect(settingGroup("Unary potential").fields.map((field) => field.key)).toEqual(["featureLengthscale", "spatialLengthscale", "priorLogit", "unaryScale", "unaryShift"]);
    expect(settingGroup("Pairwise potential").fields.map((field) => field.key)).toEqual(["coupling", "contrast", "couplingFloor"]);
    expect(settingGroup("Inference").fields.map((field) => field.key)).toEqual(["sweeps"]);
    const potentials = [...settingGroup("Unary potential").fields, ...settingGroup("Pairwise potential").fields, ...settingGroup("Inference").fields].map((field) => field.key);
    expect([...POTENTIAL_KEYS].sort()).toEqual([...potentials].sort());
  });

  it("start from an extractor's own settings over the defaults, or from the defaults alone", () => {
    const trained = { available: true, settings: { sweeps: 20, coupling: 1, unaryShift: 0 } as Partial<MRFSettings>, nativeSpacing: 0.5, note: "/x/best.pt" };
    const defaults = Object.fromEntries(POTENTIAL_KEYS.map((key) => [key, DEFAULT_MRF_SETTINGS[key]]));
    expect(settingsForExtractor("learned", trained)).toEqual({ featureExtractor: "learned", featureSpacing: null, ...defaults, sweeps: 20, coupling: 1, unaryShift: 0 });
    expect(settingsForExtractor("basic", { available: true, settings: null, nativeSpacing: null, note: null })).toEqual({ featureExtractor: "basic", featureSpacing: null, ...defaults });
    expect(settingsForExtractor("basic", undefined)).toEqual({ featureExtractor: "basic", featureSpacing: null, ...defaults });
  });

  it("leave to the image what the backend leaves to it", () => {
    expect(Object.keys(toSnakeCase(Object.fromEntries(IMAGE_DERIVED.map((key) => [key, null]))))).toEqual(backendTuple("settings.py", "IMAGE_DERIVED"));
    for (const key of IMAGE_DERIVED) expect(DEFAULT_MRF_SETTINGS[key], key).toBeNull();
  });

  it("offer the backend's extractors, and show an extractor's options only while it is chosen", () => {
    const names = Object.keys(EXTRACTOR_LABELS).sort();
    const backend = [...python("extractors.py").matchAll(/^ {4}"(\w+)": ExtractorSpec\(/gm)].map(([, name]) => name);
    expect(names).toEqual([...backend].sort());
    const features = settingGroup("Feature extraction");
    const extractor = features.fields.find((field) => field.key === "featureExtractor");
    expect(extractor?.kind === "choice" && extractor.options.map((option) => option.value).sort()).toEqual(names);
    const basicOnly = features.fields.filter((field) => field.key.startsWith("basic")).map((field) => field.key);
    expect(basicOnly.length).toBeGreaterThan(0);
    const shown = (group: string, featureExtractor: "basic" | "learned") => shownFields(settingGroup(group), { ...DEFAULT_MRF_SETTINGS, featureExtractor }).map((field) => field.key);
    expect(shown("Feature extraction", "basic")).toEqual(expect.arrayContaining(basicOnly));
    expect(shown("Feature extraction", "learned").filter((key) => basicOnly.includes(key))).toEqual([]);
    // The handcrafted unary's knobs for Basic, the network's shift for Learned; the coupling for both.
    expect(shown("Unary potential", "basic")).toEqual(["featureLengthscale", "spatialLengthscale", "priorLogit", "unaryScale"]);
    expect(shown("Unary potential", "learned")).toEqual(["unaryShift"]);
    expect(shown("Pairwise potential", "basic")).toEqual(["coupling", "contrast", "couplingFloor"]);
    expect(shown("Pairwise potential", "learned")).toEqual(["coupling"]);
  });

  it("know which extractor needs an inside point, as the backend's table says", () => {
    const table = python("extractors.py");
    const learnedEntry = table.split(/^ {4}"learned": ExtractorSpec\(/m)[1].split(/^ {4}\),/m)[0];
    expect(learnedEntry).toContain("needs_inside_point=False");
    expect(needsInsidePoint("learned")).toBe(false);
    expect(needsInsidePoint("basic")).toBe(true);
  });
});

describe("maps", () => {
  it("are the backend's, in its order, with its compact types", () => {
    const encoding = python("server/encoding.py");
    const entries = [...encoding.matchAll(/^ {4}"(\w+)": MapEncoding\(np\.(\w+), /gm)].map(([, name, dtype]) => [name, dtype]);
    expect(entries.map(([name]) => name)).toEqual([...MAP_QUANTITIES]);
    const typeNames = { Int16Array: "int16", Uint8Array: "uint8" } as const;
    expect(entries).toEqual(MAP_QUANTITIES.map((name) => [name, typeNames[MAP_DISPLAY[name].compact.name as keyof typeof typeNames]]));
  });

  it("are the fields of the backend's Segmentation", () => {
    const body = python("segmentation.py").split(/^class Segmentation\b.*$/m)[1].split(/^class /m)[0];
    const fields = [...body.matchAll(/^ {4}(\w+): vtkImageData$/gm)].map(([, name]) => name);
    expect([...fields].sort()).toEqual([...MAP_QUANTITIES].sort());
  });
});
