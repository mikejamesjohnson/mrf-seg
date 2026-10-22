"use client";

import { AlertCircle, Loader2 } from "lucide-react";

import { ErrorText, Hint, ProgressBar, Row } from "@/components/panels/controls";
import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { NumberField } from "@/components/ui/number-field";
import { currentImage } from "@/lib/scene/datasets";
import { useSceneStore } from "@/lib/scene/store";
import type { ImageDataset } from "@/lib/scene/types";
import { chooseExtractor, resetPotentials } from "@/lib/segmentation/extractors";
import { connectSegmentation } from "@/lib/segmentation/session";
import { EXTRACTOR_LABELS, fieldKeys, settingGroup, shownFields, type FeatureExtractorName, type MRFSettings, type SettingField } from "@/lib/segmentation/settings";

const percent = (fraction: number) => `${Math.round(fraction * 100)}%`;

/**
 * How the segmentation is set up, in the side panel's sections: the affinities (from the
 * image's contrast or the network, on what grid), the seeds, and the solver, each a table of
 * settings. Not disabled while a Generate runs: it took its own copy of the settings, and a
 * change made meanwhile marks it out of date.
 */

/** Affinities: the backend's session and its features, then the extractor and its options. */
export function FeaturesPanel() {
  const image = useSceneStore((s) => currentImage(s));
  const settings = useSceneStore((s) => s.segmentation.settings);
  const extractors = useSceneStore((s) => s.extractors);
  const defaultsFrom = useSceneStore((s) => s.segmentation.defaultsFrom);
  const info = extractors[settings.featureExtractor];
  return (
    <div className="space-y-3">
      {image && <SessionStatus image={image} />}
      <p className="font-mono text-[10px] text-muted-foreground">{settingGroup("Affinities").note}</p>
      <SettingRows group="Affinities" />
      {info && !info.available && <ErrorText>{info.note}</ErrorText>}
      {info?.available && info.note && settings.featureExtractor === "learned" && (
        <Hint>
          Checkpoint: <span className="break-all">{info.note}</span>
        </Hint>
      )}
      <div className="flex items-start justify-between gap-2">
        <Hint>
          {settings.featureExtractor === "basic"
            ? "Grady's random walker: an edge's affinity falls with the HU step across it. A point seed in a vessel whose walls are ramps of a few voxels leaks within millimetres; the baseline the network is measured against."
            : "The network's log-affinity on each edge, learned so that a point carries through the vessel it is in and no further: high along the lumen, far below zero across its wall, and the background a conductor to the seeded fat."}
          {defaultsFrom === "learned" && settings.featureExtractor === "learned" && " Loaded from the checkpoint: the settings it was trained with."}
        </Hint>
        <Button size="xs" variant="ghost" className="shrink-0 text-muted-foreground" onClick={resetPotentials} title="Back to the current extractor's affinity, seed and solver settings">
          Reset
        </Button>
      </div>
    </div>
  );
}

/** Whether the MRFSegmentation exists and has its features, or what went wrong, with a way to open it again. */
function SessionStatus({ image }: { image: ImageDataset }) {
  const { status, sessionId, features, job, settings } = useSceneStore((s) => s.segmentation);
  const line = "flex items-center gap-1.5 text-[11px] text-muted-foreground";

  if (status.state === "error" && job.state !== "running") {
    return (
      <div className="space-y-1.5">
        <p className="flex gap-1.5 text-[11px] leading-4 text-destructive">
          <AlertCircle className="mt-px size-3 shrink-0" />
          <span className="min-w-0 break-words">{status.message}</span>
        </p>
        <Button size="xs" variant="outline" onClick={() => void connectSegmentation(image)}>
          Retry
        </Button>
      </div>
    );
  }
  if (!sessionId || features.state === "running") {
    return (
      <div className="space-y-1">
        <p className={line}>
          <Loader2 className="size-3 shrink-0 animate-spin" />
          {sessionId ? `Extracting ${EXTRACTOR_LABELS[settings.featureExtractor]} features… ${percent(features.progress)}` : status.state === "running" ? status.message : "Starting MRFSegmentation…"}
        </p>
        {sessionId && <ProgressBar value={features.progress} />}
      </div>
    );
  }
  return (
    <p className={line}>
      <span className="size-1.5 rounded-full bg-emerald-500" />
      MRFSegmentation ready
      {features.dims && features.extractor && (
        <span className="font-mono text-[10px]">
          · {EXTRACTOR_LABELS[features.extractor]}: {features.channels} features on {features.dims.join("×")}
          {features.spacing !== undefined && ` at ${Number(features.spacing.toPrecision(3))} mm`} ({features.seconds?.toFixed(1)} s)
        </span>
      )}
    </p>
  );
}

/** The rows of one setting group, for the current settings. */
function SettingRows({ group }: { group: string }) {
  const settings = useSceneStore((s) => s.segmentation.settings);
  return (
    <div className="space-y-1.5">
      {shownFields(settingGroup(group), settings).map((field) => (
        <Row key={fieldKeys(field).join()} label={field.label} hint={field.hint}>
          <SettingControl field={field} settings={settings} />
        </Row>
      ))}
    </div>
  );
}

/** One setting's control: a number, a dropdown or a box. The extractor dropdown loads that extractor's settings too. */
function SettingControl({ field, settings }: { field: SettingField; settings: MRFSettings }): React.ReactNode {
  const extractors = useSceneStore((s) => s.extractors);
  const set = (patch: Partial<MRFSettings>) => useSceneStore.getState().setSegmentationSettings(patch);
  switch (field.kind) {
    case "number":
      return (
        <NumberField
          value={settings[field.key]}
          min={field.min}
          max={field.max}
          step={field.step}
          integer={field.integer}
          onChange={(value) => set({ [field.key]: value })}
          placeholder="image"
          aria-label={field.label}
        />
      );
    case "choice":
      return (
        <select
          className="h-6 max-w-44 rounded-md border border-input bg-transparent px-1 text-xs"
          value={settings[field.key]}
          onChange={(event) => {
            const value = event.target.value;
            if (field.key === "featureExtractor") chooseExtractor(value as FeatureExtractorName);
            else set({ [field.key]: value } as Partial<MRFSettings>);
          }}
          aria-label={field.label}
        >
          {field.options.map((option) => {
            const unavailable = field.key === "featureExtractor" && extractors[option.value as FeatureExtractorName]?.available === false;
            return (
              <option key={option.value} value={option.value} disabled={unavailable}>
                {option.label}
                {unavailable && " (no checkpoint)"}
              </option>
            );
          })}
        </select>
      );
    case "check":
      return <Checkbox checked={settings[field.key]} onCheckedChange={(checked) => set({ [field.key]: checked })} aria-label={field.label} />;
    default: {
      const unknown: never = field;
      throw new Error(`Unknown setting kind: ${JSON.stringify(unknown)}`);
    }
  }
}

/** A section of the walker's settings: the group's formula, its rows, and what they mean. */
function PotentialSection({ group, children }: { group: string; children?: React.ReactNode }) {
  const note = settingGroup(group).note;
  return (
    <div className="space-y-2">
      <p className="font-mono text-[10px] text-muted-foreground">{note}</p>
      <SettingRows group={group} />
      {children}
    </div>
  );
}

/** The seeds: the points, held at their labels, and the background below a threshold held at 0. */
export function SeedsPanel() {
  const generated = useSceneStore((s) => s.segmentation.generated);
  return (
    <PotentialSection group="Seeds">
      <Hint>
        An inside point is held at 1, an outside point at 0, exactly: the walker conditions on them rather than being told about them, so a corrected voxel
        flips, and another inside point can only raise the probability, another outside point only lower it.
        {generated && (
          <>
            {" "}
            Last Generate: {generated.inference.seeds.inside} inside, {generated.inference.seeds.outside} outside
            {generated.inference.seeds.background > 0 && `, ${generated.inference.seeds.background.toLocaleString()} background`} seeds.
          </>
        )}
      </Hint>
    </PotentialSection>
  );
}

/** The solver: conjugate gradients on the walker's system, preconditioned by a multigrid cycle. */
export function SolverPanel() {
  const generated = useSceneStore((s) => s.segmentation.generated);
  return (
    <PotentialSection group="Solver">
      <Hint>
        Every free voxel ends up at the affinity-weighted mean of its neighbours, the seeds held: the probability that a walker from there reaches an inside
        point before an outside one. No piece of the result can be without an inside point.
        {generated && (
          <>
            {" "}
            Last Generate: {generated.inference.iterations} iterations, residual {generated.inference.residual.toExponential(1)}
            {generated.inference.converged ? ", converged" : ", not converged (allow more iterations)"}; {generated.inference.components}{" "}
            {generated.inference.components === 1 ? "piece" : "pieces"}.
          </>
        )}
      </Hint>
    </PotentialSection>
  );
}
