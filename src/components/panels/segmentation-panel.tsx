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
 * How the segmentation is set up, in the side panel's sections: the features (which extractor,
 * on what grid), then the CRF's unary and pairwise potentials and its inference, each a table
 * of settings. Not disabled while a Generate runs: it took its own copy of the settings, and a
 * change made meanwhile marks it out of date.
 */

/** Feature extraction: the backend's session and its features, then the extractor and its options. */
export function FeaturesPanel() {
  const image = useSceneStore((s) => currentImage(s));
  const settings = useSceneStore((s) => s.segmentation.settings);
  const extractors = useSceneStore((s) => s.extractors);
  const info = extractors[settings.featureExtractor];
  return (
    <div className="space-y-3">
      {image && <SessionStatus image={image} />}
      <SettingRows group="Feature extraction" />
      {info && !info.available && <ErrorText>{info.note}</ErrorText>}
      {info?.available && info.note && settings.featureExtractor === "learned" && (
        <Hint>
          Checkpoint: <span className="break-all">{info.note}</span>
        </Hint>
      )}
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

/** A section of the CRF's settings: the group's formula, its rows, and what they mean. */
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

/** The unary potential: the evidence for lumen at each voxel, from the points (Basic) or the network (Learned). */
export function UnaryPanel() {
  const extractor = useSceneStore((s) => s.segmentation.settings.featureExtractor);
  const defaultsFrom = useSceneStore((s) => s.segmentation.defaultsFrom);
  return (
    <PotentialSection group="Unary potential">
      <div className="flex items-start justify-between gap-2">
        <Hint>
          {extractor === "basic"
            ? "Each voxel is compared with the inside points (what lumen looks like here) against the image as a whole and the outside points (what it does not): a kernel density ratio. A voxel like no point is background."
            : "The network's unary logit at each voxel. It sees the points: an inside point selects the vessel it is in, an outside point drops the structure it is in, and with no points it segments every coronary. Generate re-runs it in windows around the points."}
          {defaultsFrom === "learned" && extractor === "learned" && " Loaded from the checkpoint: the settings it was trained with."}
        </Hint>
        <Button size="xs" variant="ghost" className="shrink-0 text-muted-foreground" onClick={resetPotentials} title="Back to the current extractor's potential and inference settings">
          Reset
        </Button>
      </div>
    </PotentialSection>
  );
}

/** The pairwise potential: how much a label change between two neighbours costs. */
export function PairwisePanel() {
  const extractor = useSceneStore((s) => s.segmentation.settings.featureExtractor);
  return (
    <PotentialSection group="Pairwise potential">
      <Hint>
        {extractor === "basic"
          ? "Contrast-sensitive: a label may change cheaply where the image does, so the boundary snaps to the vessel wall. The same cost in every direction."
          : "The network's coupling on each edge, learned: cheap across a wall, expensive along a vessel."}
      </Hint>
    </PotentialSection>
  );
}

/** The inference: mean field over the grid, with the points clamped. */
export function InferencePanel() {
  const generated = useSceneStore((s) => s.segmentation.generated);
  return (
    <PotentialSection group="Inference">
      <Hint>
        The points are clamped to their labels; every other voxel takes the probability its evidence and its neighbours give it, sweep after sweep. The
        probability map is this marginal; the labelmap thresholds it.
        {generated && (
          <>
            {" "}
            Last Generate: {generated.inference.sweeps} sweeps, last change {generated.inference.delta.toExponential(1)}
            {generated.inference.converged ? ", converged" : ", not yet converged (more sweeps would change it)"}
            {generated.inference.components !== null && `; kept ${generated.inference.components} connected ${generated.inference.components === 1 ? "component" : "components"}`}.
          </>
        )}
      </Hint>
    </PotentialSection>
  );
}
