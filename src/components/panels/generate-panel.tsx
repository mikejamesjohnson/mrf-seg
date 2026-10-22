"use client";

import { Play } from "lucide-react";
import { useShallow } from "zustand/react/shallow";

import { StepButton } from "@/components/panels/auto-update-panel";
import { ErrorText, Hint, ProgressBar, SamplingRow, SliderRow, StaleText, SwitchRow } from "@/components/panels/controls";
import { LabelmapProduct } from "@/components/panels/labelmap-panel";
import { SurfaceProduct } from "@/components/panels/surface-panel";
import { currentImage } from "@/lib/scene/datasets";
import { useSceneStore } from "@/lib/scene/store";
import type { OverlayNode } from "@/lib/scene/types";
import { generateSegmentation } from "@/lib/segmentation/generate";
import { MAP_DISPLAY } from "@/lib/segmentation/maps";
import { enoughPoints, isBusy, staleReasons, type TaskProgress, type TaskStage } from "@/lib/segmentation/state";
import { OVERLAY_RAMPS, rampGradient } from "@/lib/viewer/colorRamp";

const STAGES: Record<TaskStage, string> = {
  queued: "Waiting",
  features: "Extracting features",
  affinities: "Building the affinities",
  seeds: "Seeding",
  solve: "Solving the walker",
  maps: "Fetching the maps",
};

const percent = (fraction: number) => `${Math.round(fraction * 100)}%`;

/** Where a task has got to, for the line under its button. */
export function stageText(task: TaskProgress, features: TaskProgress): string {
  if (task.stage === "queued" && features.state === "running") return `Waiting for features… ${percent(features.progress)}`;
  return `${(task.stage && STAGES[task.stage]) ?? "Working"}… ${percent(task.progress)}`;
}

/**
 * The pipeline's three outputs, each a button with an Auto box: the segmentation (the walker's
 * probability at every voxel, with the settings as they are), and the labelmap and surface built
 * from it. Each step's own settings and display options sit under its button.
 */
export function GeneratePanel() {
  const image = useSceneStore((s) => currentImage(s));
  if (!image) return <Hint>Open an image to segment it.</Hint>;
  return (
    <div className="space-y-4">
      <Segmentation />
      <div className="border-t border-border/40 pt-3">
        <LabelmapProduct />
      </div>
      <div className="border-t border-border/40 pt-3">
        <SurfaceProduct />
      </div>
    </div>
  );
}

/** The Generate segmentation step: the button, its progress, what the last one found, and the maps it drew. */
function Segmentation() {
  const { job, features, generated } = useSceneStore((s) => s.segmentation);
  const reasons = useSceneStore(useShallow(staleReasons));
  const ready = useSceneStore(enoughPoints);
  const busy = useSceneStore(isBusy);
  const running = job.state === "running";

  return (
    <div className="space-y-2">
      <StepButton step="segmentation" icon={<Play />} running={running} disabled={busy || !ready} onClick={() => void generateSegmentation()} variant="default">
        Generate segmentation
      </StepButton>
      {running ? (
        <div className="space-y-1">
          <Hint>{stageText(job, features)}</Hint>
          <ProgressBar value={job.stage === "queued" ? features.progress : job.progress} />
        </div>
      ) : (
        <>
          {!ready && <Hint>Needs at least one inside point.</Hint>}
          {job.state === "error" && job.message && <ErrorText>{job.message}</ErrorText>}
          {generated && (
            <div className="space-y-0.5">
              <Hint>
                From {generated.points} points with the settings above: solved in {generated.seconds.solve.toFixed(1)} s ({generated.inference.iterations}{" "}
                iterations{generated.inference.converged ? "" : ", not converged"}; {generated.inference.components}{" "}
                {generated.inference.components === 1 ? "piece" : "pieces"} from {generated.inference.seeds.inside} inside{" "}
                {generated.inference.seeds.inside === 1 ? "point" : "points"}
                {generated.inference.seeds.background > 0 && `, ${(generated.inference.seeds.background / 1e6).toFixed(1)} M background seeds`})
                {generated.seconds.features > 0 && `, features ${generated.seconds.features.toFixed(1)} s`}.
              </Hint>
              {generated.inference.components > generated.inference.seeds.inside && (
                <StaleText>More pieces than inside points: the solve did not converge. Allow more iterations, or loosen the tolerance.</StaleText>
              )}
              {reasons.length > 0 && <StaleText>Out of date: {reasons.join("; ")}. Generate to update.</StaleText>}
            </div>
          )}
        </>
      )}
      <Overlays />
    </div>
  );
}

/** One row per map, once there are maps. */
function Overlays() {
  const nodes = useSceneStore(
    useShallow((s) => s.nodeOrder.map((id) => s.nodes[id]).filter((node): node is OverlayNode => node?.type === "overlay")),
  );
  if (nodes.length === 0) return null;
  return (
    <div className="space-y-3 pl-1">
      {nodes.map((node) => (
        <OverlayRow key={node.id} node={node} />
      ))}
    </div>
  );
}

function OverlayRow({ node }: { node: OverlayNode }) {
  const dataset = useSceneStore((s) => s.datasets[node.datasetId]);
  const display = dataset?.kind === "map" ? MAP_DISPLAY[dataset.quantity] : null;
  const label = display?.label ?? node.name;
  const update = (patch: Partial<OverlayNode>) => useSceneStore.getState().updateNode(node.id, patch);
  const ends = display?.legend(node.range) ?? [String(node.range[0]), String(node.range[1])];

  return (
    <div className="space-y-1.5">
      <SwitchRow
        id={`${node.id}-visible`}
        label={
          <>
            {label}
            {display?.meaning && <span className="font-normal text-muted-foreground">{display.meaning}</span>}
          </>
        }
        checked={node.visible}
        onChange={(visible) => update({ visible })}
      />
      <div className="h-1.5 rounded-full" style={{ background: rampGradient(OVERLAY_RAMPS[node.colorMap]) }} />
      <div className="flex justify-between font-mono text-[10px] text-muted-foreground">
        <span>{ends[0]}</span>
        <span>{ends[1]}</span>
      </div>
      <SliderRow label="Opacity" name={`${label} opacity`} value={node.opacity} onChange={(opacity) => update({ opacity })} />
      <SamplingRow name={label} value={node.interpolation} onChange={(interpolation) => update({ interpolation })} />
    </div>
  );
}
