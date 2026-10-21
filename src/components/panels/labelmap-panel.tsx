"use client";

import { Grid3x3 } from "lucide-react";

import { StepButton } from "@/components/panels/auto-update-panel";
import { ColorRow, Hint, ProductStatus, Row, SamplingRow, SliderRow, SwitchRow, useNode } from "@/components/panels/controls";
import { NumberField } from "@/components/ui/number-field";
import { currentImage } from "@/lib/scene/datasets";
import { useSceneStore } from "@/lib/scene/store";
import type { LabelmapNode } from "@/lib/scene/types";
import { buildSegmentationLabelmap } from "@/lib/segmentation/products";
import { canBuild, isBusy } from "@/lib/segmentation/state";

/** The labelmap step of the Generate section: p(y = 1) thresholded per voxel, built in the backend, and how it is drawn over the image. */
export function LabelmapProduct() {
  const labelmap = useSceneStore((s) => s.labelmap);
  const busy = useSceneStore(isBusy);
  const ready = useSceneStore(canBuild);
  const voxelMm3 = useSceneStore((s) => {
    const spacing = currentImage(s)?.spacing;
    return spacing ? spacing[0] * spacing[1] * spacing[2] : 0;
  });
  const node = useNode("labelmap");
  const { built } = labelmap;

  return (
    <div className="space-y-2">
      <StepButton step="labelmap" icon={<Grid3x3 />} running={labelmap.status.state === "running"} disabled={busy || !ready} onClick={() => void buildSegmentationLabelmap()}>
        Generate labelmap
      </StepButton>
      <Row label="Threshold p" hint="Voxels with p(y = 1) at least this are labelled 1, the rest 0">
        <NumberField
          value={labelmap.threshold}
          min={0.01}
          max={0.99}
          step={0.05}
          onChange={(threshold) => useSceneStore.getState().setProduct("labelmap", { threshold })}
          aria-label="Labelmap threshold"
        />
      </Row>
      <SwitchRow
        id="labelmap-connected"
        label="Connected to the inside points"
        checked={labelmap.connected}
        onChange={(connected) => useSceneStore.getState().setProduct("labelmap", { connected })}
      />
      <Hint>
        The CRF cannot promise one connected vessel tree: islands of lumen-like voxels are labelled too. Connected keeps only the components an inside point
        touches (the largest, with no inside point).
      </Hint>
      <ProductStatus product="labelmap">
        {built &&
          `${built.voxels.toLocaleString()} voxels (${((built.voxels * voxelMm3) / 1000).toFixed(1)} mL) with p ≥ ${built.threshold}${built.connected ? ", connected" : ""}, ${built.seconds.toFixed(2)} s.`}
      </ProductStatus>
      {node && <Appearance node={node} />}
    </div>
  );
}

function Appearance({ node }: { node: LabelmapNode }) {
  const update = (patch: Partial<LabelmapNode>) => useSceneStore.getState().updateNode(node.id, patch);
  return (
    <div className="space-y-2 pl-1">
      <SwitchRow id="labelmap-visible" label="Overlay on the image" checked={node.visible} onChange={(visible) => update({ visible })} />
      <ColorRow name="Labelmap" value={node.color} onChange={(color) => update({ color })} />
      <SliderRow label="Opacity" name="Labelmap opacity" value={node.opacity} onChange={(opacity) => update({ opacity })} />
      <SamplingRow name="Labelmap" value={node.interpolation} onChange={(interpolation) => update({ interpolation })} />
    </div>
  );
}
