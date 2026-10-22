"use client";

import { Box } from "lucide-react";

import { StepButton } from "@/components/panels/auto-update-panel";
import { ColorRow, Hint, ProductStatus, Row, SliderRow, SwitchRow, useNode } from "@/components/panels/controls";
import { NumberField } from "@/components/ui/number-field";
import { useSceneStore } from "@/lib/scene/store";
import type { SurfaceNode } from "@/lib/scene/types";
import { buildSegmentationSurface } from "@/lib/segmentation/products";
import { canBuild, isBusy } from "@/lib/segmentation/state";

/** The surface step of the Generate section: marching cubes in the backend on the probability, and how the mesh looks. */
export function SurfaceProduct() {
  const surface = useSceneStore((s) => s.surface);
  const busy = useSceneStore(isBusy);
  const ready = useSceneStore(canBuild);
  const node = useNode("surface");
  const set = (patch: Partial<typeof surface>) => useSceneStore.getState().setProduct("surface", patch);
  const { built } = surface;

  return (
    <div className="space-y-2">
      <StepButton step="surface" icon={<Box />} running={surface.status.state === "running"} disabled={busy || !ready} onClick={() => void buildSegmentationSurface()}>
        Generate surface
      </StepButton>
      <Row label="Threshold p" hint="The probability's level set; the labelmap at the same threshold agrees with it, and every piece of either holds an inside point">
        <NumberField value={surface.threshold} min={0.01} max={0.99} step={0.05} onChange={(threshold) => set({ threshold })} aria-label="Probability threshold" />
      </Row>
      <ProductStatus product="surface">{built && `${built.triangles.toLocaleString()} triangles from p = ${built.level}, ${built.seconds.toFixed(2)} s.`}</ProductStatus>
      {node ? <Appearance node={node} /> : <Hint>The mesh is drawn in 3D and as an outline on the slices once built.</Hint>}
    </div>
  );
}

function Appearance({ node }: { node: SurfaceNode }) {
  const update = (patch: Partial<SurfaceNode>) => useSceneStore.getState().updateNode(node.id, patch);
  return (
    <div className="space-y-2 pl-1">
      <SwitchRow id="surface-visible" label="Show in 3D" checked={node.visible} onChange={(visible) => update({ visible })} />
      <ColorRow name="Surface" value={node.color} onChange={(color) => update({ color })} />
      <SliderRow label="Opacity" name="Surface opacity" min={0.05} value={node.opacity} onChange={(opacity) => update({ opacity })} />
      <SwitchRow id="surface-wireframe" label="Wireframe" checked={node.wireframe} onChange={(wireframe) => update({ wireframe })} />
      <SwitchRow id="surface-on-slices" label="Show on slices" checked={node.onSlices} onChange={(onSlices) => update({ onSlices })} />
      <SliderRow
        label="Line"
        name="Slice outline width"
        min={1}
        max={8}
        step={0.5}
        value={node.lineWidth}
        disabled={!node.onSlices}
        format={(width) => `${width} px`}
        onChange={(lineWidth) => update({ lineWidth })}
      />
    </div>
  );
}
