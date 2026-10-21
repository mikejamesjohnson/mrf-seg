"use client";

import { Hint, SamplingRow, SwitchRow } from "@/components/panels/controls";
import { WindowLevel } from "@/components/panels/window-level";
import { currentImage, datasetBytes } from "@/lib/scene/datasets";
import { histogramOf } from "@/lib/scene/histogram";
import { useSceneStore } from "@/lib/scene/store";
import type { ImageDataset, ImageSliceNode, OutlineNode } from "@/lib/scene/types";
import { autoWindowLevel } from "@/lib/scene/windowLevel";
import { GRAYSCALE } from "@/lib/viewer/colorRamp";
import { isIdentityMat3 } from "@/lib/world";

const fmt = (values: readonly number[], digits: number) => values.map((v) => v.toFixed(digits)).join(" × ");

/** What is loaded, and how it is drawn. */
export function ImagePanel() {
  const image = useSceneStore((s) => currentImage(s));
  // Each node by itself, so a change to another (an overlay's opacity) does not re-render this.
  const slices = useSceneStore((s) =>
    Object.values(s.nodes).find((n): n is ImageSliceNode => n.type === "imageSlice" && n.datasetId === image?.id),
  );
  const outline = useSceneStore((s) =>
    Object.values(s.nodes).find((n): n is OutlineNode => n.type === "outline" && n.datasetId === image?.id),
  );
  if (!image) return <Hint>No image loaded.</Hint>;
  const update = (id: string, patch: Partial<ImageSliceNode | OutlineNode>) => useSceneStore.getState().updateNode(id, patch);

  return (
    <div className="space-y-5">
      <ImageInfo image={image} />
      {slices && (
        <WindowLevel
          histogram={histogramOf(image)}
          auto={autoWindowLevel(image, histogramOf(image))}
          ramp={GRAYSCALE}
          window={slices.colorWindow}
          level={slices.colorLevel}
          onChange={(patch) => update(slices.id, patch)}
        />
      )}
      {slices && <SamplingRow name="Image" value={slices.interpolation} onChange={(interpolation) => update(slices.id, { interpolation })} />}
      {outline && (
        <SwitchRow id="show-outline" label="Bounding box in 3D" checked={outline.visible} onChange={(visible) => update(outline.id, { visible })} />
      )}
    </div>
  );
}

function ImageInfo({ image }: { image: ImageDataset }) {
  const rows: [string, React.ReactNode][] = [
    ["File", image.path],
    ["Voxels", fmt(image.dimensions, 0)],
    ["Spacing", `${fmt(image.spacing, 3)} mm`],
    ["Origin", `${fmt(image.gridOrigin, 1)} mm`],
    ["Orientation", isIdentityMat3(image.direction) ? "axis-aligned" : "oblique"],
    ["Values", `${image.scalarType.replace(/Array$/, "")}, ${image.scalarRange[0]} … ${image.scalarRange[1]}`],
    ["Size", `${(datasetBytes(image) / 1e6).toFixed(0)} MB`],
  ];
  return (
    <div className="space-y-0.5">
      <p className="truncate text-sm font-medium" title={image.label}>
        {image.label}
      </p>
      {rows.map(([label, value]) => (
        <div key={label} className="grid grid-cols-[5.5rem_1fr] gap-2 text-[11px] leading-5">
          <span className="text-muted-foreground">{label}</span>
          <span className="min-w-0 font-mono break-all text-foreground/90">{value}</span>
        </div>
      ))}
    </div>
  );
}
