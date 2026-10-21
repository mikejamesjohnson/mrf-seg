"use client";

/** The pieces the side panels are built from, so every panel reads and behaves the same. */

import { Label } from "@/components/ui/label";
import { Slider } from "@/components/ui/slider";
import { Switch } from "@/components/ui/switch";
import { hexRgb, rgbCss, rgbHex, type Rgb } from "@/lib/color";
import { useSceneStore } from "@/lib/scene/store";
import type { Interpolation, SceneNode, SceneNodeType } from "@/lib/scene/types";
import { productStale, type ProductKey } from "@/lib/segmentation/state";
import { cn } from "@/lib/utils";

type Children = { children: React.ReactNode };

/** The node of `type`, if its dataset is there: how a panel finds what it adjusts. */
export function useNode<T extends SceneNodeType>(type: T): Extract<SceneNode, { type: T }> | undefined {
  return useSceneStore((s) => Object.values(s.nodes).find((node): node is Extract<SceneNode, { type: T }> => node.type === type));
}

/** Small print under a control. */
export function Hint({ children }: Children) {
  return <p className="text-[11px] leading-4 text-muted-foreground">{children}</p>;
}

export function ErrorText({ children }: Children) {
  return <p className="text-[11px] leading-4 break-words text-destructive">{children}</p>;
}

/** Something on screen no longer matches what it was made from. */
export function StaleText({ children }: Children) {
  return <p className="text-[11px] leading-4 text-amber-500">{children}</p>;
}

export function ProgressBar({ value }: { value: number }) {
  return (
    <div className="h-1 overflow-hidden rounded-full bg-muted">
      <div className="h-full bg-primary transition-[width]" style={{ width: `${Math.round(value * 100)}%` }} />
    </div>
  );
}

/** A label on the left and its control on the right; `hint` shows on hover. */
export function Row({ label, hint, children }: Children & { label: React.ReactNode; hint?: string }) {
  return (
    <div className="flex items-center justify-between gap-2" title={hint}>
      <span className="text-[11px] text-foreground/80">{label}</span>
      {children}
    </div>
  );
}

export function SwitchRow({ id, label, checked, onChange }: { id: string; label: React.ReactNode; checked: boolean; onChange: (checked: boolean) => void }) {
  return (
    <div className="flex items-center justify-between gap-2">
      <Label htmlFor={id} className="text-xs">
        {label}
      </Label>
      <Switch id={id} size="sm" checked={checked} onCheckedChange={onChange} />
    </div>
  );
}


/**
 * A short label and a slider over [min, max]; `name` is what it is called to assistive tech,
 * and `format`, if given, shows the value after it.
 */
export function SliderRow({
  label,
  name,
  value,
  onChange,
  min = 0,
  max = 1,
  step = 0.05,
  disabled,
  format,
}: {
  label: string;
  name?: string;
  value: number;
  onChange: (value: number) => void;
  min?: number;
  max?: number;
  step?: number;
  disabled?: boolean;
  format?: (value: number) => string;
}) {
  return (
    <div className={cn("flex items-center gap-2", disabled && "opacity-50")}>
      <span className="w-12 shrink-0 text-[11px] text-muted-foreground">{label}</span>
      <Slider
        min={min}
        max={max}
        step={step}
        value={[value]}
        disabled={disabled}
        onValueChange={(next) => onChange(typeof next === "number" ? next : next[0])}
        aria-label={name ?? label}
      />
      {format && <span className="w-9 shrink-0 text-right font-mono text-[10px] text-muted-foreground">{format(value)}</span>}
    </div>
  );
}

/** Colours to pick from: red first, the surface's amber second. */
const SWATCHES: Rgb[] = [
  [1, 0, 0],
  [0.95, 0.75, 0.3],
  [0.3, 0.8, 0.35],
  [0.25, 0.5, 1],
  [0.3, 0.8, 0.85],
  [0.9, 0.9, 0.9],
];

/** A colour: a row of swatches, then any colour. */
export function ColorRow({ value, onChange, name }: { value: Rgb; onChange: (rgb: Rgb) => void; name: string }) {
  const current = rgbHex(value);
  return (
    <div className="flex items-center gap-1.5">
      <span className="w-12 shrink-0 text-[11px] text-muted-foreground">Colour</span>
      {SWATCHES.map((rgb) => (
        <button
          key={rgbHex(rgb)}
          type="button"
          title={rgbHex(rgb)}
          aria-label={`${name} colour ${rgbHex(rgb)}`}
          onClick={() => onChange(rgb)}
          className={cn("size-4 rounded-full border", rgbHex(rgb) === current ? "border-foreground ring-1 ring-foreground" : "border-border")}
          style={{ backgroundColor: rgbCss(rgb) }}
        />
      ))}
      <input
        type="color"
        value={current}
        onChange={(event) => onChange(hexRgb(event.target.value))}
        className="h-5 w-6 cursor-pointer rounded border border-border bg-transparent"
        aria-label={`${name} colour`}
      />
    </div>
  );
}

/** How a product's last build went: its error, or what was built and whether it is out of date. */
export function ProductStatus({ product, children }: Children & { product: ProductKey }) {
  const status = useSceneStore((s) => s[product].status);
  const built = useSceneStore((s) => s[product].built !== null);
  const stale = useSceneStore((s) => productStale(s, product));
  if (status.state === "error") return <ErrorText>{status.message}</ErrorText>;
  if (!built) return null;
  return (
    <div>
      <Hint>{children}</Hint>
      {stale && <StaleText>Built from an earlier segmentation or other settings: build again to update.</StaleText>}
    </div>
  );
}

export interface ChoiceOption<T extends string> {
  value: T;
  label: string;
  title?: string;
  icon?: React.ReactNode;
}

/**
 * One of a few options, as a row of buttons: `inline` and small for the right side of a `Row`,
 * or across the panel's width.
 */
export function ChoiceButtons<T extends string>({
  value,
  options,
  onChange,
  label,
  inline = false,
  disabled,
}: {
  value: T;
  options: ChoiceOption<T>[];
  onChange: (value: T) => void;
  label: string;
  inline?: boolean;
  disabled?: boolean;
}) {
  return (
    <div
      className={cn(inline ? "flex shrink-0 gap-px rounded-md bg-muted/40 p-0.5" : "grid gap-1 rounded-lg bg-muted/40 p-0.5", disabled && "opacity-50")}
      style={inline ? undefined : { gridTemplateColumns: `repeat(${options.length}, minmax(0, 1fr))` }}
      role="group"
      aria-label={label}
    >
      {options.map((option) => (
        <button
          key={option.value}
          type="button"
          aria-pressed={value === option.value}
          title={option.title}
          disabled={disabled}
          onClick={() => onChange(option.value)}
          className={cn(
            "transition-colors",
            inline ? "h-5 rounded px-1.5 text-[10px]" : "flex h-7 items-center justify-center gap-1.5 rounded-md text-xs",
            value === option.value ? "bg-background text-foreground shadow-sm" : "text-muted-foreground hover:text-foreground",
          )}
        >
          {option.icon}
          {option.label}
        </button>
      ))}
    </div>
  );
}

const SAMPLING: ChoiceOption<Interpolation>[] = [
  { value: "trilinear", label: "Trilinear", title: "Blend between neighbouring voxels" },
  { value: "nearest", label: "Nearest", title: "Each voxel as the block it is" },
];

/** How an image is sampled where a plane cuts it. */
export function SamplingRow({ name, value, onChange }: { name: string; value: Interpolation; onChange: (value: Interpolation) => void }) {
  return (
    <Row label="Sampling">
      <ChoiceButtons inline label={`${name} sampling`} value={value} options={SAMPLING} onChange={onChange} />
    </Row>
  );
}
