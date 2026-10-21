"use client";

import { ChevronDown } from "lucide-react";

import {
  DropdownMenu,
  DropdownMenuCheckboxItem,
  DropdownMenuContent,
  DropdownMenuGroup,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuRadioGroup,
  DropdownMenuRadioItem,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from "@/components/ui/dropdown-menu";
import { rgbCss } from "@/lib/color";
import { useSceneStore } from "@/lib/scene/store";
import {
  DEFAULT_ORIENTATION,
  ORIENTATION_LABELS,
  ORIENTATIONS,
  orientationOf,
  planePositionLabel,
  ROLE_LABELS,
  scrollHint,
  SLICE_COLORS,
  type Orientation,
  type SliceRole,
} from "@/lib/scene/views";
import { cn } from "@/lib/utils";
import { boundsCenter } from "@/lib/world";

/** A translucent chip over the canvas. */
export const CHIP = "rounded bg-background/70 backdrop-blur-sm";

/** A slice view's name in its colour, what it shows, and its options: orientation, the 3D handle, reset. */
export function SliceViewMenu({ role }: { role: SliceRole }) {
  // Only the orientation, so the menu does not re-render on every scroll notch.
  const orientation = useSceneStore((s) => orientationOf(s.slicePlanes[role]));
  const hint = useSceneStore((s) => scrollHint(s.slicePlanes[role]));
  const reformatting = useSceneStore((s) => s.reformatInThreeD[role]);
  const hasData = useSceneStore((s) => s.worldBounds !== null);
  const store = useSceneStore.getState;

  return (
    <div className="pointer-events-auto flex items-center gap-1.5">
      <DropdownMenu>
        <DropdownMenuTrigger
          className={cn(
            CHIP,
            "flex items-center gap-1.5 px-1.5 py-0.5 text-[10px] font-medium tracking-wide text-foreground/80 outline-none hover:text-foreground focus-visible:ring-2 focus-visible:ring-ring/50",
          )}
          title="Slice view options"
        >
          <span className="size-2 rounded-full" style={{ backgroundColor: rgbCss(SLICE_COLORS[role]) }} />
          <span className="uppercase">{ROLE_LABELS[role]}</span>
          <span className="font-normal text-muted-foreground">{orientation ? ORIENTATION_LABELS[orientation] : "Oblique"}</span>
          <ChevronDown className="size-3 text-muted-foreground" />
        </DropdownMenuTrigger>
        <DropdownMenuContent>
          <DropdownMenuGroup>
            <DropdownMenuLabel>Orientation</DropdownMenuLabel>
            <DropdownMenuRadioGroup value={orientation ?? ""} onValueChange={(value: Orientation) => store().setSliceOrientation(role, value)}>
              {ORIENTATIONS.map((option) => (
                <DropdownMenuRadioItem key={option} value={option} disabled={!hasData}>
                  {ORIENTATION_LABELS[option]}
                </DropdownMenuRadioItem>
              ))}
            </DropdownMenuRadioGroup>
          </DropdownMenuGroup>
          <DropdownMenuSeparator />
          <DropdownMenuCheckboxItem checked={reformatting} disabled={!hasData} onCheckedChange={(checked) => store().setReformatInThreeD(role, checked)}>
            Reformat handle in 3D
          </DropdownMenuCheckboxItem>
          <DropdownMenuItem disabled={!hasData} onClick={() => store().resetSliceView(role)}>
            Reset to {ORIENTATION_LABELS[DEFAULT_ORIENTATION[role]].toLowerCase()}
          </DropdownMenuItem>
        </DropdownMenuContent>
      </DropdownMenu>
      <span className={cn(CHIP, "px-1.5 py-0.5 text-[10px] text-muted-foreground")}>{hint}</span>
    </div>
  );
}

/** Where the view's plane is; its own component so only it re-renders on every notch. */
export function SlicePosition({ role }: { role: SliceRole }) {
  const label = useSceneStore((s) => (s.worldBounds ? planePositionLabel(s.slicePlanes[role], boundsCenter(s.worldBounds)) : null));
  if (!label) return null;
  return (
    <span className={cn(CHIP, "pointer-events-none absolute bottom-1.5 left-1.5 px-1.5 py-0.5 font-mono text-[10px] text-muted-foreground")}>
      {label}
    </span>
  );
}
