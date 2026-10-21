"use client";

/**
 * The Auto controls: every step of the pipeline (Generate segmentation, labelmap,
 * surface) is a button with an Auto box beside it, and a ticked step runs by itself whenever
 * what it is made from changes (`segmentation/autoUpdate.ts`). One component, so the three
 * steps read and behave the same.
 */

import { Loader2 } from "lucide-react";

import { Button } from "@/components/ui/button";
import { Checkbox } from "@/components/ui/checkbox";
import { Label } from "@/components/ui/label";
import { useSceneStore } from "@/lib/scene/store";
import type { AutoKey } from "@/lib/segmentation/state";

/** A step's button, with the Auto box that lets it run by itself. */
export function StepButton({
  step,
  icon,
  running,
  disabled,
  onClick,
  children,
  variant = "secondary",
}: {
  step: AutoKey;
  icon: React.ReactNode;
  running: boolean;
  disabled: boolean;
  onClick: () => void;
  children: React.ReactNode;
  variant?: "default" | "secondary";
}) {
  const auto = useSceneStore((s) => s.autoUpdate[step]);
  const id = `auto-${step}`;
  return (
    <div className="flex items-center gap-2">
      <Button className="min-w-0 flex-1" size="sm" variant={variant} disabled={disabled} onClick={onClick}>
        {running ? <Loader2 className="animate-spin" /> : icon}
        {children}
      </Button>
      <div className="flex shrink-0 items-center gap-1" title="Run this step by itself whenever it is out of date: points, settings or thresholds changed">
        <Checkbox id={id} checked={auto} onCheckedChange={(checked) => useSceneStore.getState().setAutoUpdate({ [step]: checked === true })} />
        <Label htmlFor={id} className="text-[11px] font-normal text-muted-foreground">
          Auto
        </Label>
      </div>
    </div>
  );
}
