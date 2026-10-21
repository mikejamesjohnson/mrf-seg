"use client";

import { MousePointer2, Trash2 } from "lucide-react";
import { useEffect, useRef } from "react";
import { useShallow } from "zustand/react/shallow";

import { ChoiceButtons, Hint, SwitchRow, type ChoiceOption } from "@/components/panels/controls";
import { Button } from "@/components/ui/button";
import { currentImage } from "@/lib/scene/datasets";
import { pointColor, pointCounts, type DataPoint, type PlaceMode } from "@/lib/scene/points";
import { useSceneStore } from "@/lib/scene/store";
import { cn } from "@/lib/utils";

const dot = (color: string) => <span className="size-2 shrink-0 rounded-full" style={{ backgroundColor: color }} />;

const MODES: ChoiceOption<PlaceMode>[] = [
  { value: "navigate", label: "Navigate", title: "Navigate (1)", icon: <MousePointer2 className="size-3" /> },
  { value: "inside", label: "Inside", title: "Inside (2)", icon: dot(pointColor(1)) },
  { value: "outside", label: "Outside", title: "Outside (3)", icon: dot(pointColor(-1)) },
];

/**
 * The labelled points: the place mode, the list, and removing them. The list's selection is
 * the scene's, so a row picked here is highlighted in every view.
 */
export function PointsPanel() {
  const hasImage = useSceneStore((s) => currentImage(s) !== undefined);
  const mode = useSceneStore((s) => s.placeMode);
  const points = useSceneStore((s) => s.points);
  const selected = useSceneStore(useShallow((s) => s.selectedPointIds));
  const visible = useSceneStore((s) => s.pointsVisible);
  const { inside, outside } = pointCounts(points);
  const store = useSceneStore.getState;

  if (!hasImage) return <Hint>Open an image to place points on it.</Hint>;

  return (
    <div className="space-y-3">
      <ChoiceButtons label="Place mode" value={mode} options={MODES} onChange={(next) => store().setPlaceMode(next)} />
      <Hint>
        {mode === "navigate"
          ? "Pick Inside or Outside (keys 2, 3), then click a slice to add points. Drag a point to move it; Shift-drag to select several."
          : `Click a slice to add ${mode} points. Drag a point to move it; Shift-drag to select several; Delete removes.`}
      </Hint>
      <div className="space-y-1.5">
        <SwitchRow id="points-in-3d" label="Show in 3D" checked={visible.threeD} onChange={(on) => store().setPointsVisible("threeD", on)} />
        <SwitchRow id="points-on-slices" label="Show on slices" checked={visible.slices} onChange={(on) => store().setPointsVisible("slices", on)} />
      </div>

      <div className="flex items-center justify-between text-[11px] text-muted-foreground">
        <span>
          {points.length === 0 ? "No points yet" : `${inside} inside · ${outside} outside`}
          {selected.length > 0 && ` · ${selected.length} selected`}
        </span>
        <Button
          size="xs"
          variant="ghost"
          disabled={selected.length === 0}
          onClick={() => store().removePoints(store().selectedPointIds)}
          title="Remove the selected points (Delete)"
        >
          <Trash2 />
          Remove
        </Button>
      </div>

      {points.length > 0 && <PointList points={points} selected={selected} />}
    </div>
  );
}

/** Click selects a point and moves the slice views through it; Cmd/Ctrl-click toggles; Shift-click takes a run. */
function PointList({ points, selected }: { points: DataPoint[]; selected: number[] }) {
  const anchor = useRef<number | null>(null);
  const list = useRef<HTMLDivElement>(null);
  const chosen = new Set(selected);
  const store = useSceneStore.getState;

  // Keep the most recently selected row in view, e.g. after clicking its point in a slice. By
  // the list's own scroll: scrollIntoView would scroll the side panel as well.
  const last = selected[selected.length - 1];
  useEffect(() => {
    const container = list.current;
    const row = last === undefined ? null : container?.querySelector<HTMLElement>(`[data-point-id="${last}"]`);
    if (!container || !row) return;
    // Under the sticky header is not in view.
    const top = row.offsetTop - ((container.firstElementChild as HTMLElement | null)?.offsetHeight ?? 0);
    const bottom = row.offsetTop + row.offsetHeight - container.clientHeight;
    if (container.scrollTop > top) container.scrollTop = top;
    else if (container.scrollTop < bottom) container.scrollTop = bottom;
  }, [last]);

  const pick = (point: DataPoint, event: React.MouseEvent) => {
    const toggle = event.metaKey || event.ctrlKey;
    const from = points.findIndex((p) => p.id === anchor.current);
    if (event.shiftKey && from >= 0) {
      const to = points.findIndex((p) => p.id === point.id);
      store().selectPoints(points.slice(Math.min(from, to), Math.max(from, to) + 1).map((p) => p.id), toggle ? "add" : "replace");
      return;
    }
    anchor.current = point.id;
    if (toggle) {
      store().selectPoints([point.id], "toggle");
    } else {
      store().selectPoints([point.id], "replace");
      store().jumpSlicesTo(point.position);
    }
  };

  const columns = "grid grid-cols-[2.25rem_1fr_1fr_1fr] gap-1 px-2";
  return (
    <div
      ref={list}
      tabIndex={0}
      role="listbox"
      aria-label="Points"
      aria-multiselectable
      className="relative max-h-56 overflow-y-auto rounded-md border border-border/50 outline-none focus-visible:ring-2 focus-visible:ring-ring/50"
      onKeyDown={(event) => {
        if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === "a") {
          event.preventDefault();
          store().selectPoints(points.map((p) => p.id), "replace");
        }
      }}
    >
      <div className={cn(columns, "sticky top-0 border-b border-border/50 bg-background/95 py-1 font-mono text-[10px] text-muted-foreground backdrop-blur-sm")}>
        <span>ID</span>
        <span className="text-right">L</span>
        <span className="text-right">P</span>
        <span className="text-right">S</span>
      </div>
      {points.map((point) => (
        <div
          key={point.id}
          data-point-id={point.id}
          role="option"
          aria-selected={chosen.has(point.id)}
          onMouseDown={(event) => event.shiftKey && event.preventDefault()}
          onClick={(event) => pick(point, event)}
          className={cn(
            columns,
            "cursor-default items-center py-0.5 font-mono text-[11px] select-none",
            chosen.has(point.id) ? "bg-primary/20 text-foreground" : "text-foreground/80 hover:bg-muted/50",
          )}
        >
          <span className="flex items-center gap-1.5" title={point.label > 0 ? "Inside (+1)" : "Outside (-1)"}>
            {dot(pointColor(point.label))}
            {point.id}
          </span>
          {point.position.map((v, axis) => (
            <span key={axis} className="text-right tabular-nums">
              {v.toFixed(1)}
            </span>
          ))}
        </div>
      ))}
    </div>
  );
}
