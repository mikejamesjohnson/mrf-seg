"use client";

import { Crosshair, Layers, Maximize2, Minimize2 } from "lucide-react";
import { useEffect, useRef } from "react";

import { CHIP, SlicePosition, SliceViewMenu } from "@/components/viewer/slice-view-menu";
import { Button } from "@/components/ui/button";
import { useSceneStore, type SceneState } from "@/lib/scene/store";
import { isSliceRole, ROLE_LABELS, type ViewportRole } from "@/lib/scene/views";
import { cn } from "@/lib/utils";
import { registerViewport, unregisterViewport } from "@/lib/viewer/activeViewport";
import type { ViewDirection } from "@/lib/viewer/cameras";
import { drawsDifferently } from "@/lib/viewer/redraw";
import { ViewportController } from "@/lib/viewer/ViewportController";

/** Camera directions, inside the 3D pane: they only apply to it. */
const VIEW_BUTTONS: { direction: ViewDirection; short: string; label: string }[] = [
  { direction: "oblique", short: "3D", label: "Oblique" },
  { direction: "anterior", short: "A", label: "Anterior" },
  { direction: "posterior", short: "P", label: "Posterior" },
  { direction: "left", short: "L", label: "Left" },
  { direction: "right", short: "R", label: "Right" },
  { direction: "superior", short: "S", label: "Superior" },
  { direction: "inferior", short: "I", label: "Inferior" },
];

/**
 * One pane, deliberately tiny on the React side: a div handed to a `ViewportController`, which
 * is fed store snapshots through a subscription that bypasses React, at most one per frame and
 * only when something it draws has changed.
 */
export function ViewportPane({ role }: { role: ViewportRole }) {
  const containerRef = useRef<HTMLDivElement>(null);
  // A ref, not state: state would drag a vtk.js object into React's render cycle.
  const controllerRef = useRef<ViewportController | null>(null);
  const maximized = useSceneStore((s) => s.maximizedRole === role);
  const planeInThreeD = useSceneStore((s) => isSliceRole(role) && s.planesInThreeD[role]);
  const store = useSceneStore.getState;

  useEffect(() => {
    const container = containerRef.current;
    if (!container) return;
    const controller = new ViewportController(container, role, {
      onExtent: (extent) => isSliceRole(role) && store().setSliceExtent(role, extent),
      onScroll: (notches) => isSliceRole(role) && store().scrollSlice(role, notches),
      onReformat: (slice, origin, normal) => store().setSlicePlane(slice, origin, normal),
      onSync: (position) => isSliceRole(role) && store().syncSlicesTo(position, role),
      onSyncEnd: () => store().endSliceSync(),
      onAddPoint: (position) => store().addPoint(position),
      onSelectPoints: (ids, how) => store().selectPoints(ids, how),
      onPlacePoints: (positions) => store().setPointPositions(positions),
    });
    registerViewport(controller);
    controllerRef.current = controller;

    // The first sync comes before the subscription, so a store write it makes itself (a camera
    // publishing its extent) is not fed straight back. After that, at most one sync per frame,
    // from the latest snapshot: a snapshot is absolute, so a superseded one is not a skipped step.
    controller.sync(store());
    let frame = 0;
    let latest: SceneState | null = null;
    const unsubscribe = useSceneStore.subscribe((state, previous) => {
      if (!drawsDifferently(role, state, previous)) return;
      latest = state;
      frame ||= requestAnimationFrame(() => {
        frame = 0;
        const pending = latest;
        latest = null;
        if (pending) controller.sync(pending);
      });
    });

    return () => {
      unsubscribe();
      if (frame) cancelAnimationFrame(frame);
      unregisterViewport(controller);
      controller.destroy();
      controllerRef.current = null;
    };
  }, [role, store]);

  const button = cn("size-6 text-muted-foreground", CHIP);
  return (
    <div className="relative min-h-0 min-w-0 overflow-hidden border border-border/40" data-viewport-role={role}>
      <div ref={containerRef} className="absolute inset-0" />

      {/* The chrome never takes drags or the wheel from the canvas; its controls opt back in. */}
      <div className="pointer-events-none absolute inset-0 flex items-start justify-between p-1.5">
        {isSliceRole(role) ? (
          <SliceViewMenu role={role} />
        ) : (
          <span className={cn(CHIP, "px-1.5 py-0.5 text-[10px] font-medium tracking-wide text-foreground/80 uppercase")}>{ROLE_LABELS[role]}</span>
        )}

        <div className="pointer-events-auto flex items-center gap-0.5">
          {isSliceRole(role) && (
            <Button
              size="icon"
              variant="ghost"
              className={cn(button, planeInThreeD && "text-primary")}
              title={planeInThreeD ? "Hide this plane in 3D" : "Show this plane in 3D"}
              onClick={() => store().setPlaneInThreeD(role, !planeInThreeD)}
            >
              <Layers className="size-3" />
            </Button>
          )}
          <Button size="icon" variant="ghost" className={button} title="Fit this view" onClick={() => controllerRef.current?.resetCamera()}>
            <Crosshair className="size-3" />
          </Button>
          <Button
            size="icon"
            variant="ghost"
            className={button}
            title={maximized ? "Restore layout" : "Maximise this view"}
            onClick={() => store().setMaximizedRole(maximized ? null : role)}
          >
            {maximized ? <Minimize2 className="size-3" /> : <Maximize2 className="size-3" />}
          </Button>
        </div>
      </div>

      {isSliceRole(role) && <SlicePosition role={role} />}

      {role === "3d" && (
        // Bottom left: the orientation cube has the bottom right.
        <div className={cn(CHIP, "pointer-events-auto absolute bottom-1.5 left-1.5 flex items-center gap-px p-0.5")}>
          {VIEW_BUTTONS.map((view) => (
            <Button
              key={view.direction}
              size="icon"
              variant="ghost"
              className="size-5 px-0 font-mono text-[10px] text-muted-foreground hover:text-foreground"
              title={`${view.label} view`}
              onClick={() => controllerRef.current?.setViewDirection(view.direction)}
            >
              {view.short}
            </Button>
          ))}
        </div>
      )}
    </div>
  );
}
