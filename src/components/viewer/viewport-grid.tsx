"use client";

import { ViewportPane } from "@/components/viewer/viewport-pane";
import { useSceneStore } from "@/lib/scene/store";
import { LAYOUT } from "@/lib/scene/views";
import { cn } from "@/lib/utils";

/**
 * 3D plus the red, yellow and green slice views, or one pane maximised. Each pane owns its
 * render window, so the grid is just CSS. Maximising unmounts the others on purpose: a hidden
 * render window still holds a WebGL context, and browsers allow only about sixteen.
 */
export function ViewportGrid() {
  const maximizedRole = useSceneStore((s) => s.maximizedRole);
  const roles = maximizedRole ? [maximizedRole] : LAYOUT;
  return (
    <div
      className={cn("grid h-full w-full gap-px bg-border/40", roles.length === 1 ? "grid-cols-1 grid-rows-1" : "grid-cols-2 grid-rows-2")}
      // Re-keyed so a layout change remounts the panes rather than reusing a controller.
      key={maximizedRole ?? "quad"}
    >
      {roles.map((role) => (
        <ViewportPane key={role} role={role} />
      ))}
    </div>
  );
}
