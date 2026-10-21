/**
 * The mounted panes, for commands that are not scene state: framing a camera, a view
 * direction. Where a camera points is derived by nothing else, and routing it through the
 * store would mean an update per interaction frame.
 */

import type { ViewportRole } from "@/lib/scene/views";
import type { Vec3 } from "@/lib/world";
import type { ViewDirection } from "./cameras";
import type { ViewportController } from "./ViewportController";

const controllers = new Map<ViewportRole, ViewportController>();

export function registerViewport(controller: ViewportController): void {
  controllers.set(controller.role, controller);
}

export function unregisterViewport(controller: ViewportController): void {
  if (controllers.get(controller.role) === controller) controllers.delete(controller.role);
}

/** Frame one pane, or every pane. */
export function resetCamera(role?: ViewportRole): void {
  if (role) controllers.get(role)?.resetCamera();
  else for (const controller of controllers.values()) controller.resetCamera();
}

/** Only the 3D pane has a free camera. */
export function setViewDirection(direction: ViewDirection): void {
  controllers.get("3d")?.setViewDirection(direction);
}

export function describeViewports(): Record<string, unknown> {
  return Object.fromEntries([...controllers].map(([role, controller]) => [role, controller.describe()]));
}

/** Where a world point lands in a pane, in CSS pixels from its top left; for tests. */
export function projectToPane(role: ViewportRole, point: Vec3): [number, number] | null {
  return controllers.get(role)?.worldToPane(point) ?? null;
}
