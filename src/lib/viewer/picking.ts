/**
 * From a pixel of a pane to the world: the line of sight through it, and where that meets a
 * plane. Shared by whatever picks in a pane -- the point layer placing and dragging points, and
 * the Shift-hover sync -- so they agree on what is under the pointer.
 */

import type vtkRenderer from "@kitware/vtk.js/Rendering/Core/Renderer";

import { add, normalize, scale, sub, type Vec3 } from "@/lib/world";
import { rayPlane } from "./planeGeometry";

/** The pane's size in CSS pixels, never zero high. */
export function paneSize(container: HTMLElement): [number, number] {
  return [container.clientWidth, Math.max(container.clientHeight, 1)];
}

/** A pointer event's position in the pane, in CSS pixels from its top left. */
export function pointerInPane(container: HTMLElement, event: { clientX: number; clientY: number }): [number, number] {
  const box = container.getBoundingClientRect();
  return [event.clientX - box.left, event.clientY - box.top];
}

/** The line through the pixel at (x, y), from the camera's side. */
export function paneRay(renderer: vtkRenderer, container: HTMLElement, x: number, y: number): { from: Vec3; direction: Vec3 } | null {
  const [width, height] = paneSize(container);
  const nx = x / width;
  const ny = 1 - y / height;
  const near = renderer.normalizedDisplayToWorld(nx, ny, 0, width / height) as Vec3;
  const far = renderer.normalizedDisplayToWorld(nx, ny, 0.5, width / height) as Vec3;
  const direction = normalize(sub(far, near));
  return direction ? { from: near, direction } : null;
}

/** Where the line through the pixel at (x, y) meets `plane`, or null if it runs along it. */
export function pickOnPlane(renderer: vtkRenderer, container: HTMLElement, x: number, y: number, plane: { origin: Vec3; normal: Vec3 }): Vec3 | null {
  const ray = paneRay(renderer, container, x, y);
  if (!ray) return null;
  const t = rayPlane(ray.from, ray.direction, plane);
  return t === null ? null : add(ray.from, scale(ray.direction, t));
}
