/**
 * The segmentation backend's API, through the editor's `/api/mrf` route: plain requests and no
 * state, and the one place that knows the wire format (snake_case fields, binary layouts). The
 * other side is `python/mrfseg/server/`.
 */

import type { MapQuantity, SurfaceSource } from "@/lib/scene/types";
import type { Mat3, Vec3 } from "@/lib/world";
import type { ExtractorInfo, ExtractorInfos, FeatureExtractorName, MRFSettings } from "./settings";
import type { FeaturesSummary, InferenceSummary, TaskProgress } from "./state";

export class BackendError extends Error {
  /** HTTP status, or 0 when the request got no usable answer. */
  readonly status: number;

  constructor(message: string, status: number) {
    super(message);
    this.status = status;
  }

  /** The session is gone: the backend restarted, or dropped it for a newer one. */
  get sessionLost(): boolean {
    return this.status === 404;
  }

  /** Worth asking again: no answer, or the server failed rather than refused. */
  get transient(): boolean {
    return this.status === 0 || this.status >= 500;
  }
}

/** The grid the backend read, the direction as [i axis, j axis, k axis]. */
export interface BackendGeometry {
  dimensions: Vec3;
  spacing: Vec3;
  origin: Vec3;
  direction: Mat3;
}

export type BackendTask<Result> = TaskProgress & { result?: Result };

/** A map's real range, and what its compact values are multiplied by. */
export interface MapSummary {
  range: [number, number];
  scale: number;
}

export interface SegmentationResult {
  points: number;
  seconds: { features: number; potentials: number; inference: number; maps: number };
  inference: InferenceSummary;
  maps: Record<MapQuantity, MapSummary>;
}

export interface SessionStatus {
  features: BackendTask<FeaturesSummary>;
  job: BackendTask<SegmentationResult>;
}

export interface LabelmapSummary {
  threshold: number;
  voxels: number;
  seconds: number;
}

export interface SurfaceSummary {
  source: SurfaceSource;
  level: number | null;
  points: number;
  triangles: number;
  seconds: number;
}

/** Points and normals (x, y, z each) and triangles (three point indices each). */
export interface Mesh {
  points: Float32Array;
  normals: Float32Array;
  triangles: Uint32Array;
}

/** `featureSpacing` -> `feature_spacing`, one level deep: the Python side's names. */
export function toSnakeCase(fields: object): Record<string, unknown> {
  return Object.fromEntries(Object.entries(fields).map(([key, value]) => [key.replace(/[A-Z]/g, (c) => `_${c.toLowerCase()}`), value]));
}

/** `feature_spacing` -> `featureSpacing`, one level deep. */
export function toCamelCase(fields: object): Record<string, unknown> {
  return Object.fromEntries(Object.entries(fields).map(([key, value]) => [key.replace(/_([a-z])/g, (_, c: string) => c.toUpperCase()), value]));
}

/** How each map comes with `?encoding=compact` (`mrfseg/server/encoding.py`). */
const COMPACT_TYPES = { unary: Int16Array, coupling: Int16Array, probability: Uint8Array } as const;

async function call(path: string, init?: RequestInit): Promise<Response> {
  let response: Response;
  try {
    response = await fetch(`/api/mrf/${path}`, { cache: "no-store", ...init });
  } catch (cause) {
    throw new BackendError(`Could not reach the editor's server (${String(cause)})`, 0);
  }
  if (!response.ok) {
    const body = await response.json().catch(() => null);
    throw new BackendError(typeof body?.error === "string" ? body.error : `${response.status} ${response.statusText}`, response.status);
  }
  return response;
}

function postJson(body: unknown): RequestInit {
  return { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body) };
}

async function bytes(path: string, what: string, expected: number): Promise<ArrayBuffer> {
  const buffer = await (await call(path)).arrayBuffer();
  if (buffer.byteLength !== expected) throw new BackendError(`The ${what} is ${buffer.byteLength} bytes; the image needs ${expected}`, 0);
  return buffer;
}

/**
 * A new MRFSegmentation for the image at `path`; the backend reads the file itself. Its
 * settings come back with what `settings` left to the image (null) chosen.
 */
export async function openSession(path: string, settings: MRFSettings): Promise<{ id: string; geometry: BackendGeometry; settings: MRFSettings }> {
  const opened = await (await call("sessions", postJson({ path, settings: toSnakeCase(settings) }))).json();
  return { ...opened, settings: toCamelCase(opened.settings) as unknown as MRFSettings };
}

export async function sessionStatus(sessionId: string): Promise<SessionStatus> {
  const { features, job } = await (await call(`sessions/${sessionId}`)).json();
  return { features, job };
}

/** Start building the potentials from the points and inferring the maps; poll `sessionStatus` for the result. */
export async function startSegmentation(sessionId: string, points: Vec3[], labels: number[], settings: MRFSettings): Promise<void> {
  await call(`sessions/${sessionId}/segmentation`, postJson({ points, labels, settings: toSnakeCase(settings) }));
}

/** What the backend says about each extractor: availability, the settings to start from, its native grid. */
export async function fetchExtractors(): Promise<ExtractorInfos> {
  const { extractors } = await (await call("extractors")).json();
  const infos: ExtractorInfos = {};
  for (const [name, info] of Object.entries(extractors as Record<string, Record<string, unknown>>)) {
    const converted = toCamelCase(info) as unknown as ExtractorInfo;
    const settings = converted.settings ? (toCamelCase(converted.settings) as Partial<MRFSettings>) : null;
    infos[name as FeatureExtractorName] = { ...converted, settings };
  }
  return infos;
}

/** One of the last segmentation's maps, compact, one value per voxel in VTK order. Multiply by its `scale`. */
export async function fetchMap(sessionId: string, quantity: MapQuantity, voxels: number): Promise<Int16Array | Uint8Array> {
  const Type = COMPACT_TYPES[quantity];
  return new Type(await bytes(`sessions/${sessionId}/segmentation/${quantity}?encoding=compact`, `${quantity} map`, voxels * Type.BYTES_PER_ELEMENT));
}

export async function buildLabelmap(sessionId: string, threshold: number): Promise<LabelmapSummary> {
  return (await call(`sessions/${sessionId}/labelmap`, postJson({ threshold }))).json();
}

/** The last labelmap: one uint8 per voxel, 1 inside, in VTK order. */
export async function fetchLabelmap(sessionId: string, voxels: number): Promise<Uint8Array> {
  return new Uint8Array(await bytes(`sessions/${sessionId}/labelmap`, "labelmap", voxels));
}

export async function buildSurface(sessionId: string, source: SurfaceSource, level?: number): Promise<SurfaceSummary> {
  return (await call(`sessions/${sessionId}/surface`, postJson({ source, level }))).json();
}

/** The last surface, from `encoding.mesh_bytes`: two uint32 counts, then the arrays. */
export async function fetchSurface(sessionId: string): Promise<Mesh> {
  const buffer = await (await call(`sessions/${sessionId}/surface`)).arrayBuffer();
  const [points, triangles] = new Uint32Array(buffer, 0, 2);
  const expected = 8 + points * 24 + triangles * 12;
  if (buffer.byteLength !== expected) throw new BackendError(`The surface is ${buffer.byteLength} bytes; its header says ${expected}`, 0);
  return {
    points: new Float32Array(buffer, 8, points * 3),
    normals: new Float32Array(buffer, 8 + points * 12, points * 3),
    triangles: new Uint32Array(buffer, 8 + points * 24, triangles * 3),
  };
}

/** Fire and forget; `keepalive` lets it finish while the page unloads. */
export function closeSession(sessionId: string): void {
  void fetch(`/api/mrf/sessions/${sessionId}`, { method: "DELETE", keepalive: true }).catch(() => {});
}
