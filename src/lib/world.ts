/**
 * The world-space contract: there is one world space, **LPS millimetres** (the DICOM patient
 * convention: +x Left, +y Posterior, +z Superior). Every actor is already in it. Loaders are
 * the only layer that may apply a corrective transform; if two datasets do not line up, the
 * bug is in a loader.
 */

export const WORLD_SPACE = "LPS-mm";

export type Vec3 = [number, number, number];

/** Nine numbers of a 3x3 matrix, row by row. */
export type Mat3 = [number, number, number, number, number, number, number, number, number];

/** VTK bounds ordering: [xMin, xMax, yMin, yMax, zMin, zMax]. */
export type Bounds = [number, number, number, number, number, number];

export const IDENTITY_MAT3: Mat3 = [1, 0, 0, 0, 1, 0, 0, 0, 1];

/**
 * Transpose a row-major 3x3. VTK takes the *columns* of an image's direction matrix as the
 * voxel axes and writes it row-major into a `.vti`'s `Direction`; vtk.js reads the same nine
 * numbers as **[i axis, j axis, k axis]**. So an oblique `.vti` is transposed on load.
 */
export function transposeMat3(m: Mat3): Mat3 {
  return [m[0], m[3], m[6], m[1], m[4], m[7], m[2], m[5], m[8]];
}

export function isIdentityMat3(m: Mat3, epsilon = 1e-9): boolean {
  return m.every((v, i) => Math.abs(v - IDENTITY_MAT3[i]) < epsilon);
}

function boundsAreValid(b: Bounds | null | undefined): b is Bounds {
  return !!b && b.every(Number.isFinite) && b[0] <= b[1] && b[2] <= b[3] && b[4] <= b[5];
}

export function unionBounds(a: Bounds | null, b: Bounds | null): Bounds | null {
  if (!boundsAreValid(a)) return boundsAreValid(b) ? b : null;
  if (!boundsAreValid(b)) return a;
  return [Math.min(a[0], b[0]), Math.max(a[1], b[1]), Math.min(a[2], b[2]), Math.max(a[3], b[3]), Math.min(a[4], b[4]), Math.max(a[5], b[5])];
}

export function boundsCenter(b: Bounds): Vec3 {
  return [(b[0] + b[1]) / 2, (b[2] + b[3]) / 2, (b[4] + b[5]) / 2];
}

export function boundsDiagonal(b: Bounds): number {
  return Math.hypot(b[1] - b[0], b[3] - b[2], b[5] - b[4]);
}

export function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

export function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export function add(a: Vec3, b: Vec3): Vec3 {
  return [a[0] + b[0], a[1] + b[1], a[2] + b[2]];
}

export function sub(a: Vec3, b: Vec3): Vec3 {
  return [a[0] - b[0], a[1] - b[1], a[2] - b[2]];
}

export function scale(v: Vec3, s: number): Vec3 {
  return [v[0] * s, v[1] * s, v[2] * s];
}

/** `v` at unit length, or null when it has none to scale. */
export function normalize(v: Vec3): Vec3 | null {
  const length = Math.hypot(v[0], v[1], v[2]);
  return length > 1e-12 ? scale(v, 1 / length) : null;
}

/** `p` moved along the unit normal `n` onto the plane through `origin`. */
export function projectOntoPlane(p: Vec3, origin: Vec3, n: Vec3): Vec3 {
  return sub(p, scale(n, dot(sub(p, origin), n)));
}

/**
 * `v` turned by the smallest rotation taking unit `from` onto unit `to` (Rodrigues). Carries a
 * slice view's screen-up along when its plane is tilted, so the view does not spin.
 */
export function rotateWith(v: Vec3, from: Vec3, to: Vec3): Vec3 {
  const axis = cross(from, to);
  const sin = Math.hypot(axis[0], axis[1], axis[2]);
  // Parallel, or opposite: then any perpendicular axis will do, and one through v leaves v as it is.
  if (sin < 1e-12) return [...v];
  const k = scale(axis, 1 / sin);
  const cos = dot(from, to);
  return add(add(scale(v, cos), scale(cross(k, v), sin)), scale(k, dot(k, v) * (1 - cos)));
}

// --------------------------------------------------------------------------- //
// Voxel grids
// --------------------------------------------------------------------------- //

/** A voxel grid's geometry. `direction` is in vtk.js layout: [i axis, j axis, k axis]. */
export interface GridGeometry {
  gridOrigin: Vec3;
  spacing: Vec3;
  dimensions: Vec3;
  direction: Mat3;
}

/** The world direction of voxel axis `a` (0 = i, 1 = j, 2 = k). */
export function voxelAxis(grid: GridGeometry, a: 0 | 1 | 2): Vec3 {
  return [grid.direction[a * 3], grid.direction[a * 3 + 1], grid.direction[a * 3 + 2]];
}

/**
 * The eight outermost voxel *centres*, in world space: half a voxel inside vtk.js's
 * `getBounds()`, which reaches the outer voxel faces. A plane between the two cuts no data.
 */
export function voxelCenterCorners(grid: GridGeometry): Vec3[] {
  return Array.from({ length: 8 }, (_, corner) => {
    let p = grid.gridOrigin;
    for (const a of [0, 1, 2] as const) {
      if (corner & (1 << a)) p = add(p, scale(voxelAxis(grid, a), grid.spacing[a] * (grid.dimensions[a] - 1)));
    }
    return p;
  });
}

/** `p` moved to the nearest place between the grid's outermost voxel centres, in its own frame. */
export function clampToGrid(grid: GridGeometry, p: Vec3): Vec3 {
  let out = grid.gridOrigin;
  for (const a of [0, 1, 2] as const) {
    const index = dot(sub(p, grid.gridOrigin), voxelAxis(grid, a)) / grid.spacing[a];
    out = add(out, scale(voxelAxis(grid, a), Math.min(Math.max(index, 0), grid.dimensions[a] - 1) * grid.spacing[a]));
  }
  return out;
}

/** Whether `p` lies between the grid's outermost voxel centres. */
export function insideGrid(grid: GridGeometry, p: Vec3, tolerance = 1e-6): boolean {
  const back = clampToGrid(grid, p);
  return Math.hypot(back[0] - p[0], back[1] - p[1], back[2] - p[2]) <= tolerance;
}
