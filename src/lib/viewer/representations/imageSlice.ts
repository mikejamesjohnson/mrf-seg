/**
 * Cutting planes through a volume: the image itself (`imageSlice`), or what is drawn over it,
 * a map (`overlay`) or a labelmap (`labelmap`).
 *
 * A slice pane has one actor, cut by its view's plane; the 3D pane has one per plane shown
 * there, which is why this holds a map of actors rather than one (a node per view would mean
 * three copies of the window/level). What is drawn over the image is sampled on exactly the
 * same plane, coloured by a ramp (the labelmap's with per-value opacity), always drawn
 * translucent so it composites over the image rather than painting over it, and placed by its
 * layer (`layers.ts`): the *actor* is moved a hair toward the camera, never the sampled plane,
 * so every layer shows the same voxels; a labelmap is drawn in the foreground renderer too.
 */

import vtkCellArray from "@kitware/vtk.js/Common/Core/CellArray";
import vtkDataArray from "@kitware/vtk.js/Common/Core/DataArray";
import vtkPoints from "@kitware/vtk.js/Common/Core/Points";
import vtkPiecewiseFunction from "@kitware/vtk.js/Common/DataModel/PiecewiseFunction";
import vtkPlane from "@kitware/vtk.js/Common/DataModel/Plane";
import vtkPolyData from "@kitware/vtk.js/Common/DataModel/PolyData";
import vtkActor from "@kitware/vtk.js/Rendering/Core/Actor";
import type vtkCamera from "@kitware/vtk.js/Rendering/Core/Camera";
import vtkColorTransferFunction from "@kitware/vtk.js/Rendering/Core/ColorTransferFunction";
import vtkImageResliceMapper from "@kitware/vtk.js/Rendering/Core/ImageResliceMapper";
import vtkImageSlice from "@kitware/vtk.js/Rendering/Core/ImageSlice";
import { InterpolationType } from "@kitware/vtk.js/Rendering/Core/ImageProperty/Constants";
import vtkMapper from "@kitware/vtk.js/Rendering/Core/Mapper";

import { getImage } from "@/lib/scene/dataRegistry";
import { isSliceRole, planesForThreeD, SLICE_COLORS, type SlicePlane, type SliceRole } from "@/lib/scene/views";
import { opacityStops } from "@/lib/viewer/colorRamp";
import { FOREGROUND, layerOffset, type Layer } from "@/lib/viewer/layers";
import { colouringOf, type Colouring, type SliceNode } from "@/lib/viewer/sliceColouring";
import { visiblePolygon, wholePlane } from "@/lib/viewer/planeGeometry";
import { add, boundsDiagonal, dot, scale, sub, type Bounds, type Vec3 } from "@/lib/world";
import { inputKey, type Representation, type RepresentationContext } from "./types";

const LAYER: Record<SliceNode["type"], Layer> = { imageSlice: "image", overlay: "map", labelmap: "labelmap" };

/**
 * Colour and per-value opacity: the reslice mapper looks each up over its own function's
 * range. Always explicit functions, so every ramp takes one path -- and an opacity function
 * even for an opaque ramp: without one, the mapper makes a new opacity texture on every
 * rebuild (every scroll step) and never frees the last.
 */
function transferFunctions({ ramp, window, level }: Colouring) {
  const half = Math.max(window / 2, 1e-6);
  const ctf = vtkColorTransferFunction.newInstance();
  for (const stop of ramp.colors) ctf.addRGBPoint(level + stop.at * half, ...stop.rgb);
  const pwf = vtkPiecewiseFunction.newInstance();
  for (const [value, alpha] of opacityStops(ramp, window, level)) pwf.addPoint(value, alpha);
  return { ctf, pwf };
}

/** Which side of the plane the camera is on: +1 where the normal points. */
function cameraSide(camera: vtkCamera, plane: SlicePlane): 1 | -1 {
  return dot(sub([...camera.getPosition()] as Vec3, plane.origin), plane.normal) >= 0 ? 1 : -1;
}

/**
 * The plane as the reslice mapper's geometry in the 3D pane: the rectangle its slice view
 * shows, clipped to the volume (`planeGeometry.ts`), as a triangle fan. A rectangle reaching
 * past the volume would have an outside the mapper paints, and writes depth for. Point normals
 * are required (without them nothing draws), and so is a bumped MTime.
 */
function writeQuad(poly: vtkPolyData, polygon: Vec3[], normal: Vec3): void {
  const fan = new Uint32Array((polygon.length - 2) * 4);
  for (let t = 0; t < polygon.length - 2; t += 1) fan.set([3, 0, t + 1, t + 2], t * 4);
  poly.getPoints().setData(new Float32Array(polygon.flat()), 3);
  poly.setPolys(vtkCellArray.newInstance({ values: fan }));
  poly.getPointData().setNormals(vtkDataArray.newInstance({ name: "Normals", numberOfComponents: 3, values: new Float32Array(polygon.flatMap(() => normal)) }));
  poly.modified();
}

/** The polygon's outline, pushed out by `margin` so it does not fight the image over depth. */
function writeBorder(poly: vtkPolyData, polygon: Vec3[], margin: number): void {
  const centroid = scale(polygon.reduce<Vec3>((sum, p) => add(sum, p), [0, 0, 0]), 1 / polygon.length);
  const pushed = polygon.map((p) => {
    const d = sub(p, centroid);
    return add(p, scale(d, margin / (Math.hypot(...d) || 1)));
  });
  poly.getPoints().setData(new Float32Array(pushed.flat()), 3);
  poly.setLines(vtkCellArray.newInstance({ values: new Uint32Array([polygon.length + 1, ...polygon.keys(), 0]) }));
  poly.modified();
}

interface SliceActor {
  plane: vtkPlane;
  mapper: vtkImageResliceMapper;
  actor: vtkImageSlice;
  inputKey: string | null;
  colourKey: string;
  /** The quad and border last written, so an unchanged one is not rebuilt on the GPU. */
  geometryKey: string;
  ctf: vtkColorTransferFunction | null;
  pwf: vtkPiecewiseFunction | null;
  /** The visible-rectangle geometry: 3D pane only. */
  quad: vtkPolyData | null;
  /** The plane's border in its view's colour: 3D pane, image only. */
  border: { poly: vtkPolyData; mapper: vtkMapper; actor: vtkActor } | null;
}

export function createImageSliceRepresentation(first: SliceNode, ctx: RepresentationContext): Representation<SliceNode> {
  const actors = new Map<SliceRole, SliceActor>();
  const inThreeD = ctx.role === "3d";
  const overlay = first.type !== "imageSlice";
  const layer = LAYER[first.type];
  const renderer = FOREGROUND.has(layer) ? ctx.foreground : ctx.renderer;
  const camera = ctx.renderer.getActiveCamera();

  function ensure(role: SliceRole): SliceActor {
    const existing = actors.get(role);
    if (existing) return existing;

    const plane = vtkPlane.newInstance();
    const mapper = vtkImageResliceMapper.newInstance();
    mapper.setSlicePlane(plane);
    // Transparent where the plane leaves the volume, not opaque black.
    mapper.setBackgroundColor(0, 0, 0, 0);
    const actor = vtkImageSlice.newInstance();
    actor.setMapper(mapper);
    // vtk.js judges an image slice opaque by its overall opacity alone, ignoring per-value
    // opacity: an overlay at full opacity would go in the opaque pass and paint over the image.
    if (overlay) actor.setForceTranslucent(true);
    renderer.addActor(actor);

    let quad: vtkPolyData | null = null;
    let border: SliceActor["border"] = null;
    if (inThreeD) {
      // Not pickable, so a reformat handle lying in the plane can be grabbed through it.
      actor.setPickable(false);
      quad = vtkPolyData.newInstance();
      quad.setPoints(vtkPoints.newInstance());
      mapper.setSlicePolyData(quad);
      if (!overlay) {
        const poly = vtkPolyData.newInstance();
        poly.setPoints(vtkPoints.newInstance());
        const lineMapper = vtkMapper.newInstance();
        lineMapper.setInputData(poly);
        const lineActor = vtkActor.newInstance({ pickable: false });
        lineActor.setMapper(lineMapper);
        lineActor.getProperty().setColor(...SLICE_COLORS[role]);
        lineActor.getProperty().setLineWidth(2);
        lineActor.getProperty().setLighting(false);
        ctx.renderer.addActor(lineActor);
        border = { poly, mapper: lineMapper, actor: lineActor };
      }
    }

    const created: SliceActor = { plane, mapper, actor, inputKey: null, colourKey: "", geometryKey: "", ctf: null, pwf: null, quad, border };
    actors.set(role, created);
    return created;
  }

  function hide(slice: SliceActor): void {
    slice.actor.setVisibility(false);
    slice.border?.actor.setVisibility(false);
  }

  /** The camera side each overlay plane was last moved toward. */
  const nudgedToward = new Map<SliceRole, 1 | -1>();
  let lastNode: SliceNode = first;

  function update(next: SliceNode): void {
    lastNode = next;
    const state = ctx.getState();
    const image = getImage(next.datasetId);
    const dataset = state.datasets[next.datasetId];
    const wanted = new Set(next.visible && image ? (isSliceRole(ctx.role) ? [ctx.role] : planesForThreeD(state)) : []);
    for (const [role, slice] of actors) {
      if (wanted.has(role)) continue;
      hide(slice);
      nudgedToward.delete(role); // the camera watch has nothing to keep right for a plane not drawn
    }
    if (!image || !next.visible) return;

    const bounds = [...image.getBounds()] as Bounds;
    const colouring = colouringOf(next, dataset);
    const { window, level, key: colourKey } = colouring;
    const interpolation = next.interpolation === "nearest" ? InterpolationType.NEAREST : InterpolationType.LINEAR;
    for (const role of wanted) {
      const plane = state.slicePlanes[role];
      const slice = ensure(role);
      const key = inputKey(dataset);
      if (slice.inputKey !== key) {
        slice.mapper.setInputData(image);
        slice.inputKey = key;
      }

      // Sampled on the view's plane itself. The mapper picks up in-place changes to the plane:
      // its MTime is in the rebuild key.
      slice.plane.setOrigin(...plane.origin);
      slice.plane.setNormal(...plane.normal);
      // What is over the image is *drawn* a hair in front of it, on whichever side the camera
      // is, by moving the actor: the mapper samples the volume where the geometry is, so moving
      // the plane would show other voxels (a layer or two off, on a fine grid).
      if (overlay) {
        const side = cameraSide(camera, plane);
        nudgedToward.set(role, side);
        slice.actor.setPosition(...layerOffset(layer, plane.normal, side));
      }

      // 3D pane only: a slice pane keeps the whole plane, so it can be panned.
      if (slice.quad) {
        const polygon = visiblePolygon(plane, state.sliceExtents[role] ?? wholePlane(bounds), bounds);
        if (polygon.length < 3) {
          hide(slice);
          continue;
        }
        const geometryKey = `${polygon.flat().join()}|${plane.normal.join()}`;
        if (geometryKey !== slice.geometryKey) {
          writeQuad(slice.quad, polygon, plane.normal);
          if (slice.border) writeBorder(slice.border.poly, polygon, boundsDiagonal(bounds) * 0.002);
          slice.geometryKey = geometryKey;
        }
        slice.border?.actor.setVisibility(true);
      }

      const property = slice.actor.getProperty();
      if (colourKey !== slice.colourKey) {
        const previous = [slice.ctf, slice.pwf];
        ({ ctf: slice.ctf, pwf: slice.pwf } = transferFunctions(colouring));
        property.setRGBTransferFunction(0, slice.ctf);
        property.setPiecewiseFunction(0, slice.pwf);
        slice.colourKey = colourKey;
        for (const fn of previous) fn?.delete();
      }
      // Kept with the transfer function: the mapper feeds both to the shader, which then maps
      // exactly the function's range.
      property.setColorWindow(window);
      property.setColorLevel(level);
      property.setInterpolationType(interpolation);
      property.setOpacity(next.opacity);
      slice.actor.setVisibility(true);
    }
  }

  // The 3D camera can orbit to the other side of a plane, which would leave an overlay behind
  // its image: move it across when that happens.
  const cameraWatch =
    overlay && inThreeD
      ? camera.onModified(() => {
          const planes = ctx.getState()?.slicePlanes;
          if (planes && [...nudgedToward].some(([role, side]) => cameraSide(camera, planes[role]) !== side)) {
            update(lastNode);
            ctx.requestRender();
          }
        })
      : null;

  return {
    update,

    describe: () => ({
      planes: [...actors].map(([role, slice]) => ({
        role,
        visible: slice.actor.getVisibility(),
        origin: [...slice.plane.getOrigin()],
        normal: [...slice.plane.getNormal()],
        /** Where the actor is drawn relative to the plane it samples: the layer's offset. */
        drawnAt: [...slice.actor.getPosition()],
        quadBounds: slice.quad ? [...slice.quad.getPoints().getBounds()] : null,
        borderVisible: slice.border ? slice.border.actor.getVisibility() : null,
        translucent: !slice.actor.getIsOpaque(),
        interpolation: slice.actor.getProperty().getInterpolationType() === InterpolationType.NEAREST ? "nearest" : "trilinear",
      })),
    }),

    destroy() {
      cameraWatch?.unsubscribe();
      for (const slice of actors.values()) {
        renderer.removeActor(slice.actor);
        slice.actor.delete();
        slice.mapper.delete();
        slice.plane.delete();
        slice.ctf?.delete();
        slice.pwf?.delete();
        slice.quad?.delete();
        if (slice.border) {
          ctx.renderer.removeActor(slice.border.actor);
          slice.border.actor.delete();
          slice.border.mapper.delete();
          slice.border.poly.delete();
        }
      }
      actors.clear();
    },
  };
}
