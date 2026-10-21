/**
 * The segmentation's surface mesh. In the 3D pane, the mesh itself, lit as a closed surface
 * (the backend sends outward normals). In a slice view, the line where the view's plane cuts
 * it, recomputed whenever the plane or the mesh changes, and drawn in the foreground, over
 * everything else (`layers.ts`). Neither is pickable, so neither stands between the pointer
 * and a point or a handle.
 */

import vtkCellArray from "@kitware/vtk.js/Common/Core/CellArray";
import vtkPolyData from "@kitware/vtk.js/Common/DataModel/PolyData";
import vtkActor from "@kitware/vtk.js/Rendering/Core/Actor";
import vtkMapper from "@kitware/vtk.js/Rendering/Core/Mapper";
import { Representation as Drawn } from "@kitware/vtk.js/Rendering/Core/Property/Constants";

import { getMesh } from "@/lib/scene/dataRegistry";
import type { SurfaceNode } from "@/lib/scene/types";
import { isSliceRole, type SliceRole } from "@/lib/scene/views";
import { layerOffset } from "@/lib/viewer/layers";
import { meshPlaneSegments } from "@/lib/viewer/planeGeometry";
import { inputKey, type Representation, type RepresentationContext } from "./types";

export function createSurfaceRepresentation(first: SurfaceNode, ctx: RepresentationContext): Representation<SurfaceNode> {
  return isSliceRole(ctx.role) ? createOutline(first, ctx, ctx.role) : createMesh(first, ctx);
}

function createMesh(first: SurfaceNode, ctx: RepresentationContext): Representation<SurfaceNode> {
  const mapper = vtkMapper.newInstance({ scalarVisibility: false });
  const actor = vtkActor.newInstance({ pickable: false });
  actor.setMapper(mapper);
  const property = actor.getProperty();
  property.setSpecular(0.25);
  property.setSpecularPower(30);
  property.setDiffuse(0.8);
  property.setAmbient(0.15);
  ctx.renderer.addActor(actor);
  let currentKey: string | null = null;

  return {
    update(next) {
      const key = inputKey(ctx.getState().datasets[next.datasetId]);
      const mesh = getMesh(next.datasetId);
      if (key !== currentKey && mesh) {
        mapper.setInputData(mesh);
        currentKey = key;
      }
      property.setColor(...next.color);
      property.setOpacity(next.opacity);
      property.setRepresentation(next.wireframe ? Drawn.WIREFRAME : Drawn.SURFACE);
      actor.setVisibility(next.visible && !!mesh);
    },

    describe: () => ({
      visible: actor.getVisibility(),
      bounds: [...actor.getBounds()],
      points: getMesh(first.datasetId)?.getNumberOfPoints() ?? 0,
      color: [...property.getColor()],
      opacity: property.getOpacity(),
      wireframe: property.getRepresentation() === Drawn.WIREFRAME,
    }),

    destroy() {
      ctx.renderer.removeActor(actor);
      actor.delete();
      mapper.delete();
    },
  };
}

function createOutline(first: SurfaceNode, ctx: RepresentationContext, role: SliceRole): Representation<SurfaceNode> {
  const lines = vtkPolyData.newInstance();
  const mapper = vtkMapper.newInstance({ scalarVisibility: false });
  mapper.setInputData(lines);
  const actor = vtkActor.newInstance({ pickable: false });
  actor.setMapper(mapper);
  const property = actor.getProperty();
  property.setLighting(false);
  ctx.foreground.addActor(actor);
  /** The mesh version and plane the outline was last cut for. */
  let cutFor = "";
  let segments = 0;

  return {
    update(next) {
      const state = ctx.getState();
      const mesh = getMesh(next.datasetId);
      if (!next.onSlices || !mesh) {
        actor.setVisibility(false);
        return;
      }
      const plane = state.slicePlanes[role];
      const key = `${inputKey(state.datasets[next.datasetId])}|${plane.origin.join()}|${plane.normal.join()}`;
      if (key !== cutFor) {
        const points = meshPlaneSegments(mesh.getPoints().getData(), mesh.getPolys().getData(), plane);
        segments = points.length / 6;
        const cells = new Uint32Array(segments * 3);
        for (let s = 0; s < segments; s += 1) cells.set([2, 2 * s, 2 * s + 1], 3 * s);
        lines.getPoints().setData(points, 3);
        lines.setLines(vtkCellArray.newInstance({ values: cells }));
        lines.modified();
        cutFor = key;
      }
      // Toward the camera, which a slice view puts on the normal's side: nothing moves on screen.
      actor.setPosition(...layerOffset("outline", plane.normal, 1));
      property.setColor(...next.color);
      // Screen pixels: the render window is sized in device pixels.
      property.setLineWidth(next.lineWidth * (window.devicePixelRatio || 1));
      actor.setVisibility(segments > 0);
    },

    describe: () => ({
      visible: actor.getVisibility(),
      segments,
      lineWidth: property.getLineWidth(),
      color: [...property.getColor()],
      datasetId: first.datasetId,
    }),

    destroy() {
      ctx.foreground.removeActor(actor);
      actor.delete();
      mapper.delete();
      lines.delete();
    },
  };
}
