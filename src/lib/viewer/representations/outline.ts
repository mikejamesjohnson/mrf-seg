/**
 * A wireframe box around a volume. It is drawn in the image's own oriented frame, so an
 * oblique volume shows a tilted box and a direction-matrix bug is obvious.
 */

import type vtkPolyData from "@kitware/vtk.js/Common/DataModel/PolyData";
import vtkImageDataOutlineFilter from "@kitware/vtk.js/Filters/General/ImageDataOutlineFilter";
import vtkActor from "@kitware/vtk.js/Rendering/Core/Actor";
import vtkMapper from "@kitware/vtk.js/Rendering/Core/Mapper";

import { getImage } from "@/lib/scene/dataRegistry";
import type { OutlineNode } from "@/lib/scene/types";
import { inputKey, type Representation, type RepresentationContext } from "./types";

export function createOutlineRepresentation(_node: OutlineNode, ctx: RepresentationContext): Representation<OutlineNode> {
  const filter = vtkImageDataOutlineFilter.newInstance();
  // Its cube source defaults to solid faces: an opaque box hiding the slices inside.
  filter.setGenerateFaces(false);
  filter.setGenerateLines(true);
  const mapper = vtkMapper.newInstance();
  // Not pickable, so it never stands between the pointer and a reformat handle.
  const actor = vtkActor.newInstance({ pickable: false });
  actor.setMapper(mapper);
  actor.getProperty().setLighting(false);
  ctx.renderer.addActor(actor);
  let currentKey: string | null = null;

  return {
    update(next) {
      const key = inputKey(ctx.getState().datasets[next.datasetId]);
      const image = getImage(next.datasetId);
      if (key !== currentKey && image) {
        filter.setInputData(image);
        mapper.setInputData(filter.getOutputData() as vtkPolyData);
        currentKey = key;
      }
      actor.getProperty().setColor(...next.color);
      actor.setVisibility(next.visible && !!image);
    },

    describe: () => ({ visible: actor.getVisibility(), bounds: [...actor.getBounds()] }),

    destroy() {
      ctx.renderer.removeActor(actor);
      actor.delete();
      mapper.delete();
      filter.delete();
    },
  };
}
