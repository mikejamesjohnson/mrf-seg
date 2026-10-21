/** A small image to put in the store, and a store reset between tests. */

import vtkDataArray from "@kitware/vtk.js/Common/Core/DataArray";
import vtkImageData from "@kitware/vtk.js/Common/DataModel/ImageData";
import { beforeEach } from "vitest";

import { useSceneStore } from "@/lib/scene/store";
import type { ImageDataset } from "@/lib/scene/types";
import { IDENTITY_MAT3, type Bounds, type Mat3, type Vec3 } from "@/lib/world";

let images = 0;

/** A volume of `dimensions` voxels whose values are their k index, as the loaders would describe it. */
export function makeImage(dimensions: Vec3, spacing: Vec3, origin: Vec3, direction: Mat3 = IDENTITY_MAT3) {
  const [nx, ny, nz] = dimensions;
  const values = new Int16Array(nx * ny * nz).map((_, n) => Math.floor(n / (nx * ny)));
  const image = vtkImageData.newInstance();
  image.setDimensions(dimensions);
  image.setSpacing(spacing);
  image.setOrigin(origin);
  image.setDirection(...direction);
  image.getPointData().setScalars(vtkDataArray.newInstance({ name: "Scalars", numberOfComponents: 1, values }));
  images += 1;
  const dataset: ImageDataset = {
    id: `test-image-${images}`,
    kind: "image",
    label: "test",
    path: "/test.mha",
    format: "mha",
    version: 1,
    bounds: [...image.getBounds()] as Bounds,
    dimensions,
    spacing,
    gridOrigin: origin,
    direction,
    arrayName: "Scalars",
    scalarType: "Int16Array",
    scalarRange: [0, nz - 1],
  };
  return { dataset, image };
}

/** `makeImage`, opened the way `openImage` does: views reset through its centre. */
export function openImage(...args: Parameters<typeof makeImage>): ImageDataset {
  const { dataset, image } = makeImage(...args);
  const store = useSceneStore.getState();
  store.addDataset(dataset, image);
  store.resetViews();
  return dataset;
}

/** Every test starts from the store as it was created. */
export function freshStorePerTest(): void {
  const initial = useSceneStore.getState();
  beforeEach(() => useSceneStore.setState(initial, true));
}
