/** What the mouse does to each pane's camera, the 3D pane's view directions, and its LPS cube. */

import vtkGestureCameraManipulator from "@kitware/vtk.js/Interaction/Manipulators/GestureCameraManipulator";
import vtkMouseCameraTrackballPanManipulator from "@kitware/vtk.js/Interaction/Manipulators/MouseCameraTrackballPanManipulator";
import vtkMouseCameraTrackballRotateManipulator from "@kitware/vtk.js/Interaction/Manipulators/MouseCameraTrackballRotateManipulator";
import vtkMouseCameraTrackballZoomManipulator from "@kitware/vtk.js/Interaction/Manipulators/MouseCameraTrackballZoomManipulator";
import vtkInteractorStyleManipulator from "@kitware/vtk.js/Interaction/Style/InteractorStyleManipulator";
import vtkAnnotatedCubeActor from "@kitware/vtk.js/Rendering/Core/AnnotatedCubeActor";

import type { Vec3 } from "@/lib/world";

export type ViewDirection = "oblique" | "anterior" | "posterior" | "left" | "right" | "superior" | "inferior";

/** Camera placements in LPS: `position` is the direction from the focal point to the camera. */
export const VIEW_DIRECTIONS: Record<ViewDirection, { position: Vec3; viewUp: Vec3 }> = {
  // The default: from an axis every slice plane is edge-on and the scene reads as flat.
  oblique: { position: [-0.62, -0.68, 0.39], viewUp: [0, 0, 1] },
  anterior: { position: [0, -1, 0], viewUp: [0, 0, 1] },
  posterior: { position: [0, 1, 0], viewUp: [0, 0, 1] },
  left: { position: [1, 0, 0], viewUp: [0, 0, 1] },
  right: { position: [-1, 0, 0], viewUp: [0, 0, 1] },
  superior: { position: [0, 0, 1], viewUp: [0, -1, 0] },
  inferior: { position: [0, 0, -1], viewUp: [0, 1, 0] },
};

/**
 * A slice pane is a true 2D view: every button pans and nothing rotates -- not even a
 * two-finger twist, which would roll the camera off its plane's screen up. Rotation is absent,
 * not suppressed; zoom is the wheel's (`wheel.ts`) or a pinch.
 */
export function panOnlyStyle(): vtkInteractorStyleManipulator {
  const style = vtkInteractorStyleManipulator.newInstance();
  style.removeAllManipulators();
  for (const button of [1, 2, 3]) style.addMouseManipulator(vtkMouseCameraTrackballPanManipulator.newInstance({ button }));
  style.addGestureManipulator(vtkGestureCameraManipulator.newInstance({ rotateEnabled: false }));
  return style;
}

/**
 * The 3D pane: rotate on left, pan on middle and right, zoom on the wheel. No shift-left pan:
 * Shift-drag selects points (`pointLayer.ts`).
 */
export function orbitStyle(): vtkInteractorStyleManipulator {
  const style = vtkInteractorStyleManipulator.newInstance();
  style.removeAllManipulators();
  // About the focal point, not the style's centre of rotation, which defaults to the world
  // origin -- often hundreds of mm from the patient.
  style.addMouseManipulator(vtkMouseCameraTrackballRotateManipulator.newInstance({ button: 1, useFocalPointAsCenterOfRotation: true }));
  style.addMouseManipulator(vtkMouseCameraTrackballPanManipulator.newInstance({ button: 2 }));
  style.addMouseManipulator(vtkMouseCameraTrackballPanManipulator.newInstance({ button: 3 }));
  style.addMouseManipulator(vtkMouseCameraTrackballZoomManipulator.newInstance({ scrollEnabled: true, dragEnabled: false }));
  style.addGestureManipulator(vtkGestureCameraManipulator.newInstance());
  return style;
}

/** The orientation cube, labelled for LPS: +x Left, +y Posterior, +z Superior. */
export function lpsCube(): vtkAnnotatedCubeActor {
  const cube = vtkAnnotatedCubeActor.newInstance();
  cube.setDefaultStyle({
    fontStyle: "bold",
    fontFamily: "system-ui, sans-serif",
    fontColor: "#0b0d12",
    faceColor: "#cbd5e1",
    edgeThickness: 0.08,
    edgeColor: "#0b0d12",
    resolution: 320,
  });
  cube.setXPlusFaceProperty({ text: "L" });
  cube.setXMinusFaceProperty({ text: "R" });
  cube.setYPlusFaceProperty({ text: "P" });
  cube.setYMinusFaceProperty({ text: "A" });
  cube.setZPlusFaceProperty({ text: "S" });
  cube.setZMinusFaceProperty({ text: "I" });
  return cube;
}
