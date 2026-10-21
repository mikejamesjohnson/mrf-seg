/**
 * vtk.js ships `ImplicitPlaneWidget` without type declarations. These cover what
 * `viewer/reformatHandles.ts` uses, checked against the vtk.js 36 source.
 */
declare module "@kitware/vtk.js/Widgets/Widgets3D/ImplicitPlaneWidget" {
  import type vtkAbstractWidget from "@kitware/vtk.js/Widgets/Core/AbstractWidget";
  import type { vtkAbstractWidgetFactory } from "@kitware/vtk.js/Widgets/Core/AbstractWidgetFactory";
  import type vtkWidgetState from "@kitware/vtk.js/Widgets/Core/WidgetState";

  /** Where the widget's plane is. */
  export interface vtkImplicitPlaneWidgetState extends vtkWidgetState {
    getOrigin(): number[];
    setOrigin(x: number, y: number, z: number): boolean;
    getNormal(): number[];
    setNormal(x: number, y: number, z: number): boolean;
  }

  /** Per-pipeline styles: `plane`, `normal`, `origin`, each with `color` and `opacity`. */
  export interface ImplicitPlaneStyle {
    active?: Record<string, { color?: number[]; opacity?: number }>;
    inactive?: Record<string, { color?: number[]; opacity?: number }>;
  }

  /** The widget as drawn in one view. */
  export interface vtkImplicitPlaneViewWidget extends vtkAbstractWidget {
    setRepresentationStyle(style: ImplicitPlaneStyle): void;
    setOutlineVisible(visible: boolean): boolean;
    setPlaneVisible(visible: boolean): boolean;
    setHandleSizeRatio(ratio: number): boolean;
    setAxisScale(scale: number): boolean;
  }

  export interface vtkImplicitPlaneWidget extends vtkAbstractWidgetFactory<vtkImplicitPlaneViewWidget> {
    getWidgetState(): vtkImplicitPlaneWidgetState;
  }

  export function newInstance(initialValues?: object): vtkImplicitPlaneWidget;

  const vtkImplicitPlaneWidget: { newInstance: typeof newInstance };
  export default vtkImplicitPlaneWidget;
}
