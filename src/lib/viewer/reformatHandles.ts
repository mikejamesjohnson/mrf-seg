/**
 * The slice views' reformat handles in the 3D pane, like Slicer's: a vtk.js
 * `ImplicitPlaneWidget` per view whose handle is on, in its colour. Drag the rod to tilt the
 * plane about the sphere, the plane to move it along its normal, the sphere to move the point
 * it tilts about. Each change goes to the store (`onChange`), and everything follows from there.
 *
 * The widget is set *from* the store as well as writing to it, so a change applied from the
 * store must not echo back: hence `applying`, and the comparison with the plane last read.
 */

import type vtkRenderer from "@kitware/vtk.js/Rendering/Core/Renderer";
import vtkWidgetManager from "@kitware/vtk.js/Widgets/Core/WidgetManager";
import vtkImplicitPlaneWidget, {
  type ImplicitPlaneStyle,
  type vtkImplicitPlaneViewWidget,
} from "@kitware/vtk.js/Widgets/Widgets3D/ImplicitPlaneWidget";

import type { SceneState } from "@/lib/scene/store";
import { SLICE_COLORS, SLICE_ROLES, type SlicePlane, type SliceRole } from "@/lib/scene/views";
import type { Bounds, Vec3 } from "@/lib/world";

interface Handle {
  widget: ReturnType<typeof vtkImplicitPlaneWidget.newInstance>;
  view: vtkImplicitPlaneViewWidget;
  subscription: { unsubscribe: () => void } | null;
  bounds: Bounds | null;
  /** The view's plane as last read from the store. */
  plane: SlicePlane | null;
}

function differs(a: number[], b: number[]): boolean {
  return a.some((v, i) => Math.abs(v - b[i]) > 1e-9);
}

/**
 * The plane itself is invisible until the pointer is over it: the slice is already drawn
 * there, with a border in the view's colour. Rod and sphere are in that colour, lighter
 * while hovered or dragged.
 */
function styleFor(role: SliceRole): ImplicitPlaneStyle {
  const color = [...SLICE_COLORS[role]];
  const light = color.map((v) => v + (1 - v) * 0.5);
  return {
    active: {
      plane: { color, opacity: 0.2 },
      normal: { color: light, opacity: 1 },
      origin: { color: light, opacity: 1 },
    },
    inactive: {
      plane: { color, opacity: 0 },
      normal: { color, opacity: 1 },
      origin: { color, opacity: 1 },
    },
  };
}

export class ReformatHandles {
  private readonly manager = vtkWidgetManager.newInstance();
  private readonly handles = new Map<SliceRole, Handle>();
  private applying = false;

  constructor(
    renderer: vtkRenderer,
    private readonly onChange: (role: SliceRole, origin: Vec3, normal: Vec3) => void,
  ) {
    this.manager.setRenderer(renderer);
  }

  /** Whether the pointer is over a handle right now, so a click there is the handle's. */
  pointerOnHandle(): boolean {
    return this.handles.size > 0 && !!this.manager.getActiveWidget();
  }

  sync(state: SceneState): void {
    const bounds = state.worldBounds;
    for (const role of SLICE_ROLES) {
      const wanted = bounds !== null && state.reformatInThreeD[role];
      const existing = this.handles.get(role);
      if (!wanted) {
        if (existing) this.remove(role, existing);
        continue;
      }
      const handle = existing ?? this.create(role);
      const plane = state.slicePlanes[role];
      handle.plane = plane;

      this.applying = true;
      try {
        // Its box: the sphere cannot be dragged out of it, and the rod is as long as it.
        if (!handle.bounds || differs(handle.bounds, bounds)) {
          handle.widget.placeWidget(bounds);
          handle.bounds = [...bounds] as Bounds;
        }
        const widgetState = handle.widget.getWidgetState();
        if (differs(widgetState.getOrigin(), plane.origin)) widgetState.setOrigin(...plane.origin);
        if (differs(widgetState.getNormal(), plane.normal)) widgetState.setNormal(...plane.normal);
      } finally {
        this.applying = false;
      }
    }
  }

  private create(role: SliceRole): Handle {
    const widget = vtkImplicitPlaneWidget.newInstance();
    const view = this.manager.addWidget(widget) as vtkImplicitPlaneViewWidget;
    // The 3D pane already draws the volume's box.
    view.setOutlineVisible(false);
    view.setRepresentationStyle(styleFor(role));
    // Sphere 6% of the volume's smallest side across; rod a third of that, so both are
    // easy to grab without hiding the slice.
    view.setHandleSizeRatio(0.06);
    view.setAxisScale(0.3);

    const handle: Handle = { widget, view, subscription: null, bounds: null, plane: null };
    handle.subscription = widget.getWidgetState().onModified(() => {
      if (this.applying || !handle.plane) return;
      const origin = widget.getWidgetState().getOrigin();
      const normal = widget.getWidgetState().getNormal();
      // Hovering changes the state too; only a moved plane is news.
      if (!differs(origin, handle.plane.origin) && !differs(normal, handle.plane.normal)) return;
      this.onChange(role, origin as Vec3, normal as Vec3);
    });

    this.handles.set(role, handle);
    this.manager.enablePicking();
    return handle;
  }

  private remove(role: SliceRole, handle: Handle): void {
    handle.subscription?.unsubscribe();
    this.manager.removeWidget(handle.widget);
    handle.widget.delete();
    this.handles.delete(role);
  }

  destroy(): void {
    for (const [role, handle] of this.handles) this.remove(role, handle);
    this.manager.delete();
  }
}
