/**
 * Slicer's Shift-hover: while Shift is held and no button is down, every pointer move over a
 * slice pane reports where the pointer is, for the other slice views to follow. Pressing Shift
 * with the pointer already over the pane reports too, so the views follow without a wiggle. The
 * hover ends when Shift is released, a button goes down, the pointer leaves the pane or the
 * window loses focus.
 *
 * Only the hovering is here; what the position means is the caller's. A Shift-drag is not a
 * hover (it is the point layer's rectangle select), and neither is Shift pressed while typing.
 */

export interface ShiftHoverCallbacks {
  /** The pointer is here, in CSS pixels from the pane's top left, with Shift held. */
  hover: (x: number, y: number) => void;
  /** The hover is over. Called only after a `hover`, once per hover. */
  end: () => void;
}

/** Whether keyboard focus is in something that takes typing, where Shift means a capital letter. */
function typing(): boolean {
  const active = document.activeElement;
  return active instanceof HTMLElement && (active instanceof HTMLInputElement || active instanceof HTMLTextAreaElement || active.isContentEditable);
}

/**
 * Report Shift-hovers over `container`. Returns the unbind. The pointer is watched in the
 * capture phase, so a gesture that claims its events (a Shift-drag selecting points) is still
 * seen for where the pointer and its buttons are.
 */
export function bindShiftHover(container: HTMLElement, callbacks: ShiftHoverCallbacks): () => void {
  let hovered: [number, number] | null = null;
  let buttons = 0;
  let hovering = false;

  const report = () => {
    if (!hovered || buttons !== 0 || typing()) return;
    hovering = true;
    callbacks.hover(...hovered);
  };
  const end = () => {
    if (!hovering) return;
    hovering = false;
    callbacks.end();
  };
  const track = (event: PointerEvent) => {
    const box = container.getBoundingClientRect();
    hovered = [event.clientX - box.left, event.clientY - box.top];
    buttons = event.buttons;
  };
  const onPointerMove = (event: PointerEvent) => {
    track(event);
    if (event.shiftKey) report();
    else end();
  };
  const onPointerDown = (event: PointerEvent) => {
    track(event);
    end();
  };
  const onPointerLeave = () => {
    hovered = null;
    end();
  };
  const onKeyDown = (event: KeyboardEvent) => {
    if (event.key === "Shift" && !event.repeat) report();
  };
  const onKeyUp = (event: KeyboardEvent) => {
    if (event.key === "Shift") end();
  };

  const onPane: [string, EventListener][] = [
    ["pointermove", onPointerMove as EventListener],
    ["pointerdown", onPointerDown as EventListener],
    ["pointerup", track as EventListener],
    ["pointerleave", onPointerLeave],
    ["pointercancel", onPointerLeave],
  ];
  const onWindow: [string, EventListener][] = [
    ["keydown", onKeyDown as EventListener],
    ["keyup", onKeyUp as EventListener],
    ["blur", end],
  ];
  for (const [type, listener] of onPane) container.addEventListener(type, listener, true);
  for (const [type, listener] of onWindow) window.addEventListener(type, listener);
  return () => {
    for (const [type, listener] of onPane) container.removeEventListener(type, listener, true);
    for (const [type, listener] of onWindow) window.removeEventListener(type, listener);
    end();
  };
}
