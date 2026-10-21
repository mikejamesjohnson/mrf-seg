"use client";

import { useEffect } from "react";

import { useSceneStore } from "@/lib/scene/store";

/**
 * Keys for working with points, anywhere in the editor except while typing in a field:
 * 1 / 2 / 3 choose Navigate / Inside / Outside, Delete or Backspace removes the selected
 * points, Escape clears the selection.
 */
export function usePointShortcuts(): void {
  useEffect(() => {
    const onKeyDown = (event: KeyboardEvent) => {
      if (event.metaKey || event.ctrlKey || event.altKey || event.defaultPrevented) return;
      const target = event.target as HTMLElement | null;
      if (target?.closest("input, textarea, select, [contenteditable='true'], [role='menu']")) return;
      const store = useSceneStore.getState();
      switch (event.key) {
        case "Delete":
        case "Backspace":
          if (store.selectedPointIds.length > 0) {
            event.preventDefault();
            store.removePoints(store.selectedPointIds);
          }
          break;
        case "Escape":
          if (store.selectedPointIds.length > 0) store.selectPoints([], "replace");
          break;
        case "1":
          store.setPlaceMode("navigate");
          break;
        case "2":
          store.setPlaceMode("inside");
          break;
        case "3":
          store.setPlaceMode("outside");
          break;
      }
    };
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, []);
}
