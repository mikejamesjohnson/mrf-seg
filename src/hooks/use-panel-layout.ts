"use client";

import { useDefaultLayout, type LayoutStorage } from "react-resizable-panels";

/**
 * Omitting `storage` is not the same as having none: `useDefaultLayout` then reads
 * `localStorage` eagerly, which throws during server rendering.
 */
const NO_STORAGE: LayoutStorage = { getItem: () => null, setItem: () => {} };

/**
 * A resizable panel group's sizes, kept across reloads in `localStorage`: they belong to this
 * browser window, not to the scene, so not to the store either.
 */
export function usePanelLayout(id: string) {
  const { defaultLayout, onLayoutChanged } = useDefaultLayout({
    id,
    storage: typeof window === "undefined" ? NO_STORAGE : window.localStorage,
    // A window resize is not the user choosing a size.
    onlySaveAfterUserInteractions: true,
  });
  return { defaultLayout, onLayoutChanged };
}
